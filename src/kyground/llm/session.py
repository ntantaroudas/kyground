"""Model calls answered by Claude sub-agents inside a Claude session, through the request cache.

No API key is used. Every model call of an experiment goes through :meth:`LLMClient.complete`, which looks the
request up in the on-disk cache. A :class:`SessionClient` answers a cache miss by recording the request in a queue
(``session.queue`` in ``configs/study.yaml``) and raising :class:`PendingRequest`; the experiment runners then
leave that query or conversation for later. A round of generation is:

1. run the experiments: everything already answered is replayed from the cache, and the next request of every
   unfinished query or conversation is queued;
2. ``kyground gen-export``: the queued requests are written as batches - one ``batch_NNN.md`` per batch, with the
   system prompt (and tool definitions) shared by the batch, then the requests, each a separate conversation;
3. one sub-agent per batch (Claude Haiku) writes ``batch_NNN.out.jsonl`` with one reply per request;
4. ``kyground gen-import`` validates the replies and stores them in the cache under each request's key.

Repeating the round until nothing is queued completes the experiments; replies are then fixed in the cache, so the
analysis can be repeated without new generation. Batches hold requests of one kind (answer, router or query
rewrite) with the same system prompt and tools, at most one request per item cluster (or edit, or canary), and a
bounded size, so that no request can draw on another request's context about the same record.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

from .base import LLMClient, LLMResult, ToolCall

KINDS = ("answer", "router", "rewrite")
LONG_SYSTEM = 20000     # characters; a longer system prompt holds the knowledge base


class PendingRequest(Exception):
    """A model request that has not been answered yet; the runners skip the query or conversation for now."""


class SessionClient(LLMClient):
    provider = "session"
    deferred = True
    _qlock = threading.Lock()

    def __init__(self, model: str, cache_dir: str | Path | None = None, queue: str | Path | None = None,
                 role: str = "", **defaults: Any) -> None:
        super().__init__(model, cache_dir=cache_dir, **defaults)
        self.queue = Path(queue) if queue else None
        self.role = role
        self._seen: set[str] = set()

    def defer(self, key: str, payload: dict, cache_path: Path | None) -> None:
        if self.queue is None or cache_path is None:
            raise RuntimeError("session generation needs session.queue and a cache directory")
        with self._qlock:
            if key not in self._seen:
                self._seen.add(key)
                self.queue.parent.mkdir(parents=True, exist_ok=True)
                with self.queue.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"key": key, "cache_path": str(cache_path), "role": self.role,
                                         "payload": payload}, ensure_ascii=False, default=str) + "\n")
        raise PendingRequest(key)

    def _call(self, messages, *, tools, params) -> LLMResult:  # pragma: no cover - never reached (deferred)
        raise PendingRequest("session client cannot call a model")


# ---------------------------------------------------------------------------------------------- batching
def request_kind(payload: dict) -> str:
    tc = (payload.get("params") or {}).get("tool_choice")
    if payload.get("tools") and tc == "required":
        return "rewrite"
    if payload.get("tools") and tc not in ("none",):
        return "router"
    return "answer"


def _topic(tag: str, clusters: dict[str, str]) -> str:
    """The item cluster, edit or canary a request is about (from its cache tag)."""
    parts = tag.split("|")
    if parts and parts[0] == "acc" and len(parts) > 3:
        qid = parts[3]
        iid = qid.split("#")[0]
        return "item:" + clusters.get(iid, iid)
    if parts and parts[0] == "fresh" and len(parts) > 2:
        return "edit:" + parts[2]
    if parts and parts[0] == "gov" and len(parts) > 3:
        return "canary:" + parts[3]
    return "tag:" + tag


def _size(payload: dict) -> int:
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in payload["messages"] if m.get("role") != "system")


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


HEADER = {
    "answer": """You are acting as the ANSWER MODEL of a chat assistant, for a research study. Each request below is a
separate, new conversation between a user and that assistant. Every conversation starts with the SYSTEM PROMPT
below (the same for all requests in this batch). Each request shows the conversation so far: user turns, earlier
assistant replies and, where present, the tool calls the assistant made in this turn and the tool results they
returned. Write the assistant's next reply to the last user message, exactly as the assistant described by the
system prompt would, in the language the system prompt asks for. Treat every request on its own: never use
anything from another request. Do not call tools: tool results, where present, are already shown. Do not look
anything up, do not open other files and do not write code.""",
    "router": """You are acting as the ROUTER MODEL of a chat assistant, for a research study. Each request below is a
separate conversation that starts with the SYSTEM PROMPT below (the same for all requests in this batch). Your job
is what the router does through function calling: decide which of the TOOLS below to call, and with which
arguments, to answer the last user message - or call no tool. Tool calls already made in the current turn, and
their results, may already be shown; in total the router may call at most two tools per user turn, so call at
most as many as remain (none if two were already called, or if the results shown are enough). Arguments must
follow the tool's parameter schema.
Treat every request on its own: never use anything from another request. Do not answer the user, do not look
anything up, do not open other files and do not write code.""",
    "rewrite": """You are acting as the ROUTER MODEL of a chat assistant, for a research study. Each request below is a
separate conversation that starts with the SYSTEM PROMPT below (the same for all requests in this batch). The
router must call the single TOOL below exactly once (function calling with tool_choice "required"), filling its
arguments to retrieve what is needed to answer the last user message. Treat every request on its own: never use
anything from another request. Do not answer the user, do not look anything up, do not open other files and do
not write code.""",
}

CHECK_NOTE = ("Lines of the form \u27e6check N: CODE\u27e7 appear at intervals in this file. They are not part of the "
              "content: ignore them, except that you must report every check code, as described under OUTPUT.")

OUTPUT = {
    "answer": '{"id": "<request id>", "text": "<the assistant\'s reply>"}',
    "router": '{"id": "<request id>", "tool_calls": [{"name": "<tool name>", "arguments": {...}}], "text": ""}'
              '  (use "tool_calls": [] to call no tool)',
    "rewrite": '{"id": "<request id>", "tool_calls": [{"name": "<tool name>", "arguments": {...}}], "text": ""}',
}


def wrap(text: str, width: int = 1000) -> str:
    """Break lines longer than ``width`` at spaces (never inside a word), keeping the text otherwise unchanged."""
    out = []
    for line in (text or "").split("\n"):
        while len(line) > width:
            cut = line.rfind(" ", 0, width)
            cut = cut if cut > width // 2 else width
            out.append(line[:cut])
            line = line[cut:].lstrip(" ")
        out.append(line)
    return "\n".join(out)


def _pretty(content: str) -> str:
    try:
        return json.dumps(json.loads(content), ensure_ascii=False, indent=1)
    except (json.JSONDecodeError, TypeError):
        return content or ""


def render_messages(msgs: list[dict]) -> str:
    """The conversation as readable text: user and assistant turns, tool calls and tool results."""
    names: dict[str, str] = {}
    parts = []
    for m in msgs:
        role = m.get("role")
        if role == "user":
            parts += ["**user:**", wrap(m.get("content") or ""), ""]
        elif role == "assistant":
            calls = m.get("tool_calls") or []
            if calls:
                parts.append("**assistant - tool calls:**")
                for c in calls:
                    f = c.get("function", {})
                    names[c.get("id", "")] = f.get("name", "")
                    parts.append(f"- {f.get('name')} {f.get('arguments')}")
                parts.append("")
            if m.get("content"):
                parts += ["**assistant:**", wrap(m["content"]), ""]
        elif role == "tool":
            parts += [f"**tool result ({names.get(m.get('tool_call_id', ''), 'tool')}):**",
                      wrap(_pretty(m.get("content") or "")), ""]
    return "\n".join(parts).rstrip()


CHECK_EVERY = 100     # lines between check codes


def add_checks(text: str, batch: str, seed: int = 20261005, every: int = CHECK_EVERY) -> tuple[str, list[str]]:
    """Insert a check line after every ``every`` lines (and one at the end). A sub-agent must report all codes, which
    shows that it read the whole batch file; a batch whose codes are missing or wrong is generated again."""
    lines = text.split("\n")
    out, codes = [], []
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

    def code(i: int) -> str:
        h = hashlib.sha256(f"{seed}|{batch}|{i}".encode()).digest()
        return "".join(alphabet[b % len(alphabet)] for b in h[:4])

    end = lines.index("## END OF REQUESTS") if "## END OF REQUESTS" in lines else len(lines)
    for i, line in enumerate(lines, start=1):
        if i == end + 1:      # the last code comes just before the end marker, after the last request
            codes.append(code(len(codes) + 1))
            out.append(f"\u27e6check {len(codes)}: {codes[-1]}\u27e7")
        out.append(line)
        if i % every == 0 and i < end:
            codes.append(code(len(codes) + 1))
            out.append(f"\u27e6check {len(codes)}: {codes[-1]}\u27e7")
    if end == len(lines):
        codes.append(code(len(codes) + 1))
        out.append(f"\u27e6check {len(codes)}: {codes[-1]}\u27e7")
    return "\n".join(out), codes


def _render_tools(tools: list[dict]) -> str:
    out = []
    for t in tools or []:
        f = t.get("function", t)
        out.append(json.dumps({"name": f.get("name"), "description": f.get("description"),
                               "parameters": f.get("parameters")}, ensure_ascii=False, indent=1))
    return "\n\n".join(out)


def _manifest(out_dir: Path) -> list[dict]:
    mp = out_dir / "manifest.jsonl"
    return [json.loads(l) for l in mp.read_text(encoding="utf-8").splitlines() if l.strip()] if mp.exists() else []


def export_queue(queue: Path, out_dir: Path, clusters: dict[str, str], seed: int = 20261005,
                 max_chars: int = 45000, max_requests: dict[str, int] | None = None, tokenizer=None) -> dict:
    """Write the queued, still unanswered requests as batches; the queue is moved aside once exported. Requests
    already in an exported batch are not exported again (an incomplete batch is generated again instead)."""
    max_requests = {"answer": 20, "router": 40, "rewrite": 40, "long_system": 5, **(max_requests or {})}
    exported = {k for m in _manifest(out_dir) for k in m["ids"].values()}
    pending: dict[str, dict] = {}
    if queue.exists():
        for line in queue.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                if not Path(e["cache_path"]).exists() and e["key"] not in exported:
                    pending[e["key"]] = e
    out_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for e in pending.values():
        p = e["payload"]
        system = next((m.get("content") or "" for m in p["messages"] if m.get("role") == "system"), "")
        kind = request_kind(p)
        groups[(kind, p["model"], _h(system), _h(json.dumps(p.get("tools"), sort_keys=True, ensure_ascii=False)))].append(e)
    rng = random.Random(seed)
    manifest, nb = [], 0
    existing = list(out_dir.glob("batch_*.md")) + list(out_dir.glob("_superseded/batch_*.md"))
    if existing:     # numbers are never reused, also not those of withdrawn batches
        nb = max(int(re.findall(r"\d+", b.stem)[0]) for b in existing)
    for gkey in sorted(groups):
        reqs = groups[gkey]
        rng.shuffle(reqs)
        # requests of the most frequent topics first (first-fit decreasing): they need the most batches, and the
        # others fill those batches instead of opening small ones at the end
        freq = defaultdict(int)
        for e in reqs:
            freq[_topic(e["payload"].get("tag", ""), clusters)] += 1
        reqs.sort(key=lambda e: -freq[_topic(e["payload"].get("tag", ""), clusters)])
        kind = gkey[0]
        sys0 = next((m.get("content") or "" for m in reqs[0]["payload"]["messages"] if m.get("role") == "system"), "")
        # a system prompt holding the whole knowledge base (C4): few requests per batch, so that each gets the
        # attention a single call would give it (a pilot batch of 20 requests ignored records in the prompt)
        cap = max_requests["long_system"] if len(sys0) > LONG_SYSTEM else max_requests[kind]
        batches: list[list[dict]] = []
        for e in reqs:   # first batch with room and no request about the same topic
            topic = _topic(e["payload"].get("tag", ""), clusters)
            size = _size(e["payload"])
            for b in batches:
                if len(b) < cap and sum(x["_size"] for x in b) + size <= max_chars and \
                        topic not in {x["_topic"] for x in b}:
                    b.append({**e, "_topic": topic, "_size": size})
                    break
            else:
                batches.append([{**e, "_topic": topic, "_size": size}])
        for b in batches:
            nb += 1
            first = b[0]["payload"]
            system = next((m.get("content") or "" for m in first["messages"] if m.get("role") == "system"), "")
            parts = [f"# Batch {nb:03d} ({kind}, {len(b)} requests)", "", HEADER[kind], "", CHECK_NOTE, "",
                     "## SYSTEM PROMPT", "", wrap(system.strip())]
            if kind in ("router", "rewrite"):
                parts += ["", "## TOOLS", "", _render_tools(first.get("tools"))]
            parts += ["", "## OUTPUT", "",
                      f"Write {len(b) + 1} lines to the output file named in your instructions and nothing else. The "
                      f"first line lists the check codes you read in this file, in order: "
                      f'{{"checks": ["<code>", "<code>", ...]}}. Then one JSON object per request, in the order of '
                      f"the requests:", "", OUTPUT[kind], "", "## REQUESTS", ""]
            in_tok = {}
            for i, e in enumerate(b, start=1):
                rid = e["key"][:16]
                msgs = [m for m in e["payload"]["messages"] if m.get("role") != "system"]
                parts += [f"### Request {i} of {len(b)} - id {rid}", "", render_messages(msgs), ""]
                if tokenizer is not None:   # what this request would cost as a separate call
                    blob = system + json.dumps(msgs, ensure_ascii=False) + (
                        json.dumps(e["payload"].get("tools"), ensure_ascii=False) if e["payload"].get("tools") else "")
                    in_tok[rid] = len(tokenizer.encode(blob))
            parts += ["## END OF REQUESTS", "", f"Now write the check line and the {len(b)} output lines as described "
                                                 f"under OUTPUT."]
            text, checks = add_checks("\n".join(parts), f"batch_{nb:03d}", seed)
            (out_dir / f"batch_{nb:03d}.md").write_text(text, encoding="utf-8")
            manifest.append({"batch": f"batch_{nb:03d}", "kind": kind, "model": first["model"],
                             "ids": {e["key"][:16]: e["key"] for e in b},
                             "cache_paths": {e["key"][:16]: e["cache_path"] for e in b},
                             "in_tokens": in_tok, "chars": len(text), "lines": text.count("\n") + 1,
                             "checks": checks})
    with (out_dir / "manifest.jsonl").open("a", encoding="utf-8") as fh:
        for m in manifest:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    if queue.exists():
        queue.rename(queue.with_name(queue.stem + f".exported{nb:03d}.jsonl"))
    return {"requests": len(pending), "batches": len(manifest),
            "by_kind": {k: sum(len(m["ids"]) for m in manifest if m["kind"] == k) for k in KINDS},
            "first": manifest[0]["batch"] if manifest else None, "last": manifest[-1]["batch"] if manifest else None}


def supersede(out_dir: Path, batches: list[str], reason: str) -> int:
    """Withdraw exported batches whose replies are not stored: their files move to ``_superseded/`` and their
    requests are queued again by the next run of the experiments (replies already stored are kept)."""
    keep, gone = [], []
    for m in _manifest(out_dir):
        (gone if m["batch"] in batches else keep).append(m)
    sdir = out_dir / "_superseded"
    sdir.mkdir(exist_ok=True)
    for m in gone:
        for f in (out_dir / f"{m['batch']}.md", out_dir / f"{m['batch']}.out.jsonl"):
            if f.exists():
                f.rename(sdir / f.name)
        with (sdir / "manifest_superseded.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({**m, "reason": reason}, ensure_ascii=False) + "\n")
    (out_dir / "manifest.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in keep),
                                            encoding="utf-8")
    return len(gone)


def open_batches(out_dir: Path) -> list[dict]:
    """Batches with replies still missing from the cache (output file absent or incomplete)."""
    rows = []
    for m in _manifest(out_dir):
        missing = [rid for rid, cp in m["cache_paths"].items() if not Path(cp).exists()]
        if missing:
            rows.append({"batch": m["batch"], "kind": m["kind"], "requests": len(m["ids"]), "missing": len(missing),
                         "chars": m.get("chars"), "lines": m.get("lines"),
                         "output": (out_dir / f"{m['batch']}.out.jsonl").exists()})
    return rows


GEN_SUBAGENT_PROMPT = """You are taking part in a research study (KyGround) as the language model behind a chat assistant. Follow these steps exactly.

1. Read the whole batch file {batch} ({lines} lines) with the Read tool. If a read shows only part of the file, read the rest with the offset the tool suggests, until you have read the line "## END OF REQUESTS".
2. Do what the file says. For each request, in order, produce the output described under OUTPUT, acting as the model described at the top of the file. Treat every request on its own. Do not write or run code, do not search the web, and do not open, list or read any other file.
3. Use the Write tool once to write the output file {out}: first the line with all check codes, then the {n} request lines - one JSON object per line (valid JSON: inside strings, escape double quotes as \\" and line breaks as \\n), nothing else in the file.
4. Reply with a single line: the number of lines you wrote, and your model identifier as stated in your own instructions (or "unknown").
"""


def subagent_prompt(out_dir: Path, batch: str) -> str:
    m = next(x for x in _manifest(out_dir) if x["batch"] == batch)
    return GEN_SUBAGENT_PROMPT.format(batch=out_dir / f"{batch}.md", out=out_dir / f"{batch}.out.jsonl",
                                      lines=m.get("lines", "?"), n=len(m["ids"]))


def _parse_lines(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith("```"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            out.append({"_bad": line[:200]})
    return out


def import_outputs(out_dir: Path, model_label: str, tokenizer=None) -> dict:
    """Store every valid reply in the cache; report batches that are missing or incomplete."""
    manifest = _manifest(out_dir)
    stored, redo = 0, {}
    for m in manifest:
        outp = out_dir / f"{m['batch']}.out.jsonl"
        missing_cache = [rid for rid, cp in m["cache_paths"].items() if not Path(cp).exists()]
        if not missing_cache:
            continue
        if not outp.exists():
            redo[m["batch"]] = "no output file"
            continue
        problems = []
        got = {}
        lines = _parse_lines(outp)
        if m.get("checks"):
            head = lines[0] if lines and isinstance(lines[0], dict) and "checks" in lines[0] else None
            if head is None or [str(c).strip().upper() for c in head.get("checks") or []] != m["checks"]:
                redo[m["batch"]] = [f"check codes missing or wrong: {head.get('checks') if head else None}"]
                continue      # the agent did not read the whole file: nothing from this batch is stored
            lines = lines[1:]
        for d in lines:
            rid = str(d.get("id", ""))[:16]
            if "_bad" in d or rid not in m["ids"]:
                problems.append(f"bad line: {str(d)[:80]}")
                continue
            if m["kind"] == "answer":
                for alias in ("response", "answer", "reply"):     # a misnamed key still holds the reply
                    if not isinstance(d.get("text"), str) and isinstance(d.get(alias), str):
                        d["text"] = d[alias]
                if not isinstance(d.get("text"), str):
                    problems.append(f"{rid}: no text")
                    continue
            else:
                calls = d.get("tool_calls") or []
                if not isinstance(calls, list) or any(not isinstance(c, dict) or not c.get("name") for c in calls):
                    problems.append(f"{rid}: malformed tool_calls")
                    continue
                if m["kind"] == "rewrite" and len(calls) != 1:
                    problems.append(f"{rid}: rewrite must call exactly one tool")
                    continue
            got[rid] = d
        for rid in missing_cache:
            d = got.get(rid)
            if d is None:
                problems.append(f"{rid}: not answered")
                continue
            calls = [ToolCall(id=f"call_{m['ids'][rid][:8]}_{i}", name=str(c["name"]),
                              arguments=c.get("arguments") if isinstance(c.get("arguments"), dict) else {},
                              raw=json.dumps(c.get("arguments", {}), ensure_ascii=False))
                     for i, c in enumerate(d.get("tool_calls") or [])] if m["kind"] != "answer" else []
            text = d.get("text") or ""
            usage = {}
            if rid in (m.get("in_tokens") or {}):
                usage["input_tokens"] = int(m["in_tokens"][rid])
            if tokenizer is not None:
                usage["output_tokens"] = len(tokenizer.encode(text + (json.dumps(d.get("tool_calls"), ensure_ascii=False)
                                                                      if d.get("tool_calls") else "")))
            res = LLMResult(text=text, tool_calls=calls, model=model_label, provider="session", usage=usage,
                            finish_reason="tool_calls" if calls else "stop")
            cp = Path(m["cache_paths"][rid])
            cp.parent.mkdir(parents=True, exist_ok=True)
            tmp = cp.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(res.to_dict(), ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, cp)
            stored += 1
        if problems:
            redo[m["batch"]] = problems[:5] + ([f"... {len(problems) - 5} more"] if len(problems) > 5 else [])
    return {"stored": stored, "batches_to_redo": redo}
