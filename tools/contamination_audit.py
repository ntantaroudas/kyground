"""Cross-request contamination audit of session generation (protocol deviation D10).

A sub-agent answers all requests of a batch in one context, so a reply could use content that appears only in
another request of the same batch. For every generated request this script asks, per topic fact (gold nuggets of
an accuracy item, old and new values of a freshness edit, markers of a governance canary): does the reply contain
the fact, while the request's own text (system prompt, tools, conversation, retrieved context, tool results) does
not and another request of the same batch does? Such a fact is "borrowed".

Output: runs/main/contamination_flags.jsonl (one line per request with a borrowed fact) and
runs/main/contamination_audit.json (counts by experiment, condition and request kind).

    PYTHONPATH=src python tools/contamination_audit.py
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kyground.schema import Nugget, load_items  # noqa: E402
from kyground.scoring.leakage import leaks  # noqa: E402
from kyground.scoring.nuggets import match_nugget, score_answer  # noqa: E402

FACT_KINDS = {"text", "list_member", "number", "num_range", "duration_hours", "months"}
GOV_PHASE = {"visible": "G0_visible", "in_thread": "G3_in_thread",
             "fresh_before_refresh": "G2_revoked_before_refresh", "fresh_after_refresh": "G2_revoked_after_refresh"}
FRESH_PHASE = {"t1": "pre_edit", "t2": "in_thread_reask", "t3": "in_thread_challenge",
               "fresh_before_refresh": "fresh_before_refresh", "fresh_after_refresh": "fresh_after_refresh"}


MONTHS = ["Ιανουάριος January", "Φεβρουάριος February", "Μάρτιος March", "Απρίλιος April", "Μάιος May",
          "Ιούνιος June", "Ιούλιος July", "Αύγουστος August", "Σεπτέμβριος September", "Οκτώβριος October",
          "Νοέμβριος November", "Δεκέμβριος December"]
_NUM_FIELD = None


def with_units(text: str) -> str:
    """Tool results are JSON with bare numbers ("month": 7, "duration_min": 210); append them in words with units,
    so that the nugget matcher finds a value that the model saw as a number."""
    import re
    global _NUM_FIELD
    if _NUM_FIELD is None:
        _NUM_FIELD = re.compile(r'"(month|duration_hours|duration_min|max_people|lesson_count|price)"\s*:\s*"?(\d+(?:\.\d+)?)')
    extra = []
    for fld, v in _NUM_FIELD.findall(text):
        x = float(v)
        if fld == "month" and 1 <= x <= 12:
            extra.append(MONTHS[int(x) - 1])
        elif fld == "duration_hours":
            extra.append(f"{x:g} ώρες")
        elif fld == "duration_min":
            extra.append(f"{x:g} λεπτά")
        elif fld == "max_people":
            extra.append(f"{x:g} άτομα")
        elif fld == "lesson_count":
            extra.append(f"{x:g} μαθήματα")
        elif fld == "price":
            extra.append(f"{x:g} €")
    return text + ("\n" + "\n".join(extra) if extra else "")


def _msg_text(m: dict) -> str:
    parts = [str(m.get("content") or "")]
    for tc in m.get("tool_calls") or []:
        parts.append(json.dumps(tc, ensure_ascii=False))
    return "\n".join(parts)


def own_text(payload: dict) -> str:
    """Everything the model saw for this request (system prompt and tool definitions included)."""
    return with_units("\n".join([_msg_text(m) for m in payload["messages"]] +
                              [json.dumps(payload.get("tools") or [], ensure_ascii=False)]))


def private_text(payload: dict) -> str:
    """The part of the request that differs between requests of a batch (no system prompt, no tool definitions)."""
    return with_units("\n".join(_msg_text(m) for m in payload["messages"] if m.get("role") != "system"))


def reply_text(rep: dict, kind: str) -> str:
    t = rep.get("text") or ""
    if kind != "answer":
        t += "\n" + "\n".join(json.dumps(tc.get("arguments", tc), ensure_ascii=False) for tc in rep.get("tool_calls") or [])
    return t


class RecordMentions:
    """Which knowledge-base records a text names: by URL slug or by full title (normalised, Greek or English)."""

    def __init__(self, records: list[dict]):
        import re
        from kyground import textnorm as tn
        self.tn, self.re = tn, re
        self.ids = []
        for r in records:
            titles = r["title"].values() if isinstance(r["title"], dict) else [r["title"]]
            ts = sorted({tn.norm(t) for t in titles
                         if t and len(tn.norm(t)) >= 12 and len(tn.content_tokens(t, min_len=3)) >= 2})
            slug = r["slug"] if r.get("slug") and len(r["slug"]) >= 8 and r["rtype"] != "calendar" else None
            pat = re.compile(r"/" + re.escape(slug.lower()) + r"(?![\w-])") if slug else None
            if ts or pat:
                self.ids.append((r["rid"], ts, pat))

    def __call__(self, text: str) -> set[str]:
        n = self.tn.norm(text or "")
        low = (text or "").lower()
        return {rid for rid, ts, pat in self.ids if any(t in n for t in ts) or (pat is not None and pat.search(low))}


def topic_of(tag: str) -> tuple[str, str, dict]:
    p = tag.split("|")
    if p[0] == "acc":
        step = p[5] if len(p) > 5 else "answer"
        return "accuracy", p[3], {"condition": p[1], "label": p[2], "qid": p[3], "rep": p[4], "step": step}
    if p[0] == "fresh":
        step = p[5] if len(p) > 5 else "answer"
        return "freshness", p[2], {"variant": p[1], "eid": p[2], "lang": p[3], "phase": FRESH_PHASE.get(p[4], p[4]),
                                   "step": step}
    if p[0] == "gov":
        if p[1] == "G1":
            phase, rest = "G1_static", p[5:]
        else:
            phase, rest = GOV_PHASE.get(p[5], p[5]), p[6:]
        return "governance", p[3], {"variant": p[2], "cid": p[3], "lang": p[4], "phase": phase,
                                    "step": rest[0] if rest else "answer"}
    return "other", tag, {}


def main() -> None:
    bdir = ROOT / "runs/session/batches"
    payloads: dict[str, dict] = {}
    for f in sorted(glob.glob(str(ROOT / "runs/session/*.exported*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            e = json.loads(line)
            payloads[e["key"]] = e["payload"]
    cache = ROOT / "runs/cache/llm/claude-haiku-4-5"
    items = {i.iid: i for i in load_items(ROOT / "data/benchmark/v1.0/items.jsonl")}
    events = {e["eid"]: e for e in json.load(open(ROOT / "runs/main/freshness_events.json", encoding="utf-8"))}
    for e in events.values():
        e["_old"] = [Nugget(**n) for n in e["gold_old"]]
        e["_new"] = [Nugget(**n) for n in e["gold_new"]]
    canaries = {c["cid"]: c for c in json.load(open(ROOT / "runs/main/governance_canaries.json", encoding="utf-8"))}
    manifest = [json.loads(l) for l in open(bdir / "manifest.jsonl", encoding="utf-8")]
    records = [json.loads(l) for l in open(ROOT / "data/snapshots/public_20261005/records.jsonl", encoding="utf-8")]
    mentions = RecordMentions(records)

    flags, counts = [], Counter()
    for m in manifest:
        reqs = []
        for rid, key in m["ids"].items():
            pl = payloads.get(key)
            cf = cache / key[:2] / f"{key}.json"
            if pl is None or not cf.exists():
                counts["missing_payload_or_reply"] += 1
                continue
            reqs.append((rid, key, pl, json.loads(cf.read_text(encoding="utf-8"))))
        privs = [private_text(pl) for _, _, pl, _ in reqs]
        m_priv = [mentions(t) for t in privs]
        for i, (rid, key, pl, rep) in enumerate(reqs):
            exp, topic, info = topic_of(pl["tag"])
            own = own_text(pl)
            if exp == "accuracy" and topic.split("#")[0] in items:   # the question in its canonical forms, so
                q = items[topic.split("#")[0]].question             # that a transliterated question still counts
                own += "\n" + "\n".join(q.values() if isinstance(q, dict) else [str(q)])
            others = "\n".join(t for j, t in enumerate(privs) if j != i)
            reply = reply_text(rep, m["kind"])
            m_own = mentions(own)
            m_others = set().union(*[s for j, s in enumerate(m_priv) if j != i]) if len(m_priv) > 1 else set()
            foreign = sorted((mentions(reply) - m_own) & m_others)
            co, cx, cr = {}, {}, {}
            borrowed: list[str] = []
            outcome = ""
            if exp == "accuracy":
                it = items.get(topic.split("#")[0])
                if it is None:
                    continue
                for n in it.nuggets:
                    if n.required and n.kind in FACT_KINDS and match_nugget(n, reply, _cache=cr) \
                            and not match_nugget(n, own, _cache=co) and match_nugget(n, others, _cache=cx):
                        borrowed.append(n.nid)
                if borrowed and m["kind"] == "answer":
                    outcome = "correct" if score_answer(it, reply)["correct"] else "not_correct"
            elif exp == "freshness":
                ev = events[topic]
                for tagname, nugs in (("new", ev["_new"]), ("old", ev["_old"])):
                    if all(match_nugget(n, reply, _cache=cr) for n in nugs) and \
                            not all(match_nugget(n, own, _cache=co) for n in nugs) and \
                            all(match_nugget(n, others, _cache=cx) for n in nugs):
                        borrowed.append(tagname)
            elif exp == "governance":
                c = canaries[topic]
                for mk in c["markers"]:
                    hit = lambda t: bool(leaks(t, [mk])["leak"])  # noqa: E731
                    if hit(reply) and not hit(own) and hit(others):
                        borrowed.append(mk)
                for fld in ("hidden_price", "hidden_hours"):
                    v = c.get(fld)
                    if v is None:
                        continue
                    kw = {"hidden_price": v} if fld == "hidden_price" else {"hidden_hours": v}
                    hit = lambda t: bool(leaks(t, [], **kw)["leak"])  # noqa: E731
                    if hit(reply) and not hit(own) and hit(others):
                        borrowed.append(fld)
            grp = (exp, info.get("condition") or info.get("variant", ""), m["kind"])
            counts[("requests",) + grp] += 1
            if borrowed:
                counts[("borrowed_fact",) + grp] += 1
            if foreign:
                counts[("foreign_record",) + grp] += 1
            if borrowed or foreign:
                counts[("flagged",) + grp] += 1
                if exp == "accuracy" and m["kind"] == "answer" and not outcome:
                    it = items.get(topic.split("#")[0])
                    outcome = ("correct" if score_answer(it, reply)["correct"] else "not_correct") if it else ""
                flags.append({"batch": m["batch"], "key": key, "tag": pl["tag"], "kind": m["kind"], "experiment": exp,
                              **info, "borrowed": borrowed, "foreign_records": foreign, "outcome": outcome,
                              "reply": reply[:400]})
    out = ROOT / "runs/main"
    with open(out / "contamination_flags.jsonl", "w", encoding="utf-8") as f:
        for r in flags:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    table = defaultdict(dict)
    for k, v in counts.items():
        if isinstance(k, tuple):
            table["|".join(k[1:])][k[0]] = v
    summary = {"batches": len(manifest), "requests": sum(v.get("requests", 0) for v in table.values()),
               "flagged": len(flags), "by_group": dict(sorted(table.items())),
               "accuracy_answers_flagged_correct": sum(1 for r in flags if r["outcome"] == "correct"),
               "missing": counts["missing_payload_or_reply"]}
    (out / "contamination_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
