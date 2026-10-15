"""E1 — accuracy and robustness, and E4 — search-normalisation ablation.

For every condition, every query (item x surface form) and every repetition,
a fresh conversation is opened and the question is asked once. Raw rows are
appended to ``<run>/accuracy.jsonl`` as they complete, so an interrupted run
resumes where it stopped (model calls are also cached on disk).
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from ..schema import Item, Query, append_jsonl, load_items, load_queries, read_jsonl, write_jsonl
from ..llm.session import PendingRequest
from ..scoring.nuggets import score_answer
from .factory import Env, config_hash, rpath

_lock = threading.Lock()


def _done_keys(path: Path) -> set[tuple]:
    return {(r["condition"], r["qid"], r.get("rep", 0)) for r in read_jsonl(path)} if path.exists() else set()


def select_queries(items: list[Item], queries: list[Query], split: str | None, forms: Iterable[str] | None,
                   statuses: Iterable[str] | None = None, limit: int | None = None) -> list[Query]:
    keep = {i.iid for i in items if (split in (None, "all") or i.split == split)
            and (not statuses or i.status in set(statuses))}
    fs = set(forms) if forms else None
    out = [q for q in queries if q.iid in keep and (fs is None or q.form in fs)]
    return out[:limit] if limit else out


def run_accuracy(env: Env, run_dir: Path, conditions: list[str], *, split: str | None = "test",
                 forms: Iterable[str] | None = None, reps: int = 1, workers: int = 8, limit: int | None = None,
                 backend_kind: str = "replica", search_mode: str | None = None, answer_role: str = "answer",
                 label: str = "", force: bool = False, part: str = "") -> Path:
    cfg = env.cfg
    if answer_role != "answer" and not label:
        label = answer_role  # never mix another answer model's rows into the main conditions
    bdir = rpath(cfg["paths"]["benchmark_dir"])
    items = load_items(bdir / "items.jsonl")
    queries = load_queries(bdir / "queries.jsonl")
    statuses = None if env.mock else cfg.get("accuracy", {}).get("item_statuses")
    qs = select_queries(items, queries, split, forms, statuses, limit)
    if not qs:
        raise RuntimeError(f"no queries selected (split={split}, statuses={statuses}); real runs use only items "
                           "that passed the automatic gold verification - run `kyground verify-gold` first")
    store = env.store()
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "accuracy.jsonl"
    meta_path = run_dir / f"meta_accuracy{('_' + label) if label else ''}{('_' + part) if part else ''}.json"
    chash = config_hash(cfg)
    if meta_path.exists() and out.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("config_hash") not in (None, chash) and not force:
            raise RuntimeError(f"{run_dir} was started with a different configuration ({old.get('config_hash')} vs "
                               f"{chash}); use a new --run-id, or --force to resume anyway")
    done = _done_keys(out)
    from ..kb.serialize import kb_text
    pub = store.public()
    meta = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "conditions": conditions,
            "n_queries": len(qs), "reps": reps, "split": split, "backend": backend_kind, "search_mode": search_mode,
            "answer_role": answer_role, "answer_model": cfg["models"].get(answer_role, {}).get("model"),
            "kb_fingerprint": store.fingerprint(), "snapshot": str(cfg["paths"]["snapshot"]),
            "kb_records": len(pub), "kb_chars": len(kb_text(pub)), "replica": cfg.get("replica", {}),
            "retrieval": cfg.get("retrieval", {}), "item_statuses": statuses, "config_hash": chash,
            "mock": env.mock, "label": label}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    systems = {}
    shared_index = None
    for c in conditions:
        backend = None
        if c in ("C2", "C3"):
            backend = env.live_backend() if backend_kind == "live" else env.replica_backend(store, search_mode)
        if c in ("C1", "C1q", "C3") and shared_index is None:
            shared_index = env.index(store)
        systems[c] = env.system(c, store, backend=backend, index=shared_index, answer_role=answer_role)

    tasks = [(c, q, rep) for rep in range(reps) for c in conditions for q in qs
             if (c + (f"@{label}" if label else ""), q.qid, rep) not in done]
    if shared_index is not None and tasks:     # query embeddings once, before the threads start
        shared_index.embedder.embed(sorted({q.text for _c, q, _r in tasks}), kind="query")
    # the search-normalisation ablation (E4) changes only the tool results: requests whose messages are identical to
    # the main run's (same tool results) are the same request and reuse its output; the others are generated anew
    tag_label = "" if label.startswith("search=") else label

    def work(t):
        c, q, rep = t
        sysm = systems[c]
        th = sysm.new_thread(q.lang)
        tag = f"acc|{c}|{tag_label}|{q.qid}|rep{rep}"
        try:
            res = sysm.ask(th, q.text, cache_tag=tag)
        except PendingRequest:      # session generation: answered in a later round
            return None
        # the run's identifiers come last so that the turn result cannot overwrite them (e.g. its
        # unlabelled ``condition``, which would merge ablation or open-model rows into the main conditions)
        row = {**res.to_dict(), "experiment": "accuracy", "condition": c + (f"@{label}" if label else ""),
               "base_condition": c, "label": label, "qid": q.qid, "iid": q.iid, "form": q.form, "lang": q.lang,
               "rep": rep, "question": q.text, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with _lock:
            append_jsonl(out, row)
        return row

    t0 = time.time()
    pending = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for f in tqdm(as_completed(futs), total=len(futs), desc=f"accuracy {','.join(conditions)} {label}",
                      disable=env.quiet):
            pending += f.result() is None
    meta["pending"] = pending
    if pending:
        print(f"accuracy {','.join(conditions)} {label}: {pending} of {len(tasks)} queries wait for generation")
    meta["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["wall_s"] = round(time.time() - t0, 1)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def score_accuracy(run_dir: Path, items_path: Path) -> Path:
    """Deterministic scoring + retrieval diagnostics -> accuracy_scored.jsonl"""
    items = {i.iid: i for i in load_items(items_path)}
    rows = []
    for r in read_jsonl(run_dir / "accuracy.jsonl"):
        it = items.get(r["iid"])
        if it is None:
            continue
        if r.get("label") and "@" not in str(r.get("condition", "")):  # rows written before the fix above
            r["condition"] = f"{r.get('base_condition') or r['condition']}@{r['label']}"
        s = score_answer(it, r.get("answer", ""))
        gold = set(it.gold_rids)
        ctx = set(r.get("context_rids") or [])
        diag = {
            "qtype": it.qtype, "rtype": it.rtype, "template": it.template, "cluster": it.cluster,
            "split": it.split, "origin": it.origin,
            "gold_in_context": int(bool(gold & ctx)) if gold else None,
            "n_tool_calls": len(r.get("tool_calls") or []),
            "tool_names": [t["name"] for t in (r.get("tool_calls") or [])],
            "tool_empty": int(any(t.get("n_results", 0) == 0 for t in (r.get("tool_calls") or []))),
        }
        rows.append({**r, **s, **diag})
    out = run_dir / "accuracy_scored.jsonl"
    write_jsonl(out, rows)
    return out
