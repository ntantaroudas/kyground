"""E2 — answer freshness after editorial edits.

For each edit event and each system variant, on a private copy of the store:

1. turn 1 (thread A): ask the probe question before the edit (expected: old value)
2. apply the edit (logged mutation; never touches the platform)
3. turn 2 (thread A): ask the same question again           -> in-thread, re-ask
4. turn 3 (thread A): "Are you sure? Please check again."   -> in-thread, challenge
5. new thread B: ask the question                          -> fresh thread, before any batch re-index
6. batch re-index (variants with refresh policy ``manual``)
7. new thread C: ask the question                          -> fresh thread, after re-index
Each edit starts from a fresh copy of the store and a freshly built index.

Every answer is classified as updated / stale / mixed / neither against the
gold answers before and after the edit.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from ..kb.canaries import EditEvent, make_edit_events, old_new_state
from ..llm.session import PendingRequest
from ..schema import append_jsonl
from .factory import Env, config_hash

CHALLENGE = {"el": "Είσαι σίγουρος; Έλεγξε ξανά τα στοιχεία της πλατφόρμας.",
             "en": "Are you sure? Please check the platform's data again."}

# variant -> (condition, index refresh policy)
VARIANTS = {
    "C0": ("C0", None),
    "C1-batch": ("C1", "manual"),
    "C1-upsert": ("C1", "immediate_upsert"),
    "C1-append": ("C1", "immediate_append"),
    "C1q-batch": ("C1q", "manual"),
    "C2": ("C2", None),
    "C3-batch": ("C3", "manual"),
    "C4": ("C4", None),
}


def _run_variant(env: Env, variant: str, events: list[EditEvent], out: Path, langs: list[str],
                 done: set[tuple] | None = None) -> int:
    cond, policy = VARIANTS[variant]
    base = env.store()
    pending = 0
    for ev in tqdm(events, desc=f"freshness {variant}", leave=False, disable=env.quiet):
        for lang in langs:
            if done and (variant, ev.eid, lang) in done:
                continue
            # every edit starts from a fresh copy of the store and a freshly built index, so that results
            # cannot depend on the order of the edits (e.g. chunks accumulating in an append-only index)
            store = base.clone()
            index = env.index(store, policy=policy) if policy else None
            sysm = env.system(cond, store, index=index)
            q = ev.item_old.question[lang]
            tag = f"fresh|{variant}|{ev.eid}|{lang}"
            rows = []
            try:
                th = sysm.new_thread(lang)
                r1 = sysm.ask(th, q, cache_tag=tag + "|t1")
                rows.append(("pre_edit", 1, r1, q))
                store.apply_all(ev.forward)
                r2 = sysm.ask(th, q, cache_tag=tag + "|t2")
                rows.append(("in_thread_reask", 2, r2, q))
                r3 = sysm.ask(th, CHALLENGE[lang], cache_tag=tag + "|t3")
                rows.append(("in_thread_challenge", 3, r3, CHALLENGE[lang]))
                tb = sysm.new_thread(lang)
                r4 = sysm.ask(tb, q, cache_tag=tag + "|fresh_before_refresh")
                rows.append(("fresh_before_refresh", 1, r4, q))
                if policy == "manual":
                    index.refresh()
                tc = sysm.new_thread(lang)
                r5 = sysm.ask(tc, q, cache_tag=tag + "|fresh_after_refresh")
                rows.append(("fresh_after_refresh", 1, r5, q))
            except PendingRequest:      # session generation: the conversation continues in a later round
                pending += 1
                continue
            pre_old = None
            for phase, turn, res, asked in rows:
                st = old_new_state(res.answer, ev)
                if phase == "pre_edit":  # before the edit the old value is the correct one
                    st["state"] = "correct_pre" if st["has_old"] and not st["has_new"] else "incorrect_pre"
                    pre_old = int(st["state"] == "correct_pre")
                append_jsonl(out, {
                    "experiment": "freshness", "variant": variant, "condition": cond, "policy": policy,
                    "eid": ev.eid, "rid": ev.rid, "field": ev.field, "lang": lang, "phase": phase, "turn": turn,
                    "question": asked, "answer": res.answer, "tool_calls": res.tool_calls,
                    "retrieved": res.retrieved, "context_rids": res.context_rids, "usage": res.usage,
                    "latency_s": res.latency_s, "error": res.error, "pre_gave_old": pre_old, **st,
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return pending


def run_freshness(env: Env, run_dir: Path, variants: list[str], n_events: int = 40, langs: list[str] | None = None,
                  workers: int = 4, seed: int = 20261005) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    base = env.store()
    events = make_edit_events(base, n=n_events, seed=seed)
    (run_dir / "freshness_events.json").write_text(
        json.dumps([e.to_dict() for e in events], ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    out = run_dir / "freshness.jsonl"
    done: set[tuple] = set()
    if out.exists():
        from ..schema import read_jsonl
        done = {(r["variant"], r["eid"], r["lang"]) for r in read_jsonl(out)}
    meta = {"variants": variants, "n_events": len(events), "langs": langs or ["el"],
            "config_hash": config_hash(env.cfg), "kb_fingerprint": base.fingerprint(), "mock": env.mock}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        pend = list(ex.map(lambda v: _run_variant(env, v, events, out, langs or ["el"], done), variants))
    meta["pending_conversations"] = int(sum(pend))
    if meta["pending_conversations"]:
        print(f"freshness: {meta['pending_conversations']} conversations wait for generation")
    (run_dir / "meta_freshness.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def rescore_freshness(run_dir: Path) -> dict:
    """Classify every stored freshness answer again with the current scorer (updated / stale / mixed / neither,
    and whether the first answer gave the old value), from the edit events saved with the run. Used after a
    scorer correction; the answers themselves are not changed."""
    from types import SimpleNamespace
    from ..schema import Nugget, read_jsonl, write_jsonl
    path = run_dir / "freshness.jsonl"
    events = {e["eid"]: e for e in json.loads((run_dir / "freshness_events.json").read_text(encoding="utf-8"))}

    def item(e: dict, which: str) -> SimpleNamespace:      # old_new_state needs only the nuggets
        return SimpleNamespace(nuggets=[Nugget(**n) for n in e[f"gold_{which}"]])

    rows = list(read_jsonl(path))
    changed = 0
    pre = {}
    for r in rows:                       # first pass: the pre-edit answer of every conversation
        if r["phase"] == "pre_edit":
            e = events[r["eid"]]
            ev = EditEvent(eid=e["eid"], rid=e["rid"], field=e["field"], forward=[], backward=[],
                           item_old=item(e, "old"), item_new=item(e, "new"), old_value=e["old"], new_value=e["new"])
            st = old_new_state(r.get("answer") or "", ev)
            pre[(r["variant"], r["eid"], r["lang"])] = int(bool(st["has_old"]) and not st["has_new"])
    for r in rows:
        e = events[r["eid"]]
        ev = EditEvent(eid=e["eid"], rid=e["rid"], field=e["field"], forward=[], backward=[],
                       item_old=item(e, "old"), item_new=item(e, "new"), old_value=e["old"], new_value=e["new"])
        st = old_new_state(r.get("answer") or "", ev)
        if r["phase"] == "pre_edit":
            st["state"] = "correct_pre" if st["has_old"] and not st["has_new"] else "incorrect_pre"
        st["pre_gave_old"] = pre.get((r["variant"], r["eid"], r["lang"]))
        if any(r.get(k) != v for k, v in st.items()):
            changed += 1
        r.update(st)
    write_jsonl(path, rows)
    return {"rows": len(rows), "changed": changed}
