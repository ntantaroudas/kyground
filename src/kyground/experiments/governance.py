"""E3 — governance: leakage of gated content.

Phases per canary and system variant (private store copy; synthetic records):

``G1_static``        canary gated from the start (index built after gating):
                     fresh-thread probes. Server-side gating should prevent
                     any leak; the ``C1-raw`` variant indexes the raw store
                     (a misconfigured indexer) to show what gating buys.
``G2_revoked``       canary public and indexed, then revoked (unpublished,
                     deactivated or price hidden): fresh-thread probe before
                     and after the batch re-index.
``G3_in_thread``     same revocation, but the question is re-asked in the
                     conversation where the content had been shown.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from ..kb.canaries import Canary, insert_canaries, make_canaries
from ..llm.session import PendingRequest
from ..schema import append_jsonl
from ..scoring.leakage import leaks
from .factory import Env, config_hash

VARIANTS = {
    "C0": ("C0", None, "public"),
    "C1-batch": ("C1", "manual", "public"),
    "C1-upsert": ("C1", "immediate_upsert", "public"),
    "C1-raw": ("C1", "never", "raw"),
    "C2": ("C2", None, "public"),
    "C3-batch": ("C3", "manual", "public"),
    "C4": ("C4", None, "public"),
}


def _row(out: Path, variant: str, cond: str, policy, c: Canary, phase: str, lang: str, q: str, res) -> None:
    lk = leaks(res.answer, c.markers, c.hidden_price, question=q, hidden_hours=c.hidden_hours)
    # model-independent exposure: did the gated record reach the model's context?
    # (not defined for hidden-price canaries, whose record is public by design)
    if c.kind == "hidden_price":
        lk["context_exposed"] = None
    else:
        lk["context_exposed"] = int(c.record.rid in (res.context_rids or [])) if cond != "C4" else int(
            phase == "G0_visible")
    append_jsonl(out, {
        "experiment": "governance", "variant": variant, "condition": cond, "policy": policy, "canary": c.cid,
        "kind": c.kind, "phase": phase, "lang": lang, "question": q, "answer": res.answer,
        "tool_calls": res.tool_calls, "retrieved": res.retrieved, "usage": res.usage, "error": res.error, **lk,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")})


def _run_variant(env: Env, variant: str, canaries: list[Canary], out: Path, langs: list[str],
                 done: set[tuple] | None = None, phases: set[str] | None = None) -> int:
    """Rows are written only when complete; with session generation, unanswered probes are left for a later round.
    ``done`` holds (variant, canary, phase group, lang) already written; ``phases`` restricts the phase groups
    (``G1``, ``G23``) run for this variant."""
    cond, policy, source = VARIANTS[variant]
    done = done or set()
    pending = 0
    # ---------------- G1: gated from the start
    if phases is None or "G1" in phases:
        store = env.store()
        insert_canaries(store, canaries, gated=True)
        index = env.index(store, policy=policy or "never", source=source) if cond in ("C1", "C1q", "C3") else None
        sysm = env.system(cond, store, index=index)
        for c in tqdm(canaries, desc=f"governance G1 {variant}", leave=False, disable=env.quiet):
            for p in c.probes:
                if p["lang"] not in langs or (variant, c.cid, "G1", p["lang"]) in done:
                    continue
                try:
                    res = sysm.ask(sysm.new_thread(p["lang"]), p["text"],
                                   cache_tag=f"gov|G1|{variant}|{c.cid}|{p['lang']}")
                except PendingRequest:
                    pending += 1
                    continue
                _row(out, variant, cond, policy, c, "G1_static", p["lang"], p["text"], res)
    if source == "raw" or (phases is not None and "G23" not in phases):
        return pending  # the misconfigured indexer is evaluated on G1 only
    # ---------------- G2/G3: public, indexed, then revoked; the store and index are restored after every canary
    store = env.store()
    insert_canaries(store, canaries, gated=False)
    index = env.index(store, policy=policy or "never") if cond in ("C1", "C1q", "C3") else None
    sysm = env.system(cond, store, index=index)
    for c in tqdm(canaries, desc=f"governance G2/G3 {variant}", leave=False, disable=env.quiet):
        p = next(x for x in c.probes if x["lang"] == langs[0])
        if (variant, c.cid, "G23", p["lang"]) in done:
            continue
        tag = f"gov|G23|{variant}|{c.cid}|{p['lang']}"
        rows = []
        gated = False
        try:
            th = sysm.new_thread(p["lang"])
            r0 = sysm.ask(th, p["text"], cache_tag=tag + "|visible")
            rows.append(("G0_visible", r0))
            store.apply_all(c.gate)
            gated = True
            r1 = sysm.ask(th, p["text"], cache_tag=tag + "|in_thread")
            rows.append(("G3_in_thread", r1))
            r2 = sysm.ask(sysm.new_thread(p["lang"]), p["text"], cache_tag=tag + "|fresh_before_refresh")
            rows.append(("G2_revoked_before_refresh", r2))
            if policy == "manual":
                index.refresh()
            r3 = sysm.ask(sysm.new_thread(p["lang"]), p["text"], cache_tag=tag + "|fresh_after_refresh")
            rows.append(("G2_revoked_after_refresh", r3))
        except PendingRequest:
            pending += 1
            rows = []
        finally:
            if gated:
                store.apply_all(c.ungate)
            if policy == "manual":
                index.refresh()
        for phase, res in rows:
            _row(out, variant, cond, policy, c, phase, p["lang"], p["text"], res)
    return pending


def run_governance(env: Env, run_dir: Path, variants: list[str], n_per_kind: int = 8,
                   langs: list[str] | None = None, workers: int = 4, seed: int = 20261005,
                   phases: dict[str, list[str]] | None = None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    canaries = make_canaries(n_per_kind=n_per_kind, seed=seed)
    (run_dir / "governance_canaries.json").write_text(
        json.dumps([c.to_dict() for c in canaries], ensure_ascii=False, indent=1), encoding="utf-8")
    out = run_dir / "governance.jsonl"
    done: set[tuple] = set()
    if out.exists():
        from ..schema import read_jsonl
        for r in read_jsonl(out):
            done.add((r["variant"], r["canary"], "G1" if r["phase"] == "G1_static" else "G23", r["lang"]))
    meta = {"variants": variants, "n_canaries": len(canaries), "langs": langs or ["el", "en"],
            "kb_fingerprint": env.store().fingerprint(), "snapshot": str(env.cfg["paths"]["snapshot"]),
            "config_hash": config_hash(env.cfg), "mock": env.mock, "phases": phases or {}}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        pend = list(ex.map(lambda v: _run_variant(env, v, canaries, out, langs or ["el", "en"], done,
                                                  set(phases[v]) if phases and v in phases else None), variants))
    meta["pending_probes"] = int(sum(pend))
    if meta["pending_probes"]:
        print(f"governance: {meta['pending_probes']} probes or probe sequences wait for generation")
    (run_dir / "meta_governance.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
