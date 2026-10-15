"""Replica-vs-live conformance.

Replays the tool calls the router actually made in an accuracy run (C2 on the
replica) against the deployed API and compares the returned records. The
match key is the normalised title, because record identifiers differ between
the scraped snapshot and the API.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..kb.replica_api import ReplicaBackend
from ..schema import read_jsonl
from .. import textnorm as tn


def _keys(res: dict) -> set[str]:
    """Identity of returned records: type, slug or id, and normalised title (two records with the same title,
    such as near-duplicates, stay distinct)."""
    out = set()
    for x in res.get("results", []):
        t = x.get("title") or x.get("name") or x.get("term") or x.get("question") or x.get("activity") or ""
        ident = x.get("slug") or str(x.get("id") or "")
        ident = ident.split(":", 1)[-1] if ident else ""
        out.add(f"{x.get('type', '')}|{tn.norm(ident)}|{tn.norm(t)}")
    return out


def _same(a: set[str], b: set[str]) -> tuple[float, int]:
    """Jaccard similarity of two result sets, matching on title when identifiers differ in form."""
    ta = {k.split("|")[-1] for k in a}
    tb = {k.split("|")[-1] for k in b}
    if len(ta) == len(a) and len(tb) == len(b):   # titles are unique within each set: compare titles
        a, b = ta, tb
    jac = len(a & b) / len(a | b) if (a | b) else 1.0
    return jac, int(a == b)


def conformance(run_rows: Path, replica: ReplicaBackend, live, max_calls: int | None = None,
                seed: int = 20261005) -> dict:
    """Replay the distinct tool calls of the main C2 run (not the ablation, live or open-model runs) against the
    deployed interface; all calls, or a seeded random sample of ``max_calls``."""
    import random
    calls: dict[str, tuple[dict, str]] = {}
    for r in read_jsonl(run_rows):
        if r.get("base_condition") != "C2" or r.get("label"):
            continue
        for tc in r.get("tool_calls") or []:
            key = json.dumps([tc["name"], tc["args"]], sort_keys=True, ensure_ascii=False)
            calls.setdefault(key, (tc, r.get("lang", "el")))
    keys = sorted(calls)
    n_distinct = len(keys)
    if max_calls and len(keys) > max_calls:
        keys = sorted(random.Random(seed).sample(keys, max_calls))
    rows = []
    for key in keys:
        tc, lang = calls[key]
        rep = replica.call(tc["name"], tc["args"], lang)
        liv = live.call(tc["name"], tc["args"], lang)
        jac, exact = _same(_keys(rep), _keys(liv))
        rows.append({"tool": tc["name"], "args": tc["args"], "n_replica": rep.get("count", 0),
                     "n_live": liv.get("count", 0), "jaccard": round(jac, 3), "exact": exact,
                     "live_status": liv.get("_status")})
    n = len(rows)
    summary = {"n_distinct_calls": n_distinct, "n_calls": n,
               "mean_jaccard": round(sum(x["jaccard"] for x in rows) / n, 3) if n else None,
               "exact_rate": round(sum(x["exact"] for x in rows) / n, 3) if n else None}
    return {"summary": summary, "rows": rows}
