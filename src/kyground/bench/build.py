"""Benchmark builder: records -> items -> splits -> automatic gold verification -> surface-form queries."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import yaml

from ..schema import Item, Nugget, Record, load_items, write_jsonl
from .forms import FORMS, realise
from .templates import template_items

log = logging.getLogger(__name__)

# Default caps keep repetitive templates from dominating the benchmark.
DEFAULT_CAPS = {"P2-price": 8, "E4-price": 4, "K2-activity-month": 12}


def load_curated(path: str | Path, known_rids: set[str]) -> list[Item]:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out: list[Item] = []
    for d in spec.get("items", []):
        nuggets = []
        for n in d.get("nuggets", []):
            nuggets.append(Nugget(nid=n["nid"], kind=n["kind"], value=n.get("value"),
                                  aliases={"el": list(n.get("el", []) or []), "en": list(n.get("en", []) or [])},
                                  required=n.get("required", True)))
        gold = list(d.get("gold_rids") or [])
        missing = [g for g in gold if g not in known_rids]
        if missing:
            log.warning("curated item %s references unknown records: %s", d["iid"], missing)
        it = Item(
            iid=d["iid"], qtype=d["qtype"], rtype=d.get("rtype", ""), template="curated",
            gold_rids=gold, cluster=d.get("cluster") or (gold[0] if gold else f"ua:{d['iid']}"),
            question=d["question"], nuggets=nuggets, answer_ref=d.get("answer_ref", {}),
            evidence=d.get("evidence", {}), origin="curated", variants=d.get("variants", {}) or {},
            notes=d.get("notes", "") or "", derive=d.get("derive") or {},
        )
        if missing:
            it.notes = (it.notes + " " if it.notes else "") + f"[missing records: {', '.join(missing)}]"
        out.append(it)
    return out


def _h(seed: int, s: str) -> int:
    return int(hashlib.sha256(f"{seed}|{s}".encode()).hexdigest()[:12], 16)


def cap_templates(items: list[Item], caps: dict[str, int], seed: int) -> list[Item]:
    by_t: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_t[it.template].append(it)
    out: list[Item] = []
    for t, its in by_t.items():
        cap = caps.get(t)
        if cap is not None and len(its) > cap:
            its = sorted(its, key=lambda i: _h(seed, i.iid))[:cap]
        out.extend(its)
    return out


def assign_components(items: list[Item]) -> None:
    """Connected components of items that share a gold record or a cluster (union-find)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for it in items:
        union("i:" + it.iid, "c:" + it.cluster)
        for g in it.gold_rids:
            union("i:" + it.iid, "r:" + g)
    roots: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        roots[find("i:" + it.iid)].append(it)
    for members in roots.values():
        name = "comp:" + min(m.cluster for m in members)
        for m in members:
            m.component = name


def assign_splits(items: list[Item], dev_frac: float, seed: int, max_share: float = 0.1) -> None:
    """Development/test split at the level of components, so that no record contributes to both splits.
    Components are taken in a seeded random order until the development split holds about ``dev_frac`` of the
    items; components larger than ``max_share`` of the items always go to the test split."""
    comps: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        comps[it.component or it.cluster].append(it)
    target = dev_frac * len(items)
    cap = max(1, int(max_share * len(items)))
    dev: set[str] = set()
    n = 0
    for c in sorted(comps, key=lambda c: _h(seed, c)):
        if n >= target:
            break
        size = len(comps[c])
        if size > cap or n + size > 1.15 * target:
            continue
        dev.add(c)
        n += size
    for it in items:
        it.split = "dev" if (it.component or it.cluster) in dev else "test"


def build(records: list[Record], curated_path: str | Path | None, out_dir: str | Path, seed: int = 20261005,
          dev_frac: float = 0.2, caps: dict[str, int] | None = None, forms: Iterable[str] = FORMS,
          decisions_path: str | Path | None = None, snapshot_label: str = "") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    known = {r.rid for r in records}
    items = template_items(records)
    items = cap_templates(items, {**DEFAULT_CAPS, **(caps or {})}, seed)
    if curated_path and Path(curated_path).exists():
        items += load_curated(curated_path, known)
    # unique ids
    seen: Counter = Counter()
    for it in items:
        seen[it.iid] += 1
        if seen[it.iid] > 1:
            it.iid = f"{it.iid}~{seen[it.iid]}"
    if decisions_path and Path(decisions_path).exists():
        from .review import apply_decisions
        items = apply_decisions(items, yaml.safe_load(Path(decisions_path).read_text(encoding="utf-8")) or {})
    assign_components(items)
    assign_splits(items, dev_frac, seed)
    # automatic verification of every gold answer against the snapshot (bench/verify.py); an item that a manual
    # review marked as wrong or unclear stays unverified
    from .templates import DROPPED
    dropped = list(DROPPED)
    from .verify import apply_verification, verify_items, write_report
    manual_bad = {it.iid for it in items if it.status in ("needs_fix", "unclear")}
    results = verify_items(items, records)
    apply_verification(items, results)
    for it in items:
        if it.iid in manual_bad:
            it.status = "needs_fix"
    vsum = write_report(results, out, snapshot=snapshot_label)
    queries = [q for it in items for q in realise(it, it.variants, tuple(forms))]
    write_jsonl(out / "items.jsonl", items)
    write_jsonl(out / "queries.jsonl", queries)
    summary = summarise(items, queries)
    summary["verification"] = {k: vsum[k] for k in ("verified", "needs_fix", "checks")}
    summary["dropped_conflicting_duplicates"] = dropped
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def summarise(items: list[Item], queries) -> dict:
    return {
        "items": len(items),
        "clusters": len({i.cluster for i in items}),
        "components": len({i.component for i in items}),
        "by_status": dict(Counter(i.status for i in items).most_common()),
        "queries": len(queries),
        "by_template": dict(Counter(i.template for i in items).most_common()),
        "by_qtype": dict(Counter(i.qtype for i in items).most_common()),
        "by_rtype": dict(Counter(i.rtype for i in items).most_common()),
        "by_origin": dict(Counter(i.origin for i in items).most_common()),
        "by_split": dict(Counter(i.split for i in items).most_common()),
        "by_form": dict(Counter(q.form for q in queries).most_common()),
        "items_with_el_infl": sum(1 for i in items if i.variants.get("el_infl")),
    }


def rebuild_queries(items_path: str | Path, out_path: str | Path, forms: Iterable[str] = FORMS) -> int:
    items = load_items(items_path)
    qs = [q for it in items for q in realise(it, it.variants, tuple(forms))]
    return write_jsonl(out_path, qs)
