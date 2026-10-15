"""Supplementary table for E2: by variant and phase, the share of turns with a tool call, the share in which the
edited record was among the retrieved records or tool results, and the answer states (updated, stale, mixed,
neither), for all conversations and for those whose first answer gave the old value (same-conversation phases).

    python tools/freshness_context.py   # -> paper/results/tables/freshness_context.{csv,md}
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASES = ["pre_edit", "in_thread_reask", "in_thread_challenge", "fresh_before_refresh", "fresh_after_refresh"]
VARIANTS = ["C2", "C1-upsert", "C1-batch"]


def main() -> None:
    tab = defaultdict(Counter)
    for line in open(ROOT / "runs/main/freshness.jsonl", encoding="utf-8"):
        r = json.loads(line)
        in_ctx = int(r["rid"] in (r.get("context_rids") or []))
        for subset in ("all", "first answer old") if r["phase"].startswith("in_thread") else ("all",):
            if subset == "first answer old" and not r.get("pre_gave_old"):
                continue
            c = tab[(r["variant"], r["phase"], subset)]
            c["n"] += 1
            c["tool_call"] += int(bool(r.get("tool_calls")))
            c["edited_record_in_context"] += in_ctx
            c[str(r["state"])] += 1
    rows = []
    for v in VARIANTS:
        for p in PHASES:
            for subset in ("all", "first answer old"):
                c = tab.get((v, p, subset))
                if not c:
                    continue
                rows.append({"variant": v, "phase": p, "conversations": subset, "n": c["n"],
                             "tool_call": c["tool_call"], "edited_record_in_context": c["edited_record_in_context"],
                             "updated": c["updated"], "stale": c["stale"], "mixed": c["mixed"],
                             "neither": c["neither"], "correct_before_edit": c["correct_pre"]})
    out = ROOT / "paper/results/tables"
    with open(out / "freshness_context.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    hdr = list(rows[0])
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    md += ["| " + " | ".join(str(r[h]) for h in hdr) + " |" for r in rows]
    (out / "freshness_context.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
