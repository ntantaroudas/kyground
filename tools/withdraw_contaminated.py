"""Withdraw the replies flagged by tools/contamination_audit.py so that they are generated again in isolation.

1. Copies the current outputs of runs/main to runs/main_batched (kept for the sensitivity analysis).
2. Moves each flagged reply out of the request cache (runs/session/_superseded_cache/contamination/) and removes
   it from its batch in the manifest (listed under "withdrawn"), so that the request can be exported again.
3. Removes the result rows that used a withdrawn reply; the runners then replay everything else from the cache
   and queue only the withdrawn requests (and whatever follows from them in a conversation).

    PYTHONPATH=src python tools/withdraw_contaminated.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kyground.schema import read_jsonl, write_jsonl  # noqa: E402

KEEP = ["accuracy.jsonl", "accuracy_scored.jsonl", "accuracy_judged.jsonl", "freshness.jsonl", "governance.jsonl",
        "contamination_flags.jsonl", "contamination_audit.json", "freshness_events.json",
        "governance_canaries.json", "config_snapshot.json"]


def main() -> None:
    run, bdir = ROOT / "runs/main", ROOT / "runs/session/batches"
    snap = ROOT / "runs/main_batched"
    if snap.exists():
        raise SystemExit(f"{snap} exists already: withdrawal was done before")
    snap.mkdir()
    for n in KEEP + [p.name for p in run.glob("meta_*.json")]:
        if (run / n).exists():
            shutil.copy2(run / n, snap / n)
    shutil.copytree(run / "judge_session", snap / "judge_session")

    flags = [json.loads(l) for l in open(run / "contamination_flags.jsonl", encoding="utf-8")]
    keys = {f["key"]: f for f in flags}
    # 2. cache and manifest
    dest = ROOT / "runs/session/_superseded_cache/contamination"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = [json.loads(l) for l in open(bdir / "manifest.jsonl", encoding="utf-8")]
    moved = 0
    for m in manifest:
        for rid, key in list(m["ids"].items()):
            if key not in keys:
                continue
            cp = Path(m["cache_paths"][rid])
            if cp.exists():
                shutil.move(str(cp), dest / cp.name)
                moved += 1
            m.setdefault("withdrawn", {})[rid] = key
            for fld in ("ids", "cache_paths", "in_tokens"):
                (m.get(fld) or {}).pop(rid, None)
    (bdir / "manifest.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest),
                                         encoding="utf-8")
    with open(dest / "withdrawn.jsonl", "w", encoding="utf-8") as f:
        for fl in flags:
            f.write(json.dumps({k: fl[k] for k in ("key", "batch", "tag", "kind", "experiment", "borrowed",
                                                   "foreign_records")}, ensure_ascii=False) + "\n")
    # 3. result rows
    acc_drop, fr_drop, gv_drop = set(), set(), set()
    for f in flags:
        if f["experiment"] == "accuracy":
            c = f["condition"]
            conds = {"C2", "C3"} if c == "router" else {c}
            acc_drop |= {(cc, f["qid"], f["rep"]) for cc in conds}
        elif f["experiment"] == "freshness":
            fr_drop.add((f["variant"], f["eid"], f["lang"]))
        elif f["experiment"] == "governance":
            grp = "G1" if f["phase"] == "G1_static" else "G23"
            gv_drop.add((f["variant"], f["cid"], grp, f["lang"]))

    def acc_hit(r: dict) -> bool:
        return (r["base_condition"], r["qid"], f"rep{r.get('rep', 0)}") in acc_drop

    def gv_hit(r: dict) -> bool:
        grp = "G1" if r["phase"] == "G1_static" else "G23"
        return (r["variant"], r["canary"], grp, r["lang"]) in gv_drop

    out = {}
    for name, hit in (("accuracy.jsonl", acc_hit),
                      ("freshness.jsonl", lambda r: (r["variant"], r["eid"], r["lang"]) in fr_drop),
                      ("governance.jsonl", gv_hit)):
        rows = list(read_jsonl(run / name))
        keep = [r for r in rows if not hit(r)]
        write_jsonl(run / name, keep)
        out[name] = len(rows) - len(keep)
    for n in ("accuracy_scored.jsonl", "accuracy_judged.jsonl"):
        (run / n).unlink(missing_ok=True)
    print(json.dumps({"flagged_requests": len(keys), "cache_files_moved": moved, "rows_removed": out}, indent=1))


if __name__ == "__main__":
    main()
