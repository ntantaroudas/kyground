"""Control for batched generation (protocol deviation D10): a seeded random sample of answer requests that the
contamination audit did not flag is generated again, one request per sub-agent, and the two replies are graded
with the deterministic scorer. Agreement without a systematic direction indicates that batching did not change
the outcomes of unflagged requests.

    PYTHONPATH=src python tools/control_isolation.py export     # writes runs/session/control_isolation/batches
    (one sub-agent per batch, as for the main generation)
    PYTHONPATH=src python tools/control_isolation.py compare    # -> runs/main/control_isolation.json
"""

from __future__ import annotations

import glob
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kyground.schema import load_items  # noqa: E402

CDIR = ROOT / "runs/session/control_isolation"
PER_CONDITION = {"C1": 25, "C1q": 25, "C2": 25, "C3": 25}
SEED = 20261005


def _payloads() -> dict[str, dict]:
    out = {}
    for f in sorted(glob.glob(str(ROOT / "runs/session/*.exported*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            e = json.loads(line)
            out[e["key"]] = e
    return out


def export() -> None:
    import tiktoken
    from kyground.llm.session import export_queue
    manifest = [json.loads(l) for l in open(ROOT / "runs/session/batches/manifest.jsonl", encoding="utf-8")]
    flagged = {json.loads(l)["key"] for l in open(ROOT / "runs/main_batched/contamination_flags.jsonl",
                                                  encoding="utf-8")}
    entries = _payloads()
    pool = defaultdict(list)
    for m in manifest:
        if m["kind"] != "answer" or len(m["ids"]) + len(m.get("withdrawn", {})) < 2:
            continue                     # only requests that shared a batch with others
        for key in m["ids"].values():
            tag = entries[key]["payload"]["tag"].split("|")
            if tag[0] == "acc" and tag[1] in PER_CONDITION and len(tag) == 5 and key not in flagged:
                pool[tag[1]].append(key)
    rng = random.Random(SEED)
    chosen = []
    for c, n in PER_CONDITION.items():
        keys = sorted(set(pool[c]))
        chosen += [(c, k) for k in rng.sample(keys, n)]
    CDIR.mkdir(parents=True, exist_ok=True)
    q = CDIR / "pending.jsonl"
    with open(q, "w", encoding="utf-8") as f:
        for c, k in chosen:
            e = dict(entries[k])
            e["cache_path"] = str(CDIR / "cache" / k[:2] / f"{k}.json")
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    (CDIR / "sample.json").write_text(json.dumps({"seed": SEED, "per_condition": PER_CONDITION,
                                                  "keys": [k for _, k in chosen]}, indent=1), encoding="utf-8")
    clusters = {i.iid: i.cluster for i in load_items(ROOT / "data/benchmark/v1.0/items.jsonl")}
    s = export_queue(q, CDIR / "batches", clusters, seed=SEED, max_chars=45000,
                     max_requests={"answer": 1, "router": 1, "rewrite": 1, "long_system": 1},
                     tokenizer=tiktoken.get_encoding("o200k_base"))
    print(json.dumps(s))


def compare() -> None:
    from kyground.llm.session import import_outputs
    from kyground.scoring.nuggets import score_answer
    r = import_outputs(CDIR / "batches", "claude-haiku-4-5 (Claude session sub-agent, isolation control)")
    print("stored", r["stored"], "to redo", r["batches_to_redo"])
    items = {i.iid: i for i in load_items(ROOT / "data/benchmark/v1.0/items.jsonl")}
    entries = _payloads()
    keys = json.loads((CDIR / "sample.json").read_text(encoding="utf-8"))["keys"]
    main_cache = ROOT / "runs/cache/llm/claude-haiku-4-5"
    tab = defaultdict(Counter)
    rows = []
    for k in keys:
        tag = entries[k]["payload"]["tag"].split("|")
        cond, qid = tag[1], tag[3]
        it = items[qid.split("#")[0]]
        a = json.loads((main_cache / k[:2] / f"{k}.json").read_text(encoding="utf-8"))["text"]
        cp = CDIR / "cache" / k[:2] / f"{k}.json"
        if not cp.exists():
            tab[cond]["missing"] += 1
            continue
        b = json.loads(cp.read_text(encoding="utf-8"))["text"]
        sa, sb = score_answer(it, a)["correct"], score_answer(it, b)["correct"]
        tab[cond][f"batched {sa} / isolated {sb}"] += 1
        tab["all"][f"batched {sa} / isolated {sb}"] += 1
        rows.append({"key": k, "condition": cond, "qid": qid, "batched_correct": sa, "isolated_correct": sb})
    summary = {c: dict(v) for c, v in sorted(tab.items())}
    for c, v in summary.items():
        n = sum(x for kk, x in v.items() if kk != "missing")
        same = v.get("batched 1 / isolated 1", 0) + v.get("batched 0 / isolated 0", 0)
        v["n"], v["agreement"] = n, round(same / n, 3) if n else None
        v["accuracy_batched"] = round((v.get("batched 1 / isolated 1", 0) + v.get("batched 1 / isolated 0", 0)) / n, 3)
        v["accuracy_isolated"] = round((v.get("batched 1 / isolated 1", 0) + v.get("batched 0 / isolated 1", 0)) / n, 3)
    out = {"design": "seeded random sample of unflagged answer requests that shared a batch, generated again one "
                     "per sub-agent; deterministic grades compared", "summary": summary, "rows": rows}
    (ROOT / "runs/main/control_isolation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    {"export": export, "compare": compare}[sys.argv[1]]()
