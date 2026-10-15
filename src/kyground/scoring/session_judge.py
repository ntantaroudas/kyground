"""LLM judge without an API key: grading by Claude inside an interactive Claude session.

The judge's rubric (``configs/prompts.yaml: judge``) is applied by Claude Sonnet 5.5 running as sub-agents of a
Claude session (Claude Code or Cowork) on the authors' subscription, instead of through the Anthropic API:

1. ``kyground judge-export`` writes blinded batches. Every answer gets an opaque id, and a batch lists only the
   question, the item type, the reference facts and the answer - no condition, item id, constructed label or
   deterministic grade - in a seeded random order that mixes conditions. ``batches/rubric.md`` holds the judge
   prompt and the output format; ``key.json`` (outside ``batches/``, never shown to a grader) maps ids back to the
   answers.
2. One sub-agent grades one batch, reading only ``rubric.md`` and its batch, grading each answer itself (no code,
   no look-ups), and writes ``batch_NNN.grades.jsonl`` with one JSON object per answer.
3. ``kyground judge-import`` validates the grades (every id exactly once, allowed labels; ``--check`` lists the
   batches to grade again) and merges them: run answers into ``accuracy_judged.jsonl`` (the file the report
   reads), controlled answers into ``scorer_suite.jsonl`` and ``scorer_suite.json``.

``kyground judge`` applies the same rubric through the API (needs ANTHROPIC_API_KEY); both write the same fields.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from ..schema import Item, read_jsonl, write_jsonl
from .judge import reference_text

LABELS = ("correct", "partially_correct", "incorrect", "abstained")

SUBAGENT_PROMPT = """You are an impartial evaluator grading answers for a research study (KyGround). Follow these steps exactly.

1. Read the grading rubric: {rubric}
2. Read the batch file: {batch} — JSON Lines, one answer per line, with fields id, question, item_type, reference, answer.
3. Grade every line yourself, one at a time and independently, exactly as the rubric says. The reference is the ground truth. Do NOT write or run code or scripts to grade, do NOT search the web, and do NOT open, list or read any other file or directory.
4. Use the Write tool to write the grades to {grades} — exactly one JSON object per line, one line per batch line, in the batch's order, with nothing else in the file:
{{"id": "<id>", "correctness": "correct" | "partially_correct" | "incorrect" | "abstained", "unsupported_specifics": true | false, "rationale": "<at most 20 words>"}}
5. Reply with a single line: the number of grades you wrote, and your model identifier as stated in your own instructions (or "unknown").
"""

OUTPUT_FORMAT = """
## Your task

The batch file is in JSON Lines format: each line is one answer to grade, with the fields `id`, `question`,
`item_type`, `reference` and `answer`. Grade every line on its own, using only the rubric above and that line's
question, item type and reference; the reference is the ground truth even if you believe otherwise. Grade each
answer yourself: do not write or run code, do not look anything up and do not open any other file.

Write one JSON object per line, in the same order as the batch and with exactly as many lines, to the grades file:

{"id": "<the line's id>", "correctness": "correct" | "partially_correct" | "incorrect" | "abstained", "unsupported_specifics": true | false, "rationale": "<at most 20 words>"}
"""


def opaque_id(key: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{key}".encode("utf-8")).hexdigest()[:12]


def run_key(r: dict) -> str:
    return f"{r['condition']}|{r['qid']}|{int(r.get('rep', 0))}"


def export_batches(rows: Iterable[dict], items: dict[str, Item], prompts: dict, out_dir: Path, kind: str,
                   batch_size: int = 50, seed: int = 20261005, limit: int | None = None) -> dict[str, Any]:
    """Blinded batches for grading. ``kind`` is ``run`` (answers of the experiments) or ``suite`` (controlled
    answers); the key file maps each opaque id to the run row (condition|qid|rep) or the suite id. With ``limit``,
    only the first ``limit`` answers of the seeded random order are kept (a random sample: the first batches of the
    full export)."""
    rows = list(rows)
    entries, key = [], {}
    for r in rows:
        k = run_key(r) if kind == "run" else r["sid"]
        oid = opaque_id(k, seed)
        if oid in key:
            raise ValueError(f"duplicate answer {k}")
        key[oid] = k
        it = items[r["iid"]]
        entries.append({"id": oid, "question": r["question"], "item_type": it.qtype, "reference": reference_text(it),
                        "answer": (r.get("answer") or "").strip() or "(empty)"})
    random.Random(seed).shuffle(entries)
    if limit is not None and limit < len(entries):
        entries = entries[:limit]
        kept = {e["id"] for e in entries}
        key = {k: v for k, v in key.items() if k in kept}
    bdir = out_dir / "batches"
    bdir.mkdir(parents=True, exist_ok=True)
    for old in bdir.glob("batch_*.jsonl"):
        if not old.name.endswith(".grades.jsonl"):
            old.unlink()
    nb = 0
    for i in range(0, len(entries), batch_size):
        nb += 1
        write_jsonl(bdir / f"batch_{nb:03d}.jsonl", entries[i:i + batch_size])
    (bdir / "rubric.md").write_text("# Grading rubric\n\n" + prompts["judge"]["system"].strip() + "\n" + OUTPUT_FORMAT,
                                    encoding="utf-8")
    first = bdir / "batch_001.jsonl"
    rel = f"<{out_dir.name}>/batches"     # the actual prompts give the absolute paths on the grading machine
    (out_dir / "subagent_prompt.txt").write_text(
        "Prompt given to each grading sub-agent (model: Claude Sonnet), shown here for batch_001:\n\n" +
        SUBAGENT_PROMPT.format(rubric=f"{rel}/rubric.md", batch=f"{rel}/{first.name}",
                               grades=f"{rel}/{first.stem}.grades.jsonl"), encoding="utf-8")
    (out_dir / "key.json").write_text(json.dumps({"kind": kind, "seed": seed, "ids": key}, ensure_ascii=False,
                                                 indent=0), encoding="utf-8")
    return {"kind": kind, "answers": len(entries), "batches": nb, "batch_size": batch_size, "dir": str(bdir)}


def read_grades(out_dir: Path) -> tuple[dict[str, dict], dict[str, Any]]:
    """Validated grades by opaque id, and a report: batches with missing, unknown, duplicate or malformed lines."""
    key = json.loads((out_dir / "key.json").read_text(encoding="utf-8"))["ids"]
    bdir = out_dir / "batches"
    grades: dict[str, dict] = {}
    bad_batches: dict[str, list[str]] = {}
    for b in sorted(bdir.glob("batch_*.jsonl")):
        if b.name.endswith(".grades.jsonl"):
            continue
        want = [e["id"] for e in read_jsonl(b)]
        gfile = b.with_name(b.stem + ".grades.jsonl")
        problems: list[str] = []
        got: dict[str, dict] = {}
        if not gfile.exists():
            problems.append("no grades file")
        else:
            for n, line in enumerate(gfile.read_text(encoding="utf-8").splitlines(), start=1):
                line = line.strip().strip(",")
                if not line or line.startswith("```"):
                    continue
                try:
                    g = json.loads(line)
                except json.JSONDecodeError:
                    problems.append(f"line {n}: not JSON")
                    continue
                oid = str(g.get("id", ""))
                if oid not in want:
                    problems.append(f"line {n}: id {oid!r} not in this batch")
                elif oid in got:
                    problems.append(f"line {n}: id {oid} graded twice")
                elif g.get("correctness") not in LABELS:
                    problems.append(f"line {n}: correctness {g.get('correctness')!r}")
                else:
                    got[oid] = {"correctness": g["correctness"],
                                "unsupported_specifics": bool(g.get("unsupported_specifics")),
                                "rationale": str(g.get("rationale", ""))[:300]}
            missing = [w for w in want if w not in got]
            if missing:
                problems.append(f"{len(missing)} answers not graded")
        if problems:
            bad_batches[b.name] = problems
        grades.update(got)
    report = {"answers": len(key), "graded": len(grades), "batches_to_redo": bad_batches}
    return grades, report


def merge_run_grades(run_dir: Path, out_dir: Path, model: str) -> int:
    """Write ``accuracy_judged.jsonl``: the scored rows that were graded, with the judge's fields."""
    grades, _ = read_grades(out_dir)
    key = json.loads((out_dir / "key.json").read_text(encoding="utf-8"))
    by_key = {k: grades[oid] for oid, k in key["ids"].items() if oid in grades}
    out = []
    for r in read_jsonl(run_dir / "accuracy_scored.jsonl"):
        g = by_key.get(run_key(r))
        if g:
            out.append({**r, "judge_model": model, "judge_error": "", "judge_correctness": g["correctness"],
                        "judge_unsupported": int(g["unsupported_specifics"]), "judge_rationale": g["rationale"]})
    write_jsonl(run_dir / "accuracy_judged.jsonl", out)
    return len(out)


def merge_suite_grades(suite_dir: Path, out_dir: Path, model: str) -> dict:
    """Add the judge's grades to ``scorer_suite.jsonl`` and recompute ``scorer_suite.json``."""
    from .suite import judge_binary, write_suite
    grades, _ = read_grades(out_dir)
    key = json.loads((out_dir / "key.json").read_text(encoding="utf-8"))
    by_sid = {k: grades[oid] for oid, k in key["ids"].items() if oid in grades}
    rows = list(read_jsonl(suite_dir / "scorer_suite.jsonl"))
    for r in rows:
        g = by_sid.get(r["sid"])
        if g:
            r.update({"judge_model": model, "judge_correctness": g["correctness"],
                      "judge_unsupported": int(g["unsupported_specifics"]), "judge_rationale": g["rationale"],
                      "judge_correct": judge_binary(g["correctness"], r["qtype"])})
    return write_suite(rows, suite_dir)
