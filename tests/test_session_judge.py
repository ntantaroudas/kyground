"""Grading by Claude sub-agents in a session: batches are blinded, grades are validated and merged back."""

import json
from pathlib import Path

from kyground.bench import nuggets as N
from kyground.schema import Item, read_jsonl, write_jsonl
from kyground.scoring.session_judge import export_batches, merge_run_grades, merge_suite_grades, read_grades
from kyground.scoring.suite import build_suite, score_suite, write_suite

PROMPTS = {"judge": {"system": "You are a strict, impartial evaluator."}}


def _items():
    return {"i1": Item(iid="i1", qtype="numeric", rtype="experience", template="t", gold_rids=["experience:a"],
                       cluster="a", question={"el": "Πόσες ώρες;", "en": "How many hours?"},
                       nuggets=[N.duration_hours("duration", 4)], answer_ref={"el": "4 ώρες", "en": "4 hours"}),
            "i2": Item(iid="i2", qtype="unanswerable", rtype="product", template="t", gold_rids=[], cluster="b",
                       question={"el": "Τηλέφωνο;", "en": "Phone?"}, nuggets=[N.abstain()],
                       answer_ref={"el": "Δεν υπάρχει στην πλατφόρμα.", "en": "Not on the platform."})}


def _run_rows():
    rows = []
    for cond in ("C1", "C2"):
        for iid in ("i1", "i2"):
            rows.append({"condition": cond, "qid": f"{iid}#el", "rep": 0, "iid": iid, "form": "el", "qtype": "x",
                         "question": "Πόσες ώρες;", "answer": f"Διαρκεί {len(rows) + 2} ώρες.", "correct": 1})
    return rows


def test_batches_are_blinded_and_grades_merge(tmp_path: Path):
    run = tmp_path / "run"
    write_jsonl(run / "accuracy_scored.jsonl", _run_rows())
    out = run / "judge_session"
    s = export_batches(_run_rows(), _items(), PROMPTS, out, "run", batch_size=3)
    assert s["answers"] == 4 and s["batches"] == 2
    lines = [e for b in sorted((out / "batches").glob("batch_*.jsonl")) for e in read_jsonl(b)]
    assert all(set(e) == {"id", "question", "item_type", "reference", "answer"} for e in lines)
    assert not any(x in json.dumps(lines) for x in ('"C1', '"C2', "i1#", "condition"))
    assert not (out / "batches" / "key.json").exists() and (out / "key.json").exists()
    assert "strict, impartial" in (out / "batches" / "rubric.md").read_text(encoding="utf-8")
    # grade only the first batch: the second is reported as missing
    b1 = sorted((out / "batches").glob("batch_*.jsonl"))[0]
    grades = [{"id": e["id"], "correctness": "correct", "unsupported_specifics": False, "rationale": "ok"}
              for e in read_jsonl(b1)]
    write_jsonl(b1.with_name(b1.stem + ".grades.jsonl"), grades)
    _, rep = read_grades(out)
    assert rep["graded"] == 3 and list(rep["batches_to_redo"]) == ["batch_002.jsonl"]
    b2 = [b for b in sorted((out / "batches").glob("batch_*.jsonl")) if ".grades" not in b.name][1]
    e = list(read_jsonl(b2))[0]
    (b2.with_name(b2.stem + ".grades.jsonl")).write_text(
        json.dumps({"id": e["id"], "correctness": "wrong"}) + "\n", encoding="utf-8")
    _, rep = read_grades(out)
    assert any("correctness" in p for p in rep["batches_to_redo"]["batch_002.jsonl"])
    write_jsonl(b2.with_name(b2.stem + ".grades.jsonl"),
                [{"id": e["id"], "correctness": "abstained", "unsupported_specifics": False, "rationale": ""}])
    _, rep = read_grades(out)
    assert rep["graded"] == 4 and rep["batches_to_redo"] == {}
    assert merge_run_grades(run, out, "claude-sonnet-5-5 (session)") == 4
    judged = list(read_jsonl(run / "accuracy_judged.jsonl"))
    assert {r["judge_correctness"] for r in judged} <= {"correct", "abstained"}


def test_suite_grades_give_judge_sensitivity(tmp_path: Path):
    items = _items()
    rows = score_suite(build_suite(list(items.values())), items)
    write_suite(rows, tmp_path)
    out = tmp_path / "judge_session_suite"
    export_batches(rows, items, PROMPTS, out, "suite", batch_size=100)
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))["ids"]
    label = {r["sid"]: r["label"] for r in rows}
    qtype = {r["sid"]: r["qtype"] for r in rows}
    grades = []
    for oid, sid in key.items():     # a perfect grader
        ok = label[sid] == 1
        lab = ("abstained" if qtype[sid] == "unanswerable" else "correct") if ok else "incorrect"
        grades.append({"id": oid, "correctness": lab, "unsupported_specifics": False, "rationale": ""})
    write_jsonl(out / "batches" / "batch_001.grades.jsonl", grades)
    summary = merge_suite_grades(tmp_path, out, "claude-sonnet-5-5 (session)")
    assert summary["judge"]["overall"]["sensitivity"] == 1.0 and summary["judge"]["overall"]["specificity"] == 1.0
