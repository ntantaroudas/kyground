"""Controlled answer suite: the deterministic scorer must grade every constructed answer as constructed."""

from pathlib import Path

import pytest

from kyground.schema import load_items
from kyground.scoring.suite import build_suite, judge_binary, score_suite, summarise

BENCH = Path(__file__).resolve().parents[1] / "data/benchmark/v1.0/items.jsonl"


def test_judge_binary():
    assert judge_binary("correct", "factoid") == 1
    assert judge_binary("abstained", "factoid") == 0
    assert judge_binary("abstained", "unanswerable") == 1
    assert judge_binary("partially_correct", "list") == 0


@pytest.mark.skipif(not BENCH.exists(), reason="benchmark not available")
def test_suite_covers_every_perturbation_and_the_scorer_passes():
    items = load_items(BENCH)
    rows = score_suite(build_suite(items), {i.iid: i for i in items})
    variants = {r["variant"] for r in rows}
    assert {"reference", "reworded", "reworded_no_accents", "wrong_text", "wrong_number", "wrong_month",
            "wrong_duration", "wrong_range", "list_member_missing", "abstention_on_answerable",
            "amount_for_price_on_request", "invented_answer"} <= variants
    assert any(v.startswith("abstention_with_") for v in variants)
    assert len({r["sid"] for r in rows}) == len(rows)
    det = summarise(rows)["deterministic"]
    assert det["overall"]["sensitivity"] == 1.0 and det["overall"]["specificity"] == 1.0, det["errors"][:5]
