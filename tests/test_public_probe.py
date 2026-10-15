"""The public-search probe must recover the matching behaviour of a search with known settings."""

from pathlib import Path

import pytest

from kyground.kb.replica_api import ReplicaAPI
from kyground.kb.store import KBStore
from kyground.schema import load_records
from kyground.sources.public_probe import battery, infer, run_battery

SNAP = Path(__file__).resolve().parents[1] / "data/snapshots/public_20261005/records.jsonl"


def _fake_search(mode: str):
    recs = [r for r in load_records(SNAP) if r.rtype == "module"]
    api = ReplicaAPI(KBStore.from_records(recs), search_mode=mode)

    def fetch(q: str) -> dict:
        return {"status": 200, "slugs": sorted(r.slug for r in recs if api._score(q, r) > 0)}

    return fetch


@pytest.mark.skipif(not SNAP.exists(), reason="snapshot not available")
@pytest.mark.parametrize("mode", ["phrase_casefold", "phrase_unaccent", "tokens_all_stem"])
def test_probe_recovers_search_mode(mode):
    rows = battery(load_records(SNAP))
    res = run_battery(rows, _fake_search(mode))
    out = infer(res)
    assert out["modules_found_by_canonical_query"] == 4
    assert out["recommended_search_mode"] == mode
    if mode == "phrase_casefold":
        p = out["properties"]
        assert p["case_insensitive"]["holds"] and p["final_sigma_folded"]["holds"]
        assert not p["accent_insensitive"]["holds"] and not p["word_order_insensitive"]["holds"]


@pytest.mark.skipif(not SNAP.exists(), reason="snapshot not available")
def test_battery_words_come_from_the_snapshot():
    rows = battery(load_records(SNAP))
    assert {r["family"] for r in rows} >= {"canonical", "upper", "no_accents", "final_sigma_as_medial", "greeklish",
                                           "other_inflection", "two_words_reversed"}
    assert all(r["query"] for r in rows)
