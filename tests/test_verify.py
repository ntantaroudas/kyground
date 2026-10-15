"""Automatic gold verification: each check must catch the error it is meant to catch."""

import copy
from pathlib import Path

import pytest

from kyground.bench import nuggets as N
from kyground.bench.templates import template_items
from kyground.bench.verify import verify_items
from kyground.schema import Item, Nugget, Record, load_items, load_records

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data/snapshots/public_20261005/records.jsonl"
BENCH = ROOT / "data/benchmark/v1.0/items.jsonl"


def _records() -> list[Record]:
    return [
        Record(rid="experience:a", rtype="experience", slug="a", title={"el": "Μάζεμα Ελιάς", "en": "Olive Picking"},
               body={"el": "Ζήστε το μάζεμα της ελιάς στον Καραβά."},
               attrs={"duration_hours": 5.0, "max_people": 4, "location_name": {"el": "Καραβάς"}}),
        Record(rid="experience:b", rtype="experience", slug="b", title={"el": "Μέλισσες", "en": "Bees"},
               body={"el": "Γνωρίστε τις μέλισσες."}, attrs={"duration_hours": 2.5, "max_people": 12}),
    ]


def _item(**kw) -> Item:
    base = dict(iid="cur-x", qtype="factoid", rtype="experience", template="curated", gold_rids=["experience:a"],
                cluster="experience:a", question={"el": "Ερώτηση;", "en": "Question?"}, origin="curated",
                nuggets=[N.text("place", ["Καραβάς"], ["Karavas"])],
                answer_ref={"el": "Στον Καραβά.", "en": "In Karavas."},
                evidence={"el": "το μάζεμα της ελιάς στον Καραβά", "en": ""})
    base.update(kw)
    return Item(**base)


def _fails(items, recs=None):
    return {r["iid"]: r["failures"] for r in verify_items(items, recs or _records()) if r["failures"]}


def test_correct_curated_item_passes():
    assert _fails([_item()]) == {}


def test_wrong_fact_and_missing_quote_are_caught():
    bad = _item(nuggets=[N.text("place", ["Μητάτα"], ["Mitata"])], answer_ref={"el": "Στα Μητάτα."},
                evidence={"el": "στα Μητάτα", "en": ""})
    f = _fails([bad])["cur-x"]
    assert any("not found in the gold records" in x for x in f)
    assert any("evidence quote" in x for x in f)


def test_elided_quote_and_unknown_record():
    ok = _item(evidence={"el": "Ζήστε ... στον Καραβά", "en": ""})
    assert _fails([ok]) == {}
    gone = _item(gold_rids=["experience:zzz"])
    assert any("not public" in x for x in _fails([gone])["cur-x"])


def test_reference_answer_must_score_correct():
    bad = _item(answer_ref={"el": "Κάπου στο νησί.", "en": "Somewhere."})
    assert any("reference answer" in x for x in _fails([bad])["cur-x"])


def test_derivation_rule_checks_aggregations():
    longest = _item(iid="cur-long", qtype="list", gold_rids=["experience:a"],
                    nuggets=[N.member("m0", ["Μάζεμα Ελιάς"], ["Olive Picking"])],
                    answer_ref={"el": "Μάζεμα Ελιάς", "en": "Olive Picking"}, evidence={},
                    derive={"rtype": "experience", "argmax": "duration_hours"})
    wrong = copy.deepcopy(longest)
    wrong.derive = {"rtype": "experience", "argmax": "max_people"}
    f = _fails([longest, wrong])
    assert "cur-long" in f and len(f) == 1


def test_unanswerable_needs_verified_absence():
    ua = _item(iid="ua-x", qtype="unanswerable", gold_rids=[], evidence={},
               nuggets=[Nugget(nid="abstain", kind="abstain", value={"absent": ["Κατάδυση"]})],
               answer_ref={"el": "Δεν υπάρχει τέτοια εμπειρία στην πλατφόρμα.", "en": "There is no such experience."})
    present = copy.deepcopy(ua)
    present.iid = "ua-present"
    present.nuggets = [Nugget(nid="abstain", kind="abstain", value={"absent": ["μέλισσες"]})]
    unspecified = copy.deepcopy(ua)
    unspecified.iid = "ua-unspecified"
    unspecified.nuggets = [N.abstain()]
    f = _fails([ua, present, unspecified])
    assert "ua-x" not in f
    assert any("occurs in the knowledge base" in x for x in f["ua-present"])
    assert any("no absence specification" in x for x in f["ua-unspecified"])


def test_template_items_are_rederived():
    recs = _records()
    its = template_items(recs)
    assert its and _fails(its, recs) == {}
    edited = copy.deepcopy(recs)
    edited[0].attrs["duration_hours"] = 6.0          # the record changed after the item was generated
    f = _fails(its, edited)
    assert any("template output differs" in x for fails in f.values() for x in fails)


@pytest.mark.skipif(not (SNAP.exists() and BENCH.exists()), reason="snapshot or benchmark not available")
def test_released_benchmark_verifies():
    res = verify_items(load_items(BENCH), load_records(SNAP))
    assert all(r["status"] == "verified" for r in res)
