"""Live checks: records changed on the platform after the snapshot are left out of the live comparison."""

import copy

import pandas as pd

from kyground.analysis.report import live_comparison
from kyground.schema import Item, Record
from kyground.sources.snapshot_diff import affected_items, diff_snapshots, record_hashes


def _rec(rid: str, hours: float) -> Record:
    return Record(rid=rid, rtype="experience", slug=rid, title={"el": rid}, attrs={"duration_hours": hours})


def test_diff_and_affected_items():
    old = [_rec("experience:a", 2.0), _rec("experience:b", 3.0), _rec("experience:c", 1.0)]
    new = copy.deepcopy(old[:2]) + [_rec("experience:d", 4.0)]
    new[0].attrs["duration_hours"] = 2.5
    new[1].url = "https://example.org/moved"          # not content: ignored
    d = diff_snapshots(old, new)
    assert d["added"] == ["experience:d"] and d["removed"] == ["experience:c"]
    assert d["changed"] == [{"rid": "experience:a", "fields": ["attrs"]}]
    items = [Item(iid=f"i-{x}", qtype="numeric", rtype="experience", template="t", gold_rids=[f"experience:{x}"],
                  cluster=x, question={}, nuggets=[]) for x in "abc"]
    assert affected_items(d, items) == {"i-a", "i-c"}
    # the released hashes give the same verdicts without the content
    h = diff_snapshots(record_hashes(old), new)
    assert h["added"] == d["added"] and h["removed"] == d["removed"]
    assert [c["rid"] for c in h["changed"]] == ["experience:a"]


def test_live_comparison_uses_the_same_unchanged_queries():
    rows = []
    for cond, correct in (("C2", [1, 1, 0]), ("C2@live", [1, 0, 0])):
        for k, iid in enumerate(["i-a", "i-b", "i-c"]):
            rows.append({"condition": cond, "qid": f"{iid}#el", "rep": 0, "iid": iid, "form": "el",
                         "qtype": "numeric", "correct": correct[k]})
    items = [Item(iid=f"i-{x}", qtype="numeric", rtype="experience", template="t", gold_rids=[f"experience:{x}"],
                  cluster=x, question={}, nuggets=[]) for x in "abc"]
    table, key = live_comparison(pd.DataFrame(rows), items, {"removed": ["experience:c"], "changed": []})
    assert key["N_LIVE_EXCLUDED"] == 1 and key["N_LIVE_ITEMS"] == 2
    assert key["LIVE_ACC_el"] == "50.0%" and key["LIVE_REPLICA_ACC_el"] == "100.0%"
    assert table.iloc[0]["same_outcome"] == 0.5
