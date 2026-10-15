import json
from pathlib import Path

from kyground.schema import Record, bil, load_records
from kyground.sources.api_merge import merge_api_snapshot


def test_merge_fills_lesson_text_and_reports_unmatched(tmp_path: Path):
    pub = [Record(rid="lesson:mod-a--01", rtype="lesson", slug="mod-a--01", title=bil("Υγιές Έδαφος", "Healthy Soil"),
                  attrs={"module_slug": "mod-a", "content_available": False}),
           Record(rid="product:honey", rtype="product", slug="honey", title=bil("Μέλι", "Honey"))]
    raw = tmp_path / "api" / "raw"
    raw.mkdir(parents=True)
    (raw / "0001_lesson_el_mod-a_healthy-soil.json").write_text(json.dumps(
        {"title": "Υγιές Έδαφος", "slug": "healthy-soil", "content": "Κείμενο μαθήματος."}, ensure_ascii=False),
        encoding="utf-8")
    (raw / "0002_products_search_el_m.json").write_text(json.dumps(
        {"results": [{"title": "Μέλι", "slug": "honey"}, {"title": "Κρασί", "slug": "wine"}]}, ensure_ascii=False),
        encoding="utf-8")
    man = merge_api_snapshot(tmp_path / "api", pub, tmp_path / "out")
    recs = {r.rid: r for r in load_records(tmp_path / "out" / "records.jsonl")}
    assert recs["lesson:mod-a--01"].body["el"] == "Κείμενο μαθήματος."
    assert recs["lesson:mod-a--01"].attrs["content_available"] is True
    assert man["stats"]["matched"] == 2 and man["stats"]["unmatched"] == 1
    assert "Κρασί" in (tmp_path / "out" / "merge_report.md").read_text(encoding="utf-8")
