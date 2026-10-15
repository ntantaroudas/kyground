"""Optional: complete a public snapshot with a data-API dump.

The study's frozen snapshot is the public scrape alone; this module is for users with the API key who want a
snapshot that also contains what the public pages lack. ``kyground fetch-api`` stores raw API responses; their
exact field names are not documented. This module matches every API result to a record of the public snapshot
(same record type, and the same slug, or the same normalised title in either language; lessons by module and
title) and fills what the public pages lack - above all lesson texts, which the public site shows only after
login. API results that match no public record are listed in ``merge_report.md`` and added as new records, so
that nothing the agent could retrieve is missing from the snapshot. Review the report before using the result.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .. import textnorm as tn
from ..schema import Record, write_jsonl
from .api_client import _normalise_result

TITLE_KEYS = ("title", "name", "term", "question", "activity")
BODY_KEYS = ("content", "body", "text", "description", "answer", "definition", "summary")
FILE_TYPE = [("products_search", "product"), ("experiences_search", "experience"),
             ("cooperatives_search", "cooperative"), ("glossary_lookup", "glossary"), ("calendar_", "calendar"),
             ("modules_", "module"), ("lesson_", "lesson"), ("crop_", "crop"), ("knowledge_search", None)]
KNOWLEDGE_TYPES = {"article": "article", "articles": "article", "faq": "faq", "faqs": "faq",
                   "tradition": "tradition", "traditions": "tradition", "practice": "tradition"}


def _first(x: dict, keys: tuple[str, ...]) -> str:
    for k in keys:
        v = x.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, str) and vv.strip():
                    return vv.strip()
    return ""


def _file_type(name: str) -> str | None:
    for prefix, t in FILE_TYPE:
        if prefix in name:
            return t
    return None


def merge_api_snapshot(api_dir: str | Path, public: list[Record], out_dir: str | Path) -> dict:
    api_dir, out = Path(api_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs = {r.rid: copy.deepcopy(r) for r in public}
    by_slug = {(r.rtype, tn.norm(r.slug)): r for r in recs.values()}
    by_title: dict[tuple[str, str], Record] = {}
    for r in recs.values():
        for lang in ("el", "en"):
            if r.title.get(lang):
                by_title.setdefault((r.rtype, tn.norm(r.title[lang])), r)
    stats: Counter = Counter()
    unmatched: list[dict[str, Any]] = []
    for f in sorted((api_dir / "raw").glob("*.json")):
        hint = _file_type(f.name)
        lang = "en" if "_en_" in f.name or f.name.endswith("_en.json") else "el"
        data = json.loads(f.read_text(encoding="utf-8"))
        for x in _normalise_result(data)["results"]:
            stats["api_results"] += 1
            rtype = hint or KNOWLEDGE_TYPES.get(str(x.get("type", "")).lower(), "article")
            title, body = _first(x, TITLE_KEYS), _first(x, BODY_KEYS)
            slug = str(x.get("slug") or "")
            r = by_slug.get((rtype, tn.norm(slug))) if slug else None
            if r is None and title:
                r = by_title.get((rtype, tn.norm(title)))
            if r is None and rtype == "lesson" and title:
                module = f.name.split("_", 3)[-1].rsplit("_", 1)[0] if f.name.count("_") >= 3 else ""
                r = next((c for c in recs.values() if c.rtype == "lesson" and c.attrs.get("module_slug") == module
                          and tn.norm(c.title.get(lang, "") or c.title.get("el", "")) == tn.norm(title)), None)
            if r is None:
                stats["unmatched"] += 1
                if len(unmatched) < 200:
                    unmatched.append({"file": f.name, "type": rtype, "slug": slug, "title": title})
                rid = f"{rtype}:api-{tn.norm(tn.to_elot(slug or title))[:60].replace(' ', '-')}"
                if rid not in recs and title:
                    recs[rid] = Record(rid=rid, rtype=rtype, slug=slug or rid.split(":", 1)[1],
                                       title={lang: title}, body={lang: body} if body else {}, source="api")
                    by_title[(rtype, tn.norm(title))] = recs[rid]
                continue
            stats["matched"] += 1
            if body and not r.body.get(lang):
                r.body[lang] = body
                stats["filled_body"] += 1
            if title and not r.title.get(lang):
                r.title[lang] = title
                stats["filled_title"] += 1
            if rtype == "lesson" and body:
                r.attrs["content_available"] = True
                if slug:
                    r.attrs["api_slug"] = slug
            r.source = "public+api"
    write_jsonl(out / "records.jsonl", recs.values())
    counts = Counter(r.rtype for r in recs.values())
    manifest = {"source": "public snapshot + data API", "api_dir": str(api_dir), "stats": dict(stats),
                "counts": dict(counts)}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Merge of the data-API dump into the public snapshot", "",
             "| Statistic | Value |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in stats.items()] + [
             "", "## API results without a matching public record (added as new records)", "",
             "| File | Type | Slug | Title |", "|---|---|---|---|"] + [
             f"| {u['file']} | {u['type']} | {u['slug']} | {u['title']} |" for u in unmatched]
    (out / "merge_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest
