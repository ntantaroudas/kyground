"""Replica of the platform's read-only bilingual data API (Table 2).

The replica reproduces the documented behaviour of the deployed API:

- nine endpoints, one per ``kythera`` MCP sub-tool;
- Greek or English output with Greek fallback;
- server-side editorial gating (drafts and inactive records excluded, hidden
  prices replaced by an on-request notice);
- substring matching on the free-text query and exact matching on structured
  filters;
- spatial filtering and proximity ordering for experiences and cooperatives,
  using the Well-Known Text (WKT) area-of-interest envelope.

How the deployed API normalises Greek text (accents, final sigma, multi-word
queries) is not documented, so the matching rule is a parameter
(``search_mode``). The live probe battery (``kyground probe-api``) measures
the real behaviour and the replica is then configured to match it; the
normalisation ablation varies it deliberately.

Search modes
------------
``phrase_casefold``     whole query must occur as a substring, case-insensitive,
                        accent-sensitive (emulates Django ``icontains``)
``phrase_unaccent``     as above but accent- and final-sigma-insensitive
``tokens_all_stem``     every content token must match (stem prefix), any order
``tokens_any_stem``     at least one content token must match; ranked by hits
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..schema import Record
from .. import textnorm as tn
from .serialize import public_view
from .store import KBStore

# Kythera + Antikythera bounding box (lon/lat, SRID 4326).
KYTHERA_WKT = "POLYGON((22.85 35.80, 23.35 35.80, 23.35 36.40, 22.85 36.40, 22.85 35.80))"

SEARCH_MODES = ("phrase_casefold", "phrase_unaccent", "tokens_all_stem", "tokens_any_stem")


def parse_wkt_bbox(wkt: str | None) -> tuple[float, float, float, float] | None:
    """Return (min_lon, min_lat, max_lon, max_lat) of a WKT polygon/point."""
    if not wkt:
        return None
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", wkt)]
    if len(nums) < 2:
        return None
    lons, lats = nums[0::2], nums[1::2]
    return (min(lons), min(lats), max(lons), max(lats))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _search_text(r: Record) -> list[tuple[str, float]]:
    """(text, weight) fields searched for a record; both languages."""
    fields: list[tuple[str, float]] = []
    for lang in ("el", "en"):
        if r.title.get(lang):
            fields.append((r.title[lang], 3.0))
        if r.summary.get(lang):
            fields.append((r.summary[lang], 1.5))
        if r.body.get(lang):
            fields.append((r.body[lang], 1.0))
    for key in ("producer", "category", "location_name", "village", "activity", "crop_name", "crop_type",
                "availability_note", "scientific_name"):
        v = r.attrs.get(key)
        if isinstance(v, dict):
            fields.extend((x, 1.5) for x in v.values() if x)
        elif isinstance(v, str) and v:
            fields.append((v, 1.5))
    fields.extend((t, 1.0) for t in r.tags)
    return fields


@dataclass
class ReplicaAPI:
    store: KBStore
    search_mode: str = "phrase_casefold"
    default_wkt: str = KYTHERA_WKT
    spatial_missing_coords: str = "include"      # include | exclude
    default_limit: int = 5
    filter_mode: str = "unaccent"                 # structured filters: unaccent (case- and accent-insensitive) |
                                                  # casefold (case-insensitive, accent-sensitive)
    ranking: str = "relevance"                    # relevance (weighted field matches) | stored (record order)
    max_body: int = 4000
    calls: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------ matching
    def _score(self, query: str, r: Record) -> float:
        q = (query or "").strip()
        if not q:
            return 0.5  # empty query: everything matches weakly
        fields = _search_text(r)
        mode = self.search_mode
        score = 0.0
        if mode == "phrase_casefold":
            qq = tn.casefold_only(q)
            score = sum(w for txt, w in fields if qq in tn.casefold_only(txt))
        elif mode == "phrase_unaccent":
            qq = tn.norm(q)
            score = sum(w for txt, w in fields if qq and qq in tn.norm(txt))
        elif mode in ("tokens_all_stem", "tokens_any_stem"):
            toks = tn.content_tokens(q, min_len=2) or tn.tokens(q)
            hits = 0.0
            matched_tokens = 0
            for tok in toks:
                best = 0.0
                for txt, w in fields:
                    if tn.stem_match(tok, txt):
                        best = max(best, w)
                if best > 0:
                    matched_tokens += 1
                    hits += best
            if mode == "tokens_all_stem" and matched_tokens < len(toks):
                return 0.0
            score = hits if matched_tokens else 0.0
        else:
            raise ValueError(f"unknown search_mode {mode}")
        return score

    def _f(self, text: str) -> str:
        """Normalisation used by the structured filters (crop type, calendar crop, category)."""
        return tn.norm(text) if self.filter_mode == "unaccent" else tn.casefold_only(text).strip()

    def _search(self, rtypes: tuple[str, ...], query: str, limit: int | None) -> list[tuple[Record, float]]:
        cands = [r for r in self.store.public() if r.rtype in rtypes]
        scored = [(r, self._score(query, r)) for r in cands]
        scored = [(r, s) for r, s in scored if s > 0]
        if self.ranking == "relevance":
            scored.sort(key=lambda x: (-x[1], x[0].rid))
        return scored[: (limit or self.default_limit)]

    def _views(self, recs: list[Record], lang: str) -> list[dict]:
        return [public_view(r, lang, max_body=self.max_body) for r in recs]

    def _log(self, endpoint: str, params: dict, results: list[dict]) -> None:
        self.calls.append({"endpoint": endpoint, "params": params, "n": len(results),
                           "ids": [x.get("id") for x in results]})

    # ----------------------------------------------------------- endpoints
    def knowledge_search(self, query: str = "", lang: str = "el", limit: int | None = None) -> dict:
        hits = self._search(("article", "faq", "tradition"), query, limit)
        res = self._views([r for r, _ in hits], lang)
        self._log("knowledge/search", {"q": query, "lang": lang, "limit": limit}, res)
        return {"count": len(res), "results": res}

    def seasonal_calendar(self, month: int | None = None, crop_type: str | None = None, lang: str = "el") -> dict:
        recs = self.store.by_type("calendar")
        if month:
            try:
                m = int(month)
                recs = [r for r in recs if r.attrs.get("month") == m]
            except (TypeError, ValueError):
                recs = []
        if crop_type:
            ct = self._f(crop_type)
            recs = [r for r in recs if ct in self._f(r.attr_text("crop_name", "el"))
                    or ct in self._f(r.attr_text("crop_name", "en"))
                    or ct == self._f(r.attrs.get("crop_slug", ""))]
        recs.sort(key=lambda r: (r.attrs.get("month", 0), r.rid))
        res = self._views(recs, lang)
        self._log("knowledge/seasonal-calendar", {"month": month, "crop_type": crop_type, "lang": lang}, res)
        return {"count": len(res), "results": res}

    def crop_info(self, crop_type: str, lang: str = "el") -> dict:
        """crops/{type}: exact match on slug or type/name (case/accent-insensitive)."""
        key = self._f(crop_type or "")
        crops = self.store.by_type("crop")
        match = [r for r in crops if key and key in (
            self._f(r.slug), self._f(r.slug.replace("-", " ")),
            self._f(r.title.get("el", "")), self._f(r.title.get("en", "")),
            self._f(r.attr_text("crop_type", "el")), self._f(r.attr_text("crop_type", "en")))]
        res = self._views(match, lang)
        related: list[dict] = []
        if match:
            names = {tn.norm(r.title.get("el", "")) for r in match} | {tn.norm(r.slug) for r in match}
            for c in self.store.by_type("calendar"):
                if tn.norm(c.attr_text("crop_name", "el")) in names or tn.norm(c.attrs.get("crop_slug", "")) in names:
                    related.append(public_view(c, lang))
            stem_names = [n.split()[0] for n in names if n]
            for k in self.store.public():
                if k.rtype in ("article", "tradition") and any(
                        tn.stem_match(n, k.title.get("el", "")) for n in stem_names):
                    related.append({"type": k.rtype, "id": k.rid, "title": k.title.get(lang) or k.title.get("el"),
                                    "url": k.url})
        out = {"count": len(res), "results": res, "related": related}
        self._log("crops/{type}", {"type": crop_type, "lang": lang}, res)
        return out

    def glossary_lookup(self, term: str = "", lang: str = "el", limit: int | None = None) -> dict:
        hits = self._search(("glossary",), term, limit or 10)
        res = self._views([r for r, _ in hits], lang)
        self._log("glossary/lookup", {"term": term, "lang": lang}, res)
        return {"count": len(res), "results": res}

    def product_search(self, query: str = "", lang: str = "el", category: str | None = None,
                       limit: int | None = None) -> dict:
        hits = self._search(("product",), query, None if category else limit or 10)
        recs = [r for r, _ in hits]
        if category:
            c = self._f(category)
            recs = [r for r in recs if c in self._f(r.attr_text("category", "el"))
                    or c in self._f(r.attr_text("category", "en"))]
            recs = recs[: (limit or 10)]
        res = self._views(recs, lang)
        self._log("products/search", {"q": query, "category": category, "lang": lang}, res)
        return {"count": len(res), "results": res}

    def _spatial(self, rtype: str, query: str, lang: str, wkt: str | None, limit: int | None,
                 endpoint: str) -> dict:
        hits = self._search((rtype,), query, 1000)
        bbox = parse_wkt_bbox(wkt or self.default_wkt)
        recs: list[tuple[Record, float, float | None]] = []
        cx = cy = None
        if bbox:
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        for r, s in hits:
            if r.lat is None or r.lon is None:
                if self.spatial_missing_coords == "exclude":
                    continue
                recs.append((r, s, None))
                continue
            if bbox and not (bbox[0] <= r.lon <= bbox[2] and bbox[1] <= r.lat <= bbox[3]):
                continue
            d = haversine_km(r.lat, r.lon, cy, cx) if cx is not None else None
            recs.append((r, s, d))
        # proximity to the envelope centroid (relevance first when ranking by relevance), as described for the
        # deployed interface; records without coordinates come last
        if self.ranking == "relevance":
            recs.sort(key=lambda x: (-x[1], x[2] if x[2] is not None else 1e9, x[0].rid))
        else:
            recs.sort(key=lambda x: (x[2] if x[2] is not None else 1e9))
        recs = recs[: (limit or 10)]
        res = []
        for r, _s, d in recs:
            v = public_view(r, lang, max_body=self.max_body)
            if d is not None:
                v["distance_km"] = round(d, 1)
            res.append(v)
        self._log(endpoint, {"q": query, "lang": lang, "wkt": bool(wkt)}, res)
        return {"count": len(res), "results": res}

    def experience_search(self, query: str = "", lang: str = "el", wkt: str | None = None,
                          limit: int | None = None) -> dict:
        return self._spatial("experience", query, lang, wkt, limit, "experiences/search")

    def cooperative_search(self, query: str = "", lang: str = "el", wkt: str | None = None,
                           limit: int | None = None) -> dict:
        return self._spatial("cooperative", query, lang, wkt, limit, "cooperatives/search")

    def training_modules(self, lang: str = "el", category: str | None = None) -> dict:
        recs = self.store.by_type("module")
        if category:
            c = self._f(category)
            recs = [r for r in recs if c in self._f(r.attr_text("category", "el"))
                    or c in self._f(r.attr_text("category", "en"))]
        res = self._views(recs, lang)
        self._log("training/modules", {"lang": lang, "category": category}, res)
        return {"count": len(res), "results": res}

    def lesson_content(self, module_slug: str, lesson_slug: str, lang: str = "el") -> dict:
        target = None
        for r in self.store.by_type("lesson"):
            if r.attrs.get("module_slug") == module_slug and (
                    r.slug == lesson_slug or r.slug.endswith(lesson_slug) or lesson_slug in r.slug):
                target = r
                break
        res = self._views([target], lang) if target else []
        self._log("training/modules/{m}/lessons/{slug}", {"module": module_slug, "lesson": lesson_slug}, res)
        if not target:
            return {"count": 0, "results": [], "error": "lesson not found"}
        return {"count": 1, "results": res}


# ---------------------------------------------------------------------------
# Tool backend: maps kythera sub-tool calls to API endpoints
# ---------------------------------------------------------------------------

class ToolBackend:
    """Interface shared by the replica and the live API backends."""

    name = "abstract"

    def call(self, tool: str, args: dict[str, Any], lang: str) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class ReplicaBackend(ToolBackend):
    api: ReplicaAPI
    wkt: str = KYTHERA_WKT
    name: str = "replica"

    def call(self, tool: str, args: dict[str, Any], lang: str) -> dict:
        a = dict(args or {})
        lg = a.pop("lang", None) or lang
        lg = lg if lg in ("el", "en") else lang
        lim = a.get("limit")
        try:
            lim = int(lim) if lim is not None else None
        except (TypeError, ValueError):
            lim = None
        fns: dict[str, Callable[[], dict]] = {
            "search_kythera_knowledge": lambda: self.api.knowledge_search(a.get("query", ""), lg, lim),
            "get_crop_info": lambda: self.api.crop_info(a.get("crop_type", ""), lg),
            "get_seasonal_calendar": lambda: self.api.seasonal_calendar(a.get("month"), a.get("crop_type"), lg),
            "lookup_glossary": lambda: self.api.glossary_lookup(a.get("term", ""), lg, lim),
            "search_local_products": lambda: self.api.product_search(a.get("query", ""), lg, a.get("category"), lim),
            "search_experiences": lambda: self.api.experience_search(a.get("query", ""), lg, self.wkt, lim),
            "search_cooperatives": lambda: self.api.cooperative_search(a.get("query", ""), lg, self.wkt, lim),
            "list_training_modules": lambda: self.api.training_modules(lg, a.get("category")),
            "get_lesson_content": lambda: self.api.lesson_content(a.get("module_slug", ""),
                                                                    a.get("lesson_slug", ""), lg),
        }
        fn = fns.get(tool)
        if fn is None:
            return {"error": f"unknown tool {tool}"}
        return fn()


def result_ids(result: dict) -> list[str]:
    """Record ids contained in a tool result (for retrieval diagnostics)."""
    ids = [x.get("id") for x in result.get("results", []) if isinstance(x, dict) and x.get("id")]
    ids += [x.get("id") for x in result.get("related", []) if isinstance(x, dict) and x.get("id")]
    return [i for i in ids if i]
