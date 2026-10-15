"""Client for the platform's live read-only data API (``/api/v1/``).

Authentication: ``X-API-Key`` header, read from the environment variable
``DASK_API_KEY`` (put it in ``.env``; never commit it). The client is
read-only and rate-limited (default 1 request/second).

Three uses:

1. :class:`LiveBackend` lets the tool agent (C2) query the *deployed* API
   instead of the replica (validity check).
2. :func:`probe_normalisation` characterises how the deployed search handles
   Greek orthographic variation (accents, final sigma, case, inflection,
   multi-word queries, Greeklish) - a result in its own right and the basis
   for configuring the replica.
3. :func:`snapshot` stores raw API responses for provenance and comparison
   with the public snapshot.

Query-parameter names are configurable (``live_api.params`` in the study
config) because the article documents the endpoints but not their parameter
names; run ``kyground probe-api --discover`` once to confirm them.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from ..kb.replica_api import ToolBackend
from .. import textnorm as tn

DEFAULT_PARAMS = {
    "knowledge/search": {"query": "q", "lang": "lang", "limit": "limit"},
    "knowledge/seasonal-calendar": {"month": "month", "crop_type": "crop_type", "lang": "lang"},
    "crops/{type}": {"lang": "lang"},
    "glossary/lookup": {"term": "term", "lang": "lang"},
    "products/search": {"query": "q", "category": "category", "lang": "lang", "limit": "limit"},
    "experiences/search": {"query": "q", "wkt": "wkt", "lang": "lang", "limit": "limit"},
    "cooperatives/search": {"query": "q", "wkt": "wkt", "lang": "lang", "limit": "limit"},
    "training/modules": {"category": "category", "lang": "lang"},
    "training/modules/{module}/lessons/{slug}": {"lang": "lang"},
}

TOOL_ENDPOINT = {
    "search_kythera_knowledge": "knowledge/search",
    "get_crop_info": "crops/{type}",
    "get_seasonal_calendar": "knowledge/seasonal-calendar",
    "lookup_glossary": "glossary/lookup",
    "search_local_products": "products/search",
    "search_experiences": "experiences/search",
    "search_cooperatives": "cooperatives/search",
    "list_training_modules": "training/modules",
    "get_lesson_content": "training/modules/{module}/lessons/{slug}",
}


@dataclass
class LiveAPI:
    base_url: str = "https://dask-platform.onrender.com/api/v1/"
    key_env: str = "DASK_API_KEY"
    params: dict[str, dict[str, str]] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    min_interval_s: float = 1.0
    timeout_s: float = 30.0
    _last: float = 0.0

    @classmethod
    def from_config(cls, cfg: dict) -> "LiveAPI":
        p = dict(DEFAULT_PARAMS)
        p.update(cfg.get("params") or {})
        return cls(base_url=cfg.get("base_url", cls.base_url), key_env=cfg.get("key_env", "DASK_API_KEY"),
                   params=p, min_interval_s=float(cfg.get("min_interval_s", 1.0)))

    @property
    def key(self) -> str:
        k = os.environ.get(self.key_env)
        if not k:
            raise RuntimeError(f"Set {self.key_env} (X-API-Key of the read-only data API) in .env")
        return k

    def get(self, endpoint: str, path_args: dict[str, str] | None = None, **kwargs: Any) -> dict:
        path = endpoint
        for k, v in (path_args or {}).items():
            path = path.replace("{" + k + "}", requests.utils.quote(str(v), safe=""))
        names = self.params.get(endpoint, {})
        q = {names.get(k, k): v for k, v in kwargs.items() if v not in (None, "")}
        wait = self.min_interval_s - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")
        t0 = time.time()
        r = requests.get(url, params=q, headers={"X-API-Key": self.key, "Accept": "application/json"},
                         timeout=self.timeout_s)
        self._last = time.time()
        try:
            data = r.json()
        except ValueError:
            data = {"error": f"non-JSON response ({r.status_code})", "text": r.text[:500]}
        if not isinstance(data, dict):
            data = {"results": data}
        data["_status"] = r.status_code
        data["_latency_s"] = round(time.time() - t0, 3)
        return data


def _normalise_result(data: dict) -> dict:
    """Coerce the live response to the replica's {count, results[]} shape and
    give each result an ``id`` usable for conformance checks."""
    res = data.get("results")
    if res is None:
        for k in ("items", "data", "records", "entries"):
            if isinstance(data.get(k), list):
                res = data[k]
                break
    if res is None:
        res = [data] if any(k in data for k in ("title", "name", "slug")) else []
    out = []
    for x in res if isinstance(res, list) else []:
        if isinstance(x, dict):
            x = dict(x)
            x.setdefault("id", x.get("slug") or x.get("id") or x.get("title") or x.get("name"))
            out.append(x)
    return {"count": len(out), "results": out, "_status": data.get("_status"),
            "_latency_s": data.get("_latency_s")}


@dataclass
class LiveBackend(ToolBackend):
    api: LiveAPI
    wkt: str = ""
    name: str = "live"

    def call(self, tool: str, args: dict[str, Any], lang: str) -> dict:
        a = dict(args or {})
        lg = a.pop("lang", None) or lang
        ep = TOOL_ENDPOINT.get(tool)
        if ep is None:
            return {"error": f"unknown tool {tool}"}
        if tool == "get_crop_info":
            data = self.api.get(ep, {"type": a.get("crop_type", "")}, lang=lg)
        elif tool == "get_lesson_content":
            data = self.api.get(ep, {"module": a.get("module_slug", ""), "slug": a.get("lesson_slug", "")}, lang=lg)
        elif tool in ("search_experiences", "search_cooperatives"):
            data = self.api.get(ep, query=a.get("query"), wkt=self.wkt, lang=lg, limit=a.get("limit"))
        else:
            data = self.api.get(ep, **a, lang=lg)
        return _normalise_result(data)


# ---------------------------------------------------------------------------
# Probe battery: how does the deployed search treat Greek variation?
# ---------------------------------------------------------------------------

PROBE_TERMS = [
    # (endpoint, canonical term, variants by family)
    ("knowledge/search", "ελιά", {"noacc": "ελια", "caps": "ΕΛΙΑ", "plural": "ελιές", "genitive": "ελιάς",
                                  "greeklish": "elia", "english": "olive"}),
    ("knowledge/search", "συγκομιδή", {"noacc": "συγκομιδη", "caps": "ΣΥΓΚΟΜΙΔΗ", "genitive": "συγκομιδής",
                                       "greeklish": "sygkomidi", "english": "harvest",
                                       "two_words": "συγκομιδή ελιάς"}),
    ("knowledge/search", "μέλι", {"noacc": "μελι", "caps": "ΜΕΛΙ", "genitive": "μελιού", "greeklish": "meli",
                                  "english": "honey", "two_words": "θυμαρίσιο μέλι"}),
    ("products/search", "Μουζούρι", {"noacc": "Μουζουρι", "caps": "ΜΟΥΖΟΥΡΙ", "lower": "μουζούρι",
                                     "greeklish": "mouzouri"}),
    ("products/search", "αλάτι", {"noacc": "αλατι", "caps": "ΑΛΑΤΙ", "genitive": "αλατιού", "greeklish": "alati",
                                  "english": "salt"}),
    ("products/search", "θυμαρίσιο", {"noacc": "θυμαρισιο", "caps": "ΘΥΜΑΡΙΣΙΟ", "genitive": "θυμαρίσιου",
                                      "greeklish": "thymarisio", "english": "thyme"}),
    ("experiences/search", "μέλισσα", {"noacc": "μελισσα", "caps": "ΜΕΛΙΣΣΑ", "plural": "μέλισσες",
                                       "related_word": "μελισσοκομία", "greeklish": "melissa", "english": "bee"}),
    ("experiences/search", "Καραβάς", {"noacc": "Καραβας", "accusative": "Καραβά", "greeklish": "Karavas"}),
    ("glossary/lookup", "δάκος", {"noacc": "δακος", "caps": "ΔΑΚΟΣ", "accusative": "δάκο", "english": "olive fruit fly"}),
]


def probe_normalisation(api: LiveAPI) -> list[dict]:
    rows = []
    for ep, canon, variants in PROBE_TERMS:
        param = "term" if ep == "glossary/lookup" else "query"
        base = _normalise_result(api.get(ep, **{param: canon}, lang="el"))
        base_ids = {r["id"] for r in base["results"]}
        rows.append({"endpoint": ep, "term": canon, "family": "canonical", "query": canon, "n": base["count"],
                     "jaccard_vs_canonical": 1.0, "status": base["_status"]})
        for fam, q in variants.items():
            res = _normalise_result(api.get(ep, **{param: q}, lang="el"))
            ids = {r["id"] for r in res["results"]}
            jac = len(ids & base_ids) / len(ids | base_ids) if (ids | base_ids) else 1.0
            rows.append({"endpoint": ep, "term": canon, "family": fam, "query": q, "n": res["count"],
                         "jaccard_vs_canonical": round(jac, 3), "status": res["_status"]})
    return rows


FILTER_PROBES = [  # structured filters: are they accent- and case-insensitive?
    ("crops/{type}", "path", ["Ελιά", "ελια", "ΕΛΙΑ", "olive"]),
    ("knowledge/seasonal-calendar", "crop_type", ["Ελιά", "ελια", "ΕΛΙΑ", "olive"]),
    ("products/search", "category", ["ΜΕΛΙ", "Μέλι", "μελι", "Honey"]),
]


def probe_behaviour(api: LiveAPI) -> dict:
    """Behaviour of the deployed interface that the replica must reproduce beyond free-text matching:
    structured-filter normalisation, spatial filtering of records without coordinates, default result limits
    and result order."""
    from ..kb.replica_api import KYTHERA_WKT
    out: dict[str, Any] = {"filters": [], "spatial": {}, "default_limit": {}, "ordering": {}}
    for ep, param, values in FILTER_PROBES:
        for v in values:
            data = api.get(ep, {"type": v}, lang="el") if param == "path" else api.get(ep, **{param: v}, lang="el")
            r = _normalise_result(data)
            out["filters"].append({"endpoint": ep, "param": param, "value": v, "n": r["count"],
                                   "ids": sorted(str(x["id"]) for x in r["results"]), "status": r["_status"]})
    with_wkt = _normalise_result(api.get("experiences/search", query="", wkt=KYTHERA_WKT, lang="el", limit=100))
    without = _normalise_result(api.get("experiences/search", query="", lang="el", limit=100))
    out["spatial"] = {"n_with_wkt": with_wkt["count"], "n_without_wkt": without["count"],
                      "titles_with_wkt": [x.get("title") for x in with_wkt["results"]]}
    for ep, q in (("products/search", "μέλι"), ("knowledge/search", "ελιά"), ("experiences/search", "Κύθηρα")):
        default = _normalise_result(api.get(ep, query=q, lang="el"))
        many = _normalise_result(api.get(ep, query=q, lang="el", limit=50))
        out["default_limit"][ep] = {"query": q, "n_default": default["count"], "n_limit50": many["count"]}
        out["ordering"][ep] = {"query": q, "titles": [x.get("title") or x.get("name") for x in many["results"]]}
    return out


def recommend_replica(norm_rows: list[dict], behaviour: dict, store: Any) -> dict:
    """Replica settings implied by the probes, with the evidence for each (review before use)."""
    import statistics
    from ..kb.replica_api import ReplicaAPI
    rec: dict[str, Any] = {}
    why: dict[str, str] = {}
    fam = {}
    for r in norm_rows:
        if r["family"] != "canonical":
            fam.setdefault(r["family"], []).append(r["jaccard_vs_canonical"])
    m = {k: statistics.mean(v) for k, v in fam.items()}
    accent_insensitive = m.get("noacc", 0) > 0.8
    inflection = statistics.mean([m.get(k, 0) for k in ("plural", "genitive", "accusative") if k in m] or [0]) > 0.8
    rec["search_mode"] = "tokens_all_stem" if inflection else ("phrase_unaccent" if accent_insensitive else
                                                              "phrase_casefold")
    why["search_mode"] = f"mean Jaccard to canonical by variant family: {', '.join(f'{k}={v:.2f}' for k, v in m.items())}"
    crops = {f["value"]: f["ids"] for f in behaviour["filters"] if f["endpoint"] == "crops/{type}"}
    rec["filter_mode"] = "unaccent" if crops.get("ελια") and crops.get("ελια") == crops.get("Ελιά") else "casefold"
    why["filter_mode"] = f"crops/{{type}} result ids: {crops}"
    exps = [r for r in store.public() if r.rtype == "experience"]
    with_coords = sum(1 for r in exps if r.lat is not None and r.lon is not None)
    n = behaviour["spatial"]["n_with_wkt"]
    rec["spatial_missing_coords"] = "exclude" if n == with_coords and n != len(exps) else "include"
    why["spatial_missing_coords"] = (f"{n} experiences returned within the envelope; snapshot has {len(exps)} "
                                     f"published experiences, {with_coords} with coordinates")
    lims = [v["n_default"] for v in behaviour["default_limit"].values() if v["n_limit50"] > v["n_default"]]
    rec["default_limit"] = min(lims) if lims else 10
    why["default_limit"] = json.dumps(behaviour["default_limit"], ensure_ascii=False)
    agree = {}
    for ranking in ("relevance", "stored"):
        api = ReplicaAPI(store, search_mode=rec["search_mode"], ranking=ranking)
        score = 0
        for ep, v in behaviour["ordering"].items():
            fn = {"products/search": api.product_search, "knowledge/search": api.knowledge_search,
                  "experiences/search": api.experience_search}[ep]
            mine = [tn.norm(x.get("title") or "") for x in fn(v["query"], "el", limit=50)["results"]]
            live = [tn.norm(t or "") for t in v["titles"]]
            score += sum(1 for a, b in zip(mine, live) if a == b)
        agree[ranking] = score
    rec["ranking"] = max(agree, key=agree.get)
    why["ranking"] = f"positions agreeing with the live order: {agree}"
    return {"replica": rec, "evidence": why}


def discover_params(api: LiveAPI) -> dict:
    """Try common parameter names for the free-text endpoints."""
    report: dict[str, Any] = {}
    for ep, canon in (("knowledge/search", "ελιά"), ("products/search", "μέλι"), ("glossary/lookup", "δάκος"),
                      ("experiences/search", "μέλισσα")):
        report[ep] = {}
        for name in ("q", "query", "search", "term", "text"):
            saved = api.params.get(ep, {}).copy()
            api.params[ep] = {**saved, "query": name, "term": name}
            res = _normalise_result(api.get(ep, query=canon, lang="el"))
            api.params[ep] = saved
            report[ep][name] = {"status": res["_status"], "n": res["count"]}
    return report


def snapshot(api: LiveAPI, out_dir: str | Path, seeds: list[str] | None = None) -> dict:
    """Store raw API responses (provenance). Enumerates the search endpoints
    with single-letter and seed queries, the calendar by month and the
    training modules with their lessons."""
    out = Path(out_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    letters = list("αεηιουω") + list("aeiou")
    seeds = (seeds or []) + letters
    calls = 0

    def save(name: str, data: dict) -> None:
        nonlocal calls
        calls += 1
        (out / "raw" / f"{calls:04d}_{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")

    for ep in ("knowledge/search", "products/search", "experiences/search", "cooperatives/search",
               "glossary/lookup"):
        param = "term" if ep == "glossary/lookup" else "query"
        for lang in ("el", "en"):
            for s in seeds:
                save(f"{ep.replace('/', '_')}_{lang}_{tn.norm(tn.to_elot(s)) or 'x'}",
                     api.get(ep, **{param: s}, lang=lang, limit=100))
    for lang in ("el", "en"):
        for m in range(1, 13):
            save(f"calendar_{lang}_{m:02d}", api.get("knowledge/seasonal-calendar", month=m, lang=lang))
        mods = api.get("training/modules", lang=lang)
        save(f"modules_{lang}", mods)
        for mod in _normalise_result(mods)["results"]:
            for les in mod.get("lessons", []) or []:
                slug = les.get("slug") or les.get("id")
                if slug:
                    save(f"lesson_{lang}_{mod.get('slug')}_{slug}",
                         api.get("training/modules/{module}/lessons/{slug}",
                                 {"module": mod.get("slug", ""), "slug": slug}, lang=lang))
    for crop in ("olive", "koroneiki-olive", "grape-vine", "grapevine", "thyme", "oregano"):
        for lang in ("el", "en"):
            save(f"crop_{lang}_{crop}", api.get("crops/{type}", {"type": crop}, lang=lang))
    manifest = {"base_url": api.base_url, "calls": calls,
                "taken_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
