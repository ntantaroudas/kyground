"""Probe of the platform's public search, which needs no credentials.

The data interface that the tool agent calls requires a key. The platform's public training-module search
(``/training/modules/?search=...``) does not, and probing it shows how the platform matches free text: whether
case, final sigma, accents, word order and inflection matter, whether a query may be part of a word, and whether
Greeklish or English words are found. The battery is built from the Greek titles of the training modules in the
snapshot (each probe word is checked against the snapshot), so that every probe has a known target.

A probe "hits" when its module is among the results. The search also covers the modules' descriptions, so every
query that should find nothing unless the search normalises (unaccented and capitalised forms, other inflections,
Greeklish, reversed word order) is checked to occur in none of the modules' texts in the snapshot; a hit can then
only come from normalisation. The inferred behaviour sets the replica's default free-text mode
(``replica.search_mode``); E4 measures how much the choice matters. Whether the data interface behaves like the public search is checked, when the interface can be
reached, by ``kyground probe-api`` and ``kyground conformance``.

Usage (CLI): ``kyground probe-public --out runs/public_probe``
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from .. import textnorm as tn
from ..schema import Record
from .public_site import BASE, USER_AGENT

SEARCH_URL = BASE + "/training/modules/"
_SLUG_RE = re.compile(r'href="/training/modules/([a-z0-9-]+)/"')

# (module slug, a word of the module's Greek title, another inflection of that word that is not a substring of
#  the title, two consecutive title words, an English title word)
PROBE_SPEC: list[tuple[str, str, str, tuple[str, str], str]] = [
    ("sustainable-farming-for-small-producers", "Παραγωγούς", "Παραγωγοί", ("Μικρούς", "Παραγωγούς"), "Producers"),
    ("getting-started-with-the-dask-platform", "Πλατφόρμα", "Πλατφορμών", ("Πλατφόρμα", "DASK"), "Platform"),
    ("ai-tools-for-small-local-businesses", "Επιχειρήσεις", "Επιχειρήσεων", ("Τοπικές", "Επιχειρήσεις"),
     "Businesses"),
    ("test-1", "Αγροτουρισμό", "Αγροτουρισμός", ("Ψηφιακή", "Επιχειρηματικότητα"), "Agritourism"),
]

# families that can find their module only if the search normalises: their queries must occur in no module text
NORMALISATION_FAMILIES = ("no_accents", "capitals_no_accents", "other_inflection", "greeklish", "two_words_reversed")


def variants(word: str, other_inflection: str, pair: tuple[str, str], english: str) -> dict[str, str]:
    """Probe queries for one title word, by variant family."""
    v = {
        "canonical": word,
        "lower": word.lower(),
        "upper": word.upper(),
        "no_accents": tn.strip_accents(word).lower(),
        "capitals_no_accents": tn.strip_accents(word).upper(),
    }
    if word.endswith("ς"):
        v["final_sigma_as_medial"] = word[:-1] + "σ"
    v.update({
        "prefix": word[:max(4, int(0.6 * len(word)))],
        "infix": word[2:],
        "other_inflection": other_inflection,
        "greeklish": tn.to_elot(word),
        "two_words": " ".join(pair),
        "two_words_reversed": " ".join(reversed(pair)),
        "english": english,
    })
    return v


def battery(records: Iterable[Record], spec: list = PROBE_SPEC) -> list[dict]:
    """Probe rows for the modules of the snapshot; fails if a probe word is not in its module's title."""
    from ..kb.serialize import record_text
    mods = [r for r in records if r.rtype == "module"]
    titles = {r.slug: r.title for r in mods}
    texts = [tn.casefold_only(record_text(r) + " " + json.dumps(r.attrs, ensure_ascii=False)) for r in mods]
    rows: list[dict] = []
    for slug, word, infl, pair, en in spec:
        t = titles.get(slug)
        if not t:
            raise ValueError(f"module {slug} is not in the snapshot")
        el, en_title = t.get("el", ""), t.get("en", "")
        for w in (word, *pair):
            if w not in el:
                raise ValueError(f"probe word {w!r} is not in the title {el!r}")
        if infl in el:
            raise ValueError(f"other inflection {infl!r} occurs in the title {el!r}")
        if en.lower() not in en_title.lower():
            raise ValueError(f"English probe word {en!r} is not in the title {en_title!r}")
        for fam, q in variants(word, infl, pair, en).items():
            if fam in NORMALISATION_FAMILIES and any(tn.casefold_only(q) in t for t in texts):
                raise ValueError(f"{fam} probe {q!r} occurs in a module text and could match without normalisation")
            rows.append({"module": slug, "word": word, "family": fam, "query": q})
    return rows


def http_fetch(delay: float = 1.0, timeout: float = 40.0) -> Callable[[str], dict]:
    """A polite fetcher for the public module search: returns {status, slugs}."""
    import requests
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "el,en;q=0.8"})
    last = [0.0]

    def fetch(query: str) -> dict:
        wait = delay - (time.time() - last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            r = session.get(SEARCH_URL, params={"search": query}, timeout=timeout)
            last[0] = time.time()
            return {"status": r.status_code, "slugs": sorted(set(_SLUG_RE.findall(r.text)))}
        except requests.RequestException as exc:
            last[0] = time.time()
            return {"status": None, "slugs": [], "error": f"{type(exc).__name__}: {exc}"}

    return fetch


def run_battery(rows: list[dict], fetch: Callable[[str], dict]) -> list[dict]:
    """Run every probe; a probe hits when its module is among the results."""
    out: list[dict] = []
    for row in rows:
        res = fetch(row["query"])
        slugs = res.get("slugs", [])
        out.append({**row, "status": res.get("status"), "n": len(slugs), "slugs": " ".join(slugs),
                    "hit": int(row["module"] in slugs), "error": res.get("error", "")})
    return out


PROPERTIES = {  # property -> variant families that show it
    "case_insensitive": ("lower", "upper"),
    "final_sigma_folded": ("final_sigma_as_medial",),
    "accent_insensitive": ("no_accents", "capitals_no_accents"),
    "matches_part_of_word": ("prefix", "infix"),
    "word_order_insensitive": ("two_words_reversed",),
    "inflection_insensitive": ("other_inflection",),
    "greeklish_matched": ("greeklish",),
    "english_title_matched": ("english",),
}


def infer(results: list[dict]) -> dict:
    """Matching properties (share of probes that hit, by family) and the replica mode they imply."""
    valid = [r for r in results if r.get("status") == 200]
    canon_ok = [r for r in valid if r["family"] == "canonical" and r["hit"]]
    modules = {r["module"] for r in canon_ok}
    props: dict[str, dict] = {}
    for prop, fams in PROPERTIES.items():
        rs = [r for r in valid if r["family"] in fams and r["module"] in modules]
        hits = sum(r["hit"] for r in rs)
        props[prop] = {"hits": hits, "probes": len(rs), "holds": bool(rs) and hits > len(rs) / 2}
    p = {k: v["holds"] for k, v in props.items()}
    if p["inflection_insensitive"] or p["word_order_insensitive"]:
        mode = "tokens_all_stem"
    elif p["accent_insensitive"]:
        mode = "phrase_unaccent"
    else:
        mode = "phrase_casefold"
    notes = []
    if not p["case_insensitive"]:
        notes.append("matching is case-sensitive, which no replica mode reproduces (phrase_casefold is closest)")
    if p["word_order_insensitive"] and not p["inflection_insensitive"]:
        notes.append("word order is ignored but inflection is not; tokens_all_stem also matches inflections")
    if not p["matches_part_of_word"]:
        notes.append("whole-word matching; the replica matches substrings")
    return {"modules_found_by_canonical_query": len(modules), "properties": props, "recommended_search_mode": mode,
            "notes": notes}


def probe_public(records: list[Record], out_dir: Path, fetch: Callable[[str], dict] | None = None,
                 delay: float = 1.0) -> dict:
    """Run the battery against the public search and write ``public_search_probe.csv`` and ``.json``."""
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = battery(records)
    res = run_battery(rows, fetch or http_fetch(delay))
    summary = {"url": SEARCH_URL + "?search=<query>", "probed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "n_probes": len(res), **infer(res)}
    with (out_dir / "public_search_probe.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["module", "word", "family", "query", "status", "n", "slugs", "hit", "error"])
        w.writeheader()
        w.writerows(res)
    (out_dir / "public_search_probe.json").write_text(json.dumps({**summary, "probes": res}, ensure_ascii=False,
                                                                 indent=1), encoding="utf-8")
    return summary
