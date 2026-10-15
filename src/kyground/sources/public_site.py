"""Polite scraper for the public pages of the DASK (Kythera) platform.

Why a scraper when the platform has a data API?  The API requires a key, while
the public pages can be read, and re-read, by anyone. The study's frozen
snapshot is therefore a scrape of the public pages (``public_20261005``); a
dump of the API (``kyground fetch-api``, ``merge-api``) is optional and only
adds what the public pages lack, such as lesson texts. A later scrape, compared
with the frozen snapshot (``kyground snapshot-diff``), shows which records have
changed since.

Only record types exposed by the data API (Table 2 of the ORE article) are
collected: articles, FAQs, traditions, crops, seasonal calendar, glossary,
products, experiences, cooperatives and training modules/lessons. Lesson
bodies require a login and are therefore not collected here.

Usage (CLI): ``kyground scrape-public --out data/snapshots/public_YYYYMMDD``
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import requests
from bs4 import BeautifulSoup

from ..schema import Record, bil, write_jsonl
from .. import textnorm as tn

log = logging.getLogger(__name__)

BASE = "https://dask-platform.onrender.com"
USER_AGENT = "KyGround-research-harness/0.1 (DASK pilot evaluation; ICCS/NTUA)"
YEAR = 2026

# Glossary pages list Greek and English entries alphabetically in their own
# language, so the two lists cannot be paired by position. This table pairs
# the entries published at the time of writing; unknown entries are kept as
# single-language records and flagged in the quality report.
GLOSSARY_PAIRS = {
    "δακοσ": "Olive Fruit Fly",
    "ελαιοτριβειο": "Olive Mill",
    "κλαδεμα": "Pruning",
    "μελισσι": "Bee Colony",
    "οξυτητα": "Acidity",
    "τρυγοσ": "Grape Harvest",
}

TEST_TITLE_RE = re.compile(r"\b(test|τεστ|δοκιμ\w*|lorem)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

@dataclass
class Fetcher:
    out_dir: Path
    base: str = BASE
    delay: float = 1.0
    timeout: float = 40.0
    use_cache: bool = True
    manifest: list[dict] = field(default_factory=list)
    _last: float = 0.0

    def __post_init__(self) -> None:
        self.raw_dir = self.out_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "el,en;q=0.8"})

    def _cache_path(self, path: str) -> Path:
        key = hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", path).strip("_")[:80] or "root"
        return self.raw_dir / f"{safe}__{key}.html.gz"

    def get(self, path: str) -> str:
        cp = self._cache_path(path)
        if self.use_cache and cp.exists():
            return gzip.decompress(cp.read_bytes()).decode("utf-8")
        wait = self.delay - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        url = self.base + path
        for attempt in range(4):
            try:
                r = self.session.get(url, timeout=self.timeout)
                self._last = time.time()
                if r.status_code == 200:
                    html = r.text
                    cp.write_bytes(gzip.compress(html.encode("utf-8")))
                    self.manifest.append({
                        "path": path, "status": 200, "bytes": len(html.encode("utf-8")),
                        "sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
                        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    })
                    return html
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt * 2)
                    continue
                self.manifest.append({"path": path, "status": r.status_code})
                return ""
            except requests.RequestException as exc:  # pragma: no cover - network
                log.warning("fetch %s failed (%s), retrying", url, exc)
                time.sleep(2 ** attempt * 2)
        return ""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def _main(soup: BeautifulSoup):
    m = soup.find("main") or soup.body or soup
    for t in m(["script", "style", "svg", "noscript", "template"]):
        t.decompose()
    return m


def _lines(node) -> list[str]:
    txt = node.get_text("\n")
    return [re.sub(r"\s+", " ", ln).strip() for ln in txt.split("\n") if ln.strip()]


def _after(lines: list[str], label: str | Iterable[str], k: int = 1) -> str:
    labels = [label] if isinstance(label, str) else list(label)
    for i, ln in enumerate(lines):
        if ln in labels and i + k < len(lines):
            return lines[i + k]
    return ""


def _between(lines: list[str], start: Iterable[str], stops: Iterable[str]) -> list[str]:
    start, stops = list(start), list(stops)
    out: list[str] = []
    on = False
    for ln in lines:
        if not on and ln in start:
            on = True
            continue
        if on and ln in stops:
            break
        if on:
            out.append(ln)
    return out


def _last_index(lines: list[str], value: str) -> int:
    """Index of the last occurrence (the h1 follows the breadcrumb copy)."""
    idx = [i for i, ln in enumerate(lines) if ln == value]
    return idx[-1] if idx else -1


def _links(soup, pattern: str) -> list[str]:
    rx = re.compile(pattern)
    seen: list[str] = []
    for a in soup.find_all("a", href=True):
        h = a["href"].split("#")[0]
        if rx.match(h) and h not in seen:
            seen.append(h)
    return seen


def _h1(soup) -> str:
    h = soup.find("h1")
    return re.sub(r"\s+", " ", h.get_text(" ", strip=True)) if h else ""


def _jsonld(soup) -> list[dict]:
    out = []
    for js in soup.find_all("script", type="application/ld+json"):
        try:
            out.append(json.loads(js.get_text()))
        except Exception:
            pass
    return out


def _slugify(text: str) -> str:
    s = tn.norm(tn.to_elot(text))
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:60] or "x"


def _hours(text: str) -> float | None:
    d = tn.durations_hours(text)
    return d[0] if d else None


def _int_in(text: str) -> int | None:
    m = re.search(r"\d+", text or "")
    return int(m.group(0)) if m else None


def _en(path: str) -> str:
    return "/en" + path


def _section_text_after_heading(main, heading_texts: Iterable[str], stop_tags=("h2",)) -> str:
    """Concatenate the text of siblings after a heading until the next heading."""
    heading_texts = [h.strip() for h in heading_texts]
    for h in main.find_all(["h2", "h3"]):
        if h.get_text(" ", strip=True) in heading_texts:
            parts: list[str] = []
            for sib in h.find_next_siblings():
                if sib.name in stop_tags:
                    break
                t = sib.get_text("\n", strip=True)
                if t:
                    parts.append(t)
            return "\n".join(parts).strip()
    return ""


# ---------------------------------------------------------------------------
# Record-type scrapers
# ---------------------------------------------------------------------------

def scrape_articles(f: Fetcher) -> list[Record]:
    out: list[Record] = []
    lst = _soup(f.get("/knowledge/articles/"))
    slugs = [h for h in _links(lst, r"^/knowledge/articles/[a-z0-9-]+/$")
             if "/category/" not in h and "/tag/" not in h]
    for path in slugs:
        slug = path.rstrip("/").split("/")[-1]
        data: dict[str, dict] = {}
        for lang, p in (("el", path), ("en", _en(path))):
            html = f.get(p)
            if not html:
                continue
            soup = _soup(html)
            ld = next((d for d in _jsonld(soup) if d.get("@type") == "Article"), {})
            main = _main(soup)
            hero = main.find("section") or main
            summary = ""
            h1 = hero.find("h1")
            if h1:
                p_el = h1.find_next("p")
                summary = p_el.get_text(" ", strip=True) if p_el else ""
            badge = h1.find_previous("span") if h1 else None
            art = main.find("article")
            body = ""
            tags: list[str] = []
            if art:
                for blk in art.find_all(["h3"]):
                    if blk.get_text(strip=True) in ("Ετικέτες", "Tags", "Κοινοποίηση", "Share"):
                        par = blk.find_parent()
                        if blk.get_text(strip=True) in ("Ετικέτες", "Tags") and par:
                            tags = [a.get_text(" ", strip=True) for a in par.find_all("a")]
                        if par:
                            par.decompose()
                body = "\n".join(_lines(art))
            data[lang] = dict(
                title=_h1(soup), summary=summary, body=body,
                category=badge.get_text(" ", strip=True) if badge else "",
                tags=tags, modified=ld.get("dateModified", ""),
            )
        if "el" not in data:
            continue
        el, en = data.get("el", {}), data.get("en", {})
        out.append(Record(
            rid=f"article:{slug}", rtype="article", slug=slug,
            title=bil(el.get("title"), en.get("title")),
            summary=bil(el.get("summary"), en.get("summary")),
            body=bil(el.get("body"), en.get("body")),
            attrs={"category": bil(el.get("category"), en.get("category"))},
            tags=el.get("tags", []), url=BASE + path, updated_at=el.get("modified", ""),
            source="public_site",
        ))
    return out


def _parse_faq(html: str) -> list[tuple[str, str, str]]:
    soup = _soup(html)
    main = _main(soup)
    cats = [h.get_text(" ", strip=True) for h in main.find_all("h2")]
    lines = _lines(main)
    out: list[tuple[str, str, str]] = []
    cat = ""
    q: str | None = None
    ans: list[str] = []
    stop = {"Δεν βρήκατε αυτό που ψάχνετε;", "Didn't find what you were looking for?",
            "Didn't find what you're looking for?", "Can't find what you're looking for?"}
    started = False
    for ln in lines:
        if ln in stop:
            break
        if ln in cats:
            if q:
                out.append((cat, q, " ".join(ans).strip()))
            q, ans, cat, started = None, [], ln, True
            continue
        if not started:
            continue
        if ln.endswith(("?", ";", ";")) and len(ln) > 8:
            if q:
                out.append((cat, q, " ".join(ans).strip()))
            q, ans = ln, []
        elif q:
            ans.append(ln)
    if q:
        out.append((cat, q, " ".join(ans).strip()))
    return out


FAQ_CATEGORY_PAIRS = {
    "βιολογικη γεωργια": "Organic Farming",
    "ελαιοκαλλιεργεια": "Olive Cultivation",
    "μελισσοκομια": "Beekeeping",
    "γενικα": "General",
    "αμπελουργια": "Viticulture",
}


def _pair_faq(el: list[tuple[str, str, str]], en: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Pair Greek and English FAQ entries.

    The English page orders categories differently, so entries are paired by
    (category, position within category) using a category translation table.
    """
    by_cat_en: dict[str, list[tuple[str, str, str]]] = {}
    for e in en:
        by_cat_en.setdefault(e[0], []).append(e)
    pos: dict[str, int] = {}
    out = []
    for cat, _q, _a in el:
        en_cat = FAQ_CATEGORY_PAIRS.get(tn.norm(cat), "")
        k = pos.get(cat, 0)
        pos[cat] = k + 1
        lst = by_cat_en.get(en_cat, [])
        out.append(lst[k] if k < len(lst) else ("", "", ""))
    return out


def scrape_faq(f: Fetcher) -> list[Record]:
    el = _parse_faq(f.get("/knowledge/faq/"))
    en = _pair_faq(el, _parse_faq(f.get(_en("/knowledge/faq/"))))
    out: list[Record] = []
    for i, (cat, q, a) in enumerate(el):
        e = en[i]
        slug = f"{i + 1:02d}-{_slugify(q)[:40]}"
        out.append(Record(
            rid=f"faq:{slug}", rtype="faq", slug=slug,
            title=bil(q, e[1]), body=bil(a, e[2]),
            attrs={"category": bil(cat, e[0])}, url=BASE + "/knowledge/faq/", source="public_site",
        ))
    return out


def scrape_traditions(f: Fetcher) -> list[Record]:
    lst = _soup(f.get("/knowledge/traditions/"))
    out: list[Record] = []
    for path in _links(lst, r"^/knowledge/traditions/[a-z0-9-]+/$"):
        slug = path.rstrip("/").split("/")[-1]
        d: dict[str, dict] = {}
        for lang, p in (("el", path), ("en", _en(path))):
            html = f.get(p)
            if not html:
                continue
            soup = _soup(html)
            main = _main(soup)
            lines = _lines(main)
            title = _h1(soup)
            ti = _last_index(lines, title)
            desc = lines[ti + 1] if 0 <= ti < len(lines) - 1 else ""
            if desc in ("Πληροφορίες", "Information"):
                desc = ""
            ptype = _after(lines, ("Τύπος Πρακτικής", "Practice Type"))
            if ptype in ("Κοινοποίηση", "Share", "Πίσω στις Παραδοσιακές Πρακτικές"):
                ptype = ""
            d[lang] = dict(title=title, body=desc, ptype=ptype)
        if "el" not in d:
            continue
        el, en = d["el"], d.get("en", {})
        out.append(Record(
            rid=f"tradition:{slug}", rtype="tradition", slug=slug,
            title=bil(el["title"], en.get("title")), body=bil(el["body"], en.get("body")),
            attrs={"practice_type": bil(el.get("ptype"), en.get("ptype"))},
            url=BASE + path, source="public_site",
        ))
    return out


def scrape_crops(f: Fetcher) -> list[Record]:
    lst = _soup(f.get("/knowledge/crops/"))
    out: list[Record] = []
    for path in _links(lst, r"^/knowledge/crops/[a-z0-9-]+/$"):
        slug = path.rstrip("/").split("/")[-1]
        d: dict[str, dict] = {}
        for lang, p in (("el", path), ("en", _en(path))):
            html = f.get(p)
            if not html:
                continue
            soup = _soup(html)
            main = _main(soup)
            lines = _lines(main)
            prose = main.find("div", class_=re.compile(r"\bprose\b"))
            dd = {dt.get_text(" ", strip=True): dt.find_next("dd").get_text(" ", strip=True)
                  for dt in main.find_all("dt") if dt.find_next("dd")}
            d[lang] = dict(
                title=_h1(soup),
                body=prose.get_text("\n", strip=True) if prose else "",
                sci=dd.get("Επιστημονικό Όνομα") or dd.get("Scientific Name") or "",
                ctype=dd.get("Τύπος") or dd.get("Type") or "",
                planting=_after(lines, ("Περίοδος Φύτευσης", "Planting Period")),
                harvest=_after(lines, ("Περίοδος Συγκομιδής", "Harvest Period")),
            )
        if "el" not in d:
            continue
        el, en = d["el"], d.get("en", {})
        attrs = {
            "crop_type": bil(el["ctype"], en.get("ctype")),
            "scientific_name": el["sci"] or en.get("sci", ""),
            "planting_period": bil(el["planting"], en.get("planting")),
            "harvest_period": bil(el["harvest"], en.get("harvest")),
        }
        for key, txt in (("planting_months", el["planting"]), ("harvest_months", el["harvest"])):
            rng = tn.parse_month_range(txt)
            if rng:
                attrs[key] = tn.months_between(*rng)
        out.append(Record(
            rid=f"crop:{slug}", rtype="crop", slug=slug,
            title=bil(el["title"], en.get("title")), body=bil(el["body"], en.get("body")),
            attrs=attrs, url=BASE + path, source="public_site",
        ))
    return out


def _parse_calendar(html: str) -> list[tuple[str, str, str]]:
    soup = _soup(html)
    main = _main(soup)
    out: list[tuple[str, str, str]] = []
    for h in main.find_all("h2"):
        if h.get_text(" ", strip=True) in ("Δραστηριότητες Μήνα", "Monthly Activities"):
            cont = h.find_next_sibling()
            if not cont:
                continue
            for h3 in cont.find_all("h3"):
                card = h3.find_parent("div")
                a = card.find("a") if card else None
                if a is None:
                    continue  # "no activities" placeholder card
                crop = a.get_text(" ", strip=True)
                crop_slug = a["href"].rstrip("/").split("/")[-1] if a and a.get("href") else ""
                out.append((h3.get_text(" ", strip=True), crop, crop_slug))
    return out


def scrape_calendar(f: Fetcher) -> list[Record]:
    out: list[Record] = []
    for m in range(1, 13):
        q = f"/knowledge/calendar/?month={m}&year={YEAR}"
        el = _parse_calendar(f.get(q))
        en = _parse_calendar(f.get(_en(q)))
        for i, (act, crop, cslug) in enumerate(el):
            e = en[i] if i < len(en) and len(en) == len(el) else ("", "", "")
            slug = f"m{m:02d}-{cslug or _slugify(crop)}-{_slugify(act)[:30]}"
            out.append(Record(
                rid=f"calendar:{slug}", rtype="calendar", slug=slug,
                title=bil(f"{act} — {crop}", f"{e[0]} — {e[1]}" if e[0] else None),
                attrs={"month": m, "activity": bil(act, e[0]), "crop_name": bil(crop, e[1]),
                       "crop_slug": cslug},
                url=BASE + q, source="public_site",
            ))
    return out


def _parse_glossary(html: str) -> list[tuple[str, str]]:
    soup = _soup(html)
    main = _main(soup)
    root = main.find(id="glossary-terms") or main
    out = []
    for h3 in root.find_all("h3"):
        p = h3.find_next("p")
        out.append((h3.get_text(" ", strip=True), p.get_text(" ", strip=True) if p else ""))
    return out


def scrape_glossary(f: Fetcher) -> list[Record]:
    el = _parse_glossary(f.get("/knowledge/glossary/"))
    en = {t: d for t, d in _parse_glossary(f.get(_en("/knowledge/glossary/")))}
    out: list[Record] = []
    for term, definition in el:
        en_term = GLOSSARY_PAIRS.get(tn.norm(term))
        en_def = en.get(en_term, "") if en_term else ""
        slug = _slugify(term)
        out.append(Record(
            rid=f"glossary:{slug}", rtype="glossary", slug=slug,
            title=bil(term, en_term), body=bil(definition, en_def),
            attrs={"paired_en": bool(en_term and en_def)},
            url=BASE + "/knowledge/glossary/", source="public_site",
        ))
    return out


def _product_paths(f: Fetcher) -> tuple[list[str], dict[str, dict]]:
    first = _soup(f.get("/shop/catalogue/"))
    paths = _links(first, r"^/shop/catalogue/[^/]+_\d+/$")
    # pagination (if any)
    for pg in range(2, 10):
        pages = _links(first, rf"^/shop/catalogue/\?page={pg}$")
        if not pages:
            break
        s = _soup(f.get(pages[0]))
        paths += [p for p in _links(s, r"^/shop/catalogue/[^/]+_\d+/$") if p not in paths]
    # category membership
    cats: dict[str, dict] = {}
    for cpath in _links(first, r"^/shop/catalogue/category/[^/]+_\d+/$"):
        s_el = _soup(f.get(cpath))
        s_en = _soup(f.get(_en(cpath)))
        name_el = _h1(s_el) or cpath
        name_en = _h1(s_en)
        for p in _links(s_el, r"^/shop/catalogue/[^/]+_\d+/$"):
            cats.setdefault(p, bil(name_el, name_en))
            if p not in paths:
                paths.append(p)
    return paths, cats


def scrape_products(f: Fetcher) -> list[Record]:
    paths, cats = _product_paths(f)
    out: list[Record] = []
    for path in paths:
        slug = path.rstrip("/").split("/")[-1]
        d: dict[str, dict] = {}
        for lang, p in (("el", path), ("en", _en(path))):
            html = f.get(p)
            if not html:
                continue
            soup = _soup(html)
            main = _main(soup)
            card = main.find("div", class_="card-body") or main
            title_el = card.find("h1") or main.find("h1")
            title = title_el.get_text(" ", strip=True) if title_el else ""
            lines = _lines(card)
            producer = next((ln.split(" ", 1)[1].strip() for ln in lines if ln.startswith(("Από ", "By "))), "")
            sku = next((ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("SKU")), "")
            price_txt = next((ln for ln in lines if "€" in ln or "κατόπιν" in ln.lower() or "on request" in ln.lower()), "")
            avail = next((ln for ln in lines if ln in ("Διαθέσιμο", "Available", "Μη διαθέσιμο", "Not available",
                                                         "Out of stock", "Εξαντλήθηκε")), "")
            desc = _section_text_after_heading(main, ("Περιγραφή", "Description"), stop_tags=("h2", "h3"))
            mlines = _lines(main)
            meet = _after(mlines, ("Γνωρίστε τον παραγωγό", "Meet the producer"))
            d[lang] = dict(title=title, producer=producer, sku=sku, price_txt=price_txt, avail=avail,
                           desc=desc, meet=meet)
        if "el" not in d:
            continue
        el, en = d["el"], d.get("en", {})
        amounts = tn.currency_amounts(el["price_txt"])
        rec = Record(
            rid=f"product:{slug}", rtype="product", slug=slug,
            title=bil(el["title"], en.get("title")), body=bil(el["desc"], en.get("desc")),
            attrs={
                "producer": bil(el["producer"], en.get("producer")),
                "producer_contact_person": el["meet"],
                "sku": el["sku"],
                "availability": bil(el["avail"], en.get("avail")),
                "category": cats.get(path, {}),
                "price_display": bil(el["price_txt"], en.get("price_txt")),
            },
            price=amounts[0] if amounts else None,
            price_visible=bool(amounts),
            url=BASE + path, source="public_site",
        )
        out.append(rec)
    return out


def scrape_experiences(f: Fetcher) -> list[Record]:
    lst_html = f.get("/market/experiences/")
    lst = _soup(lst_html)
    markers: dict[str, dict] = {}
    node = lst.find(attrs={"data-markers": True})
    if node:
        try:
            for m in json.loads(node["data-markers"]):
                markers[m.get("url", "")] = m
        except Exception:
            pass
    out: list[Record] = []
    paths = [h for h in _links(lst, r"^/market/experiences/[a-z0-9-]+/$") if "/category/" not in h]
    for path in paths:
        slug = path.rstrip("/").split("/")[-1]
        d: dict[str, dict] = {}
        for lang, p in (("el", path), ("en", _en(path))):
            html = f.get(p)
            if not html:
                continue
            soup = _soup(html)
            main = _main(soup)
            lines = _lines(main)
            title = _h1(soup)
            ti = _last_index(lines, title)
            category = lines[ti - 1] if ti > 0 else ""
            location = lines[ti + 1] if 0 <= ti < len(lines) - 1 else ""
            if location in ("Περιγραφή", "Description"):
                location = ""
            desc = _section_text_after_heading(main, ("Περιγραφή", "Description"))
            hb = _between(lines, ("Οικοδεσπότης", "Host"),
                          ("Επικοινωνία με τον οικοδεσπότη", "Contact the host", "Κριτικές", "Reviews"))
            hb = [x for x in hb if len(x) > 1]          # drop avatar initials
            host = hb[0] if hb else ""
            host_village = hb[1] if len(hb) > 1 else ""
            dur = next((ln for ln in lines if re.match(r"^\d+([.,]\d+)?\s*(ώρες|ώρα|hours?)$", ln)), "")
            cap = next((ln for ln in lines if re.match(r"^(Έως|Up to)\s+\d+", ln)), "")
            price_txt = next((ln for ln in lines if "€" in ln or ln.startswith(("Τιμή κατόπιν", "Price on request"))), "")
            notes = [ln.lstrip("* ").strip() for ln in desc.split("\n") if ln.strip().startswith("*")]
            d[lang] = dict(title=title, category=category, location=location, desc=desc, host=host,
                           host_village=host_village, dur=dur, cap=cap, price_txt=price_txt, notes=notes)
        if "el" not in d:
            continue
        el, en = d["el"], d.get("en", {})
        mk = markers.get(path, {})
        amounts = tn.currency_amounts(el["price_txt"])
        out.append(Record(
            rid=f"experience:{slug}", rtype="experience", slug=slug,
            title=bil(el["title"], en.get("title")), body=bil(el["desc"], en.get("desc")),
            attrs={
                "category": bil(el["category"], en.get("category")),
                "location_name": bil(el["location"], en.get("location")),
                "village": mk.get("subtitle", el["location"]),
                "host": el["host"],
                "host_village": el["host_village"],
                "duration_hours": _hours(el["dur"]),
                "max_people": _int_in(el["cap"]),
                "availability_note": bil(" ".join(el["notes"]), " ".join(en.get("notes", []))),
                "price_display": bil(el["price_txt"], en.get("price_txt")),
            },
            price=amounts[0] if amounts else None, price_visible=bool(amounts),
            lat=mk.get("lat"), lon=mk.get("lng"), url=BASE + path, source="public_site",
        ))
    return out


def scrape_modules(f: Fetcher) -> list[Record]:
    lst = _soup(f.get("/training/modules/"))
    out: list[Record] = []
    for path in _links(lst, r"^/training/modules/[a-z0-9-]+/$"):
        slug = path.rstrip("/").split("/")[-1]
        d: dict[str, dict] = {}
        for lang, p in (("el", path), ("en", _en(path))):
            html = f.get(p)
            if not html:
                continue
            soup = _soup(html)
            main = _main(soup)
            lines = _lines(main)
            title = _h1(soup)
            ti = _last_index(lines, title)
            category = lines[ti - 2] if ti >= 2 else ""
            level = lines[ti - 1] if ti >= 1 else ""
            dur = next((ln for ln in lines[ti:ti + 6] if re.match(r"^\d+\s*(λεπτά|minutes|min)$", ln)), "")
            nles = next((ln for ln in lines[ti:ti + 6] if re.match(r"^\d+\s*(μαθήματα|μάθημα|lessons?)$", ln)), "")
            prose = main.find("div", class_=re.compile(r"\bprose\b"))
            objectives = _between(lines, ("Μαθησιακοί Στόχοι", "Learning Objectives", "Learning Outcomes"),
                                  ("Περιεχόμενο Ενότητας", "Module Content", "Module contents", "Course Content"))
            content = _between(lines, ("Περιεχόμενο Ενότητας", "Module Content", "Module contents", "Course Content"),
                               ("Συνδεθείτε για να ξεκινήσετε", "Sign in to start", "Log in to start",
                                "Ξεκινήστε", "Start module", "Enrol", "Εγγραφή"))
            lessons = []
            i = 0
            while i < len(content):
                if re.fullmatch(r"\d+", content[i]) and i + 1 < len(content):
                    l_title = content[i + 1]
                    kind = content[i + 2] if i + 2 < len(content) else ""
                    ldur = content[i + 3] if i + 3 < len(content) else ""
                    lessons.append({"position": int(content[i]), "title": l_title, "kind": kind,
                                    "duration_min": _int_in(ldur) if re.search(r"λεπτ|min", ldur) else None})
                    i += 4
                else:
                    i += 1
            d[lang] = dict(title=title, category=category, level=level, dur=dur, nles=nles,
                           desc=prose.get_text("\n", strip=True) if prose else "",
                           objectives=objectives, lessons=lessons)
        if "el" not in d:
            continue
        el, en = d["el"], d.get("en", {})
        lessons = []
        for j, les in enumerate(el["lessons"]):
            les_en = en.get("lessons", [])[j] if j < len(en.get("lessons", [])) else {}
            lessons.append({"position": les["position"], "title": bil(les["title"], les_en.get("title")),
                            "kind": bil(les["kind"], les_en.get("kind")), "duration_min": les["duration_min"]})
        rec = Record(
            rid=f"module:{slug}", rtype="module", slug=slug,
            title=bil(el["title"], en.get("title")), body=bil(el["desc"], en.get("desc")),
            attrs={
                "category": bil(el["category"], en.get("category")),
                "level": bil(el["level"], en.get("level")),
                "duration_min": _int_in(el["dur"]),
                "lesson_count": _int_in(el["nles"]),
                "objectives": bil("\n".join(el["objectives"]), "\n".join(en.get("objectives", []))),
                "lessons": lessons,
            },
            url=BASE + path, source="public_site",
        )
        out.append(rec)
        for les in lessons:
            lslug = f"{slug}--{les['position']:02d}"
            out.append(Record(
                rid=f"lesson:{lslug}", rtype="lesson", slug=lslug, title=les["title"],
                attrs={"module_slug": slug, "position": les["position"], "duration_min": les["duration_min"],
                       "kind": les["kind"], "content_available": False},
                parent=rec.rid, url=BASE + path, source="public_site",
            ))
    return out


def scrape_cooperatives(f: Fetcher) -> list[Record]:
    soup = _soup(f.get("/market/cooperatives/"))
    out: list[Record] = []
    for path in _links(soup, r"^/market/cooperatives/[a-z0-9-]+/$"):
        slug = path.rstrip("/").split("/")[-1]
        html = f.get(path)
        s = _soup(html)
        out.append(Record(rid=f"cooperative:{slug}", rtype="cooperative", slug=slug,
                          title=bil(_h1(s)), body=bil("\n".join(_lines(_main(s))[:40])),
                          url=BASE + path, source="public_site"))
    return out


SCRAPERS: dict[str, Callable[[Fetcher], list[Record]]] = {
    "article": scrape_articles,
    "faq": scrape_faq,
    "tradition": scrape_traditions,
    "crop": scrape_crops,
    "calendar": scrape_calendar,
    "glossary": scrape_glossary,
    "product": scrape_products,
    "experience": scrape_experiences,
    "cooperative": scrape_cooperatives,
    "module": scrape_modules,
}


# ---------------------------------------------------------------------------
# Quality flags and orchestration
# ---------------------------------------------------------------------------

def _drop_untranslated(r: Record) -> list[str]:
    """The English pages fall back to Greek text when no translation exists.
    Store such fields as Greek-only so that the record reflects the content."""
    fields: list[str] = []

    def fix(d: dict, name: str) -> None:
        if isinstance(d, dict) and d.get("en") and d.get("el") and d["en"] == d["el"] and tn.is_greek(d["en"]):
            del d["en"]
            fields.append(name)

    fix(r.title, "title")
    fix(r.body, "body")
    fix(r.summary, "summary")
    for k, v in r.attrs.items():
        if isinstance(v, dict):
            fix(v, k)
    for les in r.attrs.get("lessons", []) or []:
        fix(les.get("title", {}), "lesson title")
    return fields


def flag_quality(records: list[Record]) -> list[dict]:
    """Mark test records as excluded and group near-duplicate titles.

    Returns a list of human-readable issues for the quality report.
    """
    issues: list[dict] = []
    for r in records:
        untranslated = _drop_untranslated(r)
        if untranslated and r.rtype not in ("lesson",):
            issues.append({"rid": r.rid, "issue": "English shows Greek text (untranslated): " + ", ".join(untranslated),
                           "title": " ".join(r.title.values())})
        t = " ".join(r.title.values())
        # Title-based only: slugs are unreliable (e.g. a real 10-lesson module has slug "test-1").
        if TEST_TITLE_RE.search(t):
            r.exclude_reason = "test-record"
            issues.append({"rid": r.rid, "issue": "test record excluded", "title": t})
        cat = r.attrs.get("category")
        if isinstance(cat, dict) and any(TEST_TITLE_RE.search(v or "") for v in cat.values()):
            issues.append({"rid": r.rid, "issue": "record is filed under a test category", "title": t})
        if TEST_TITLE_RE.search(r.slug.replace("-", " ")) and not r.exclude_reason and r.rtype != "lesson":
            issues.append({"rid": r.rid, "issue": "placeholder-looking URL slug (content is real; consider renaming)",
                           "title": t})
        if r.rtype not in ("calendar", "lesson") and not r.title.get("en") and "title" not in untranslated:
            issues.append({"rid": r.rid, "issue": "no English title (Greek fallback applies)", "title": t})
        if r.rtype == "glossary" and not r.attrs.get("paired_en"):
            issues.append({"rid": r.rid, "issue": "glossary entry without English pairing", "title": t})
    # near-duplicates within a type: identical normalised titles, or one title a prefix of another
    by_type: dict[str, list[Record]] = {}
    for r in records:
        if r.exclude_reason or r.rtype in ("calendar", "lesson"):
            continue
        by_type.setdefault(r.rtype, []).append(r)
    for rtype, recs in by_type.items():
        for i, a in enumerate(recs):
            for b in recs[i + 1:]:
                ta, tb = tn.norm(a.title.get("el", "")), tn.norm(b.title.get("el", ""))
                if not ta or not tb:
                    continue
                short, long_ = sorted((ta, tb), key=len)
                if ta == tb or (len(short.split()) >= 3 and long_.startswith(short + " ")):
                    canon = min(a.rid, b.rid)
                    a.attrs.setdefault("dup_group", canon)
                    b.attrs["dup_group"] = a.attrs["dup_group"]
                    issues.append({"rid": f"{a.rid} | {b.rid}", "issue": "near-duplicate records",
                                   "title": f"{a.title.get('el')} | {b.title.get('el')}"})
    return issues


def scrape(out_dir: str | Path, delay: float = 1.0, types: Iterable[str] | None = None,
           use_cache: bool = True) -> list[Record]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = Fetcher(out_dir=out, delay=delay, use_cache=use_cache)
    records: list[Record] = []
    for rtype in (types or SCRAPERS.keys()):
        log.info("scraping %s", rtype)
        try:
            got = SCRAPERS[rtype](f)
        except Exception as exc:  # keep going; report the failure
            log.exception("scraper %s failed: %s", rtype, exc)
            got = []
        records.extend(got)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for r in records:
        r.updated_at = r.updated_at or now
    issues = flag_quality(records)
    write_jsonl(out / "records.jsonl", records)
    counts: dict[str, dict[str, int]] = {}
    for r in records:
        c = counts.setdefault(r.rtype, {"total": 0, "public": 0, "excluded": 0})
        c["total"] += 1
        c["public" if r.is_public else "excluded"] += 1
    raw = sorted((out / "raw").glob("*.gz")) if (out / "raw").exists() else []
    fetched = (datetime.fromtimestamp(min(p.stat().st_mtime for p in raw), timezone.utc).isoformat(timespec="seconds")
               if raw else now)
    manifest = {
        "source": "public_site", "base": BASE, "scraped_at": fetched, "parsed_at": now, "delay_s": delay,
        "counts": counts, "pages": f.manifest,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_quality_report(out / "quality_report.md", records, issues, counts)
    return records


def write_quality_report(path: Path, records: list[Record], issues: list[dict], counts: dict) -> None:
    lines = ["# Knowledge-base quality report (public snapshot)", "",
             "Generated by `kyground scrape-public`: issues found in the platform's published content.", "",
             "## Record counts", "", "| Type | Total | Public (used) | Excluded |", "|---|---:|---:|---:|"]
    for k in sorted(counts):
        c = counts[k]
        lines.append(f"| {k} | {c['total']} | {c['public']} | {c['excluded']} |")
    lines += ["", "## Issues found", "", "| Record(s) | Issue | Title |", "|---|---|---|"]
    for it in issues:
        lines.append(f"| `{it['rid']}` | {it['issue']} | {it['title'][:90]} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
