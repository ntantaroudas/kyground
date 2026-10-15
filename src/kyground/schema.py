"""Canonical data model shared by every part of the harness.

Records are the unit of the knowledge base (KB). They are deliberately
source-agnostic: the public-site scraper, the live data-API adapter and the
synthetic generators all produce the same :class:`Record` shape, so the
replica API, the vector index, the full-context serialiser and the benchmark
builder never need to know where a record came from.

Benchmark items carry *nuggets*: atomic, typed facts that a correct answer
must contain. Each item is expanded into several *queries*, one per surface
form (canonical Greek, English, accentless Greek, Greeklish, ...).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

LANGS = ("el", "en")

# Record types exposed by the read-only data API (Table 2 of the ORE paper).
RTYPES = (
    "article",      # knowledge/search
    "faq",          # knowledge/search
    "tradition",    # knowledge/search
    "crop",         # crops/{type}
    "calendar",     # knowledge/seasonal-calendar
    "glossary",     # glossary/lookup
    "product",      # products/search
    "experience",   # experiences/search (spatial)
    "cooperative",  # cooperatives/search (spatial)
    "module",       # training/modules
    "lesson",       # training/modules/{module}/lessons/{slug}
)

KNOWLEDGE_TYPES = ("article", "faq", "tradition")


def bil(el: str | None = None, en: str | None = None) -> dict[str, str]:
    """Build a bilingual text dict, dropping empty values."""
    out: dict[str, str] = {}
    if el:
        out["el"] = el.strip()
    if en:
        out["en"] = en.strip()
    return out


def pick(texts: dict[str, str] | None, lang: str, fallback: str = "el") -> str:
    """Return the text in ``lang`` with Greek fallback (the platform's rule)."""
    if not texts:
        return ""
    if texts.get(lang):
        return texts[lang]
    if texts.get(fallback):
        return texts[fallback]
    for v in texts.values():
        if v:
            return v
    return ""


@dataclass
class Record:
    """One knowledge-base record.

    ``attrs`` holds typed, type-specific fields. Bilingual attribute values are
    stored as ``{"el": ..., "en": ...}`` dicts; scalars are stored as-is.
    Documented keys per record type (all optional):

    - product: producer, category, size, availability, sku
    - experience: category, location_name, village, host, duration_hours,
      max_people, availability_note
    - crop: crop_type, scientific_name, planting_period, harvest_period,
      planting_months, harvest_months
    - calendar: month, activity, crop_name, crop_slug
    - glossary: (title = term, body = definition)
    - faq: category (title = question, body = answer)
    - article: category
    - tradition: practice_type
    - module: category, level, duration_min, lesson_count, lessons, objectives
    - lesson: module_slug, position, duration_min, kind
    """

    rid: str
    rtype: str
    slug: str
    title: dict[str, str]
    body: dict[str, str] = field(default_factory=dict)
    summary: dict[str, str] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    status: str = "published"          # published | draft
    is_active: bool = True
    price: float | None = None
    price_visible: bool = False
    lat: float | None = None
    lon: float | None = None
    parent: str | None = None
    url: str = ""
    updated_at: str = ""
    source: str = ""                    # public_site | api | synthetic | demo
    synthetic: bool = False
    exclude_reason: str = ""            # set when a record is excluded (test/duplicate)

    # ------------------------------------------------------------------
    @property
    def is_public(self) -> bool:
        """Editorial gating: only published, active records are public."""
        return self.status == "published" and self.is_active and not self.exclude_reason

    def text(self, part: str, lang: str) -> str:
        return pick(getattr(self, part), lang)

    def attr_text(self, key: str, lang: str) -> str:
        v = self.attrs.get(key)
        if isinstance(v, dict):
            return pick(v, lang)
        if v is None:
            return ""
        return str(v)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Record":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def content_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class Nugget:
    """An atomic, typed fact a correct answer must contain.

    kinds:
      - ``text``: any alias (in any language) must appear (Greek-aware match)
      - ``number``: ``value`` must appear as a number (comma/point decimals)
      - ``duration_hours``: ``value`` hours must appear near an hour/minute unit
      - ``months``: ``value = {"start": m1, "end": m2}`` or ``{"any": [..]}``
      - ``price_on_request``: no currency amount, and an on-request cue
      - ``abstain``: the answer must say the information is not available
      - ``list_member``: like text; item scored by recall over members
      - ``no_leak``: none of ``aliases`` (canary markers) may appear
    """

    nid: str
    kind: str
    aliases: dict[str, list[str]] = field(default_factory=dict)
    value: Any = None
    required: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Nugget":
        return cls(**d)


@dataclass
class Item:
    """A base benchmark item (language-independent meaning)."""

    iid: str
    qtype: str                  # factoid|list|numeric|temporal|definition|price_on_request|unanswerable|procedural
    rtype: str
    template: str
    gold_rids: list[str]
    cluster: str                # unit of resampling: the item's own record (or product family); aggregation
                                # items over several records form their own clusters
    question: dict[str, str]    # canonical question per language (el, en)
    nuggets: list[Nugget]
    component: str = ""         # connected component of items sharing any gold record (sensitivity analysis,
                                # and the unit of the dev/test split so that no record is in both splits)
    answer_ref: dict[str, str] = field(default_factory=dict)
    evidence: dict[str, str] = field(default_factory=dict)
    source_url: str = ""
    split: str = "test"
    status: str = "draft"       # draft | verified (passed bench/verify.py) | needs_fix
    origin: str = "template"    # template | curated | llm
    variants: dict[str, str] = field(default_factory=dict)   # hand-written el_infl / el_colloq
    notes: str = ""
    derive: dict[str, Any] = field(default_factory=dict)      # rule that recomputes an aggregation item's answer
                                                              # from the snapshot (checked by bench/verify.py)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Item":
        d = dict(d)
        d["nuggets"] = [Nugget.from_dict(n) for n in d.get("nuggets", [])]
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class Query:
    """A surface-form realisation of an item, as sent to a system."""

    qid: str
    iid: str
    form: str     # el|en|el_noacc|el_caps|el_sigma|grk_elot|grk_phon|grk_visual|el_infl|el_colloq
    lang: str     # interface language used by the system (el|en)
    text: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Query":
        return cls(**d)


# ----------------------------------------------------------------------------
# JSONL helpers
# ----------------------------------------------------------------------------

def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            if hasattr(r, "to_dict"):
                r = r.to_dict()
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, row: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(row, "to_dict"):
        row = row.to_dict()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_records(path: str | Path) -> list[Record]:
    return [Record.from_dict(d) for d in read_jsonl(path)]


def load_items(path: str | Path) -> list[Item]:
    return [Item.from_dict(d) for d in read_jsonl(path)]


def load_queries(path: str | Path) -> list[Query]:
    return [Query.from_dict(d) for d in read_jsonl(path)]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
