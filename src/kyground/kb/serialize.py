"""Serialisers shared by the tool backend, the vector index and full context.

Three views of a record exist:

- :func:`public_view`  the gated, API-like JSON object returned by the replica
  data API (one language with Greek fallback; hidden prices replaced by an
  on-request notice).
- :func:`record_text`  a bilingual plain-text block (also gated) used as the
  document for vector retrieval and for the full-context condition, so that
  every condition sees exactly the same published content.
- :func:`raw_text`     an *ungated* rendering that includes drafts and hidden
  prices. It is used only by the ``naive_indexer`` misconfiguration ablation.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..schema import Record, pick
from .. import textnorm as tn

ON_REQUEST = {"el": "Τιμή κατόπιν επικοινωνίας", "en": "Price on request"}
TYPE_LABEL = {
    "el": dict(article="Άρθρο", faq="Συχνή ερώτηση", tradition="Παραδοσιακή πρακτική", crop="Καλλιέργεια",
               calendar="Εποχιακό ημερολόγιο", glossary="Γλωσσάρι", product="Προϊόν", experience="Εμπειρία",
               cooperative="Συνεταιρισμός", module="Εκπαιδευτικό πρόγραμμα", lesson="Μάθημα"),
    "en": dict(article="Article", faq="FAQ", tradition="Traditional practice", crop="Crop",
               calendar="Seasonal calendar", glossary="Glossary", product="Product", experience="Experience",
               cooperative="Cooperative", module="Training module", lesson="Lesson"),
}
LBL = {
    "el": dict(producer="Παραγωγός", category="Κατηγορία", price="Τιμή", availability="Διαθεσιμότητα",
               description="Περιγραφή", location="Τοποθεσία", village="Χωριό", duration="Διάρκεια",
               hours="ώρες", max_people="Μέγιστος αριθμός ατόμων", note="Σημείωση διαθεσιμότητας",
               sci="Επιστημονικό όνομα", ctype="Τύπος", planting="Περίοδος φύτευσης", harvest="Περίοδος συγκομιδής",
               month="Μήνας", activity="Δραστηριότητα", crop="Καλλιέργεια", level="Επίπεδο", minutes="λεπτά",
               lessons="Μαθήματα", lesson_count="Αριθμός μαθημάτων", objectives="Μαθησιακοί στόχοι",
               answer="Απάντηση", summary="Περίληψη", definition="Ορισμός", module="Πρόγραμμα", url="Σύνδεσμος",
               ptype="Τύπος πρακτικής"),
    "en": dict(producer="Producer", category="Category", price="Price", availability="Availability",
               description="Description", location="Location", village="Village", duration="Duration",
               hours="hours", max_people="Maximum group size", note="Availability note",
               sci="Scientific name", ctype="Type", planting="Planting period", harvest="Harvest period",
               month="Month", activity="Activity", crop="Crop", level="Level", minutes="minutes",
               lessons="Lessons", lesson_count="Number of lessons", objectives="Learning objectives",
               answer="Answer", summary="Summary", definition="Definition", module="Module", url="Link",
               ptype="Practice type"),
}


def price_display(r: Record, lang: str) -> str:
    if r.price_visible and r.price is not None:
        if lang == "el":
            return f"{r.price:.2f}".replace(".", ",") + " €"
        return f"€{r.price:.2f}"
    return ON_REQUEST[lang]


def _a(r: Record, key: str, lang: str) -> str:
    return r.attr_text(key, lang)


def public_view(r: Record, lang: str = "el", max_body: int = 4000) -> dict[str, Any]:
    """Gated JSON view of one record, in ``lang`` with Greek fallback."""
    t = r.rtype
    v: dict[str, Any] = {"type": t, "id": r.rid, "slug": r.slug, "title": pick(r.title, lang), "url": r.url}
    body = pick(r.body, lang)[:max_body]
    if t == "product":
        v.update(producer=_a(r, "producer", lang), category=_a(r, "category", lang),
                 availability=_a(r, "availability", lang), price=price_display(r, lang), description=body)
    elif t == "experience":
        v.update(category=_a(r, "category", lang), location=_a(r, "location_name", lang),
                 duration_hours=r.attrs.get("duration_hours"), max_people=r.attrs.get("max_people"),
                 availability_note=_a(r, "availability_note", lang), price=price_display(r, lang),
                 description=body)
        if r.lat is not None and r.lon is not None:
            v.update(lat=r.lat, lon=r.lon)
    elif t == "crop":
        v.update(name=pick(r.title, lang), scientific_name=r.attrs.get("scientific_name", ""),
                 crop_type=_a(r, "crop_type", lang), planting_period=_a(r, "planting_period", lang),
                 harvest_period=_a(r, "harvest_period", lang), description=body)
    elif t == "calendar":
        v.update(month=r.attrs.get("month"), activity=_a(r, "activity", lang), crop=_a(r, "crop_name", lang))
    elif t == "glossary":
        v.update(term=pick(r.title, lang), definition=body)
    elif t == "faq":
        v.update(question=pick(r.title, lang), answer=body, category=_a(r, "category", lang))
    elif t in ("article", "tradition"):
        v.update(summary=pick(r.summary, lang), body=body, category=_a(r, "category", lang))
    elif t == "module":
        v.update(category=_a(r, "category", lang), level=_a(r, "level", lang),
                 duration_min=r.attrs.get("duration_min"), lesson_count=r.attrs.get("lesson_count"),
                 description=body, objectives=_a(r, "objectives", lang),
                 lessons=[{"position": les.get("position"), "slug": f"{r.slug}--{int(les.get('position', 0)):02d}",
                           "title": pick(les.get("title", {}), lang), "duration_min": les.get("duration_min")}
                          for les in r.attrs.get("lessons", []) or []])
    elif t == "lesson":
        v.update(module=r.attrs.get("module_slug"), position=r.attrs.get("position"),
                 duration_min=r.attrs.get("duration_min"), content=body)
    elif t == "cooperative":
        v.update(description=body)
    return v


def _lines_for(r: Record, lang: str) -> list[str]:
    L = LBL[lang]
    v = public_view(r, lang)
    t = r.rtype
    out = [f"[{TYPE_LABEL[lang][t]}] {v['title']}"]

    def add(label: str, val: Any) -> None:
        if val not in (None, "", [], {}):
            out.append(f"{label}: {val}")

    if t == "product":
        add(L["producer"], v["producer"]); add(L["category"], v["category"]); add(L["price"], v["price"])
        add(L["availability"], v["availability"]); add(L["description"], v["description"])
    elif t == "experience":
        add(L["category"], v["category"]); add(L["location"], v["location"])
        if v.get("duration_hours") is not None:
            dh = v["duration_hours"]
            add(L["duration"], (f"{dh:g}".replace(".", ",") if lang == "el" else f"{dh:g}") + f" {L['hours']}")
        add(L["max_people"], v.get("max_people")); add(L["note"], v["availability_note"])
        add(L["price"], v["price"]); add(L["description"], v["description"])
    elif t == "crop":
        add(L["sci"], v["scientific_name"]); add(L["ctype"], v["crop_type"])
        add(L["planting"], v["planting_period"]); add(L["harvest"], v["harvest_period"])
        add(L["description"], v["description"])
    elif t == "calendar":
        mname = tn.MONTH_NAME["el" if lang == "el" else "en"].get(v["month"], v["month"])
        add(L["month"], mname); add(L["activity"], v["activity"]); add(L["crop"], v["crop"])
    elif t == "glossary":
        add(L["definition"], v["definition"])
    elif t == "faq":
        add(L["answer"], v["answer"]); add(L["category"], v["category"])
    elif t in ("article", "tradition"):
        add(L["category"], v["category"]); add(L["summary"], v["summary"]); out.append(v["body"])
    elif t == "module":
        add(L["category"], v["category"]); add(L["level"], v["level"])
        if v.get("duration_min"):
            add(L["duration"], f"{v['duration_min']} {L['minutes']}")
        add(L["lesson_count"], v.get("lesson_count")); add(L["description"], v["description"])
        add(L["objectives"], v["objectives"])
        if v["lessons"]:
            out.append(L["lessons"] + ":")
            for les in v["lessons"]:
                d = f" ({les['duration_min']} {L['minutes']})" if les.get("duration_min") else ""
                out.append(f"  {les['position']}. {les['title']}{d}")
    elif t == "lesson":
        add(L["module"], v["module"])
        if v.get("duration_min"):
            add(L["duration"], f"{v['duration_min']} {L['minutes']}")
        out.append(v.get("content") or "")
    else:
        out.append(v.get("description", ""))
    add(L["url"], v["url"])
    return [ln for ln in out if ln and ln.strip()]


def record_text(r: Record, langs: Iterable[str] = ("el", "en")) -> str:
    """Bilingual (gated) text block. English is included only when it differs
    from the Greek fallback, which keeps documents compact."""
    blocks: list[str] = []
    seen: set[str] = set()
    for lang in langs:
        txt = "\n".join(_lines_for(r, lang))
        key = tn.norm(txt)
        if key in seen:
            continue
        seen.add(key)
        blocks.append(txt)
    return "\n\n".join(blocks)


def chunk_record(r: Record, max_chars: int = 1500, langs: Iterable[str] = ("el", "en")) -> list[str]:
    """Split a record into chunks; each chunk repeats the header line."""
    out: list[str] = []
    for lang in langs:
        lines = _lines_for(r, lang)
        if lang != "el" and tn.norm("\n".join(lines)) == tn.norm("\n".join(_lines_for(r, "el"))):
            continue
        header, rest = lines[0], lines[1:]
        cur: list[str] = [header]
        size = len(header)
        for ln in rest:
            while len(ln) > max_chars:  # very long paragraph
                part, ln = ln[:max_chars], ln[max_chars:]
                if len(cur) > 1:
                    out.append("\n".join(cur))
                out.append(header + "\n" + part)
                cur, size = [header], len(header)
            if size + len(ln) + 1 > max_chars and len(cur) > 1:
                out.append("\n".join(cur))
                cur, size = [header], len(header)
            cur.append(ln)
            size += len(ln) + 1
        if len(cur) > 1 or not out:
            out.append("\n".join(cur))
    return out


def kb_text(records: list[Record], langs: Iterable[str] = ("el", "en")) -> str:
    """Whole published KB as one document (full-context condition)."""
    order = ["crop", "calendar", "glossary", "faq", "article", "tradition", "product", "experience",
             "cooperative", "module", "lesson"]
    recs = sorted(records, key=lambda r: (order.index(r.rtype) if r.rtype in order else 99, r.rid))
    return "\n\n---\n\n".join(record_text(r, langs) for r in recs)


def ungated_view(r: Record) -> Record:
    """The record as an indexer that ignores the editorial gating flags would see it: drafts and inactive
    records treated as published and active, hidden prices shown."""
    import copy
    x = copy.deepcopy(r)
    x.status, x.is_active = "published", True
    if x.price is not None:
        x.price_visible = True
    return x


def raw_text(r: Record) -> str:
    """UNGATED rendering (drafts, inactive records, hidden prices).

    Only for the misconfigured-indexer ablation; never shown to the agent."""
    txt = record_text(r)
    extra = [f"status: {r.status}", f"active: {r.is_active}"]
    if r.price is not None:
        extra.append(f"internal price: {r.price:.2f} EUR (visible={r.price_visible})")
    return txt + "\n" + "\n".join(extra)
