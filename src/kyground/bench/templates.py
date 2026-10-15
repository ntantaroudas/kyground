"""Bilingual question templates over structured record fields.

Each template returns :class:`~kyground.schema.Item` objects whose gold
nuggets are computed from the record itself, so the same functions can be
re-evaluated on an *edited* record (freshness experiment) to obtain the
updated gold answer.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable, Iterable

from ..schema import Item, Record, pick
from .. import textnorm as tn
from . import nuggets as N

PLACE_LATIN = {
    "αγια πελαγια": ["Agia Pelagia", "Ayia Pelagia", "Aghia Pelagia"],
    "καραβασ": ["Karavas"],
    "μητατα": ["Mitata"],
    "αυλαιμωνασ": ["Avlemonas", "Avlemona", "Avlaimonas"],
    "παλιοχωρα": ["Paliochora", "Paleochora", "Palaiochora"],
    "παλαιοχωρα": ["Palaiochora", "Paleochora", "Paliochora"],
    "παναγια μυρτιδιωτισσα": ["Panagia Myrtidiotissa", "Myrtidiotissa", "Mirtidiotissa"],
    "ποταμοσ": ["Potamos"],
    "λιβαδι": ["Livadi"],
    "χωρα": ["Chora", "Hora"],
    "καψαλι": ["Kapsali"],
    "μυλοποταμοσ": ["Mylopotamos"],
}
# accusative with article, for "in <village>" phrasing
PLACE_IN_EL = {
    "αγια πελαγια": "στην Αγία Πελαγία", "καραβασ": "στον Καραβά", "μητατα": "στα Μητάτα",
    "αυλαιμωνασ": "στον Αυλαίμωνα", "παλιοχωρα": "στην Παλιόχωρα", "παλαιοχωρα": "στην Παλαιόχωρα",
    "ποταμοσ": "στον Ποταμό", "λιβαδι": "στο Λιβάδι", "χωρα": "στη Χώρα", "καψαλι": "στο Καψάλι",
}
CATEGORY_PHRASE = {  # normalised category -> (Greek object phrase, English phrase)
    "μελι": ("μέλι", "honey"), "αλατι": ("αλάτι", "salt"), "λαδι": ("ελαιόλαδο", "olive oil"),
    "κρασι": ("κρασί", "wine"), "μπυρα": ("μπίρα", "beer"), "παστελια": ("παστέλι", "pasteli"),
}

_SIZE_RE = re.compile(r"\b\d+([.,]\d+)?\s?(ml|gr|g|kg|lt|l)\b", re.IGNORECASE)


def _t(r: Record, lang: str) -> str:
    return pick(r.title, lang)


def place_aliases(name_el: str) -> tuple[list[str], list[str]]:
    key = tn.norm(name_el)
    latin = PLACE_LATIN.get(key, [])
    return [name_el], latin + [tn.to_elot(name_el)]


def product_family(r: Record) -> str:
    base = _SIZE_RE.sub(" ", tn.strip_accents(r.title.get("el", "")))
    return tn.norm(base) + "|" + tn.norm(r.attr_text("producer", "el"))


def _item(r: Record, tid: str, qtype: str, q_el: str, q_en: str, nuggets: list, ref_el: str, ref_en: str,
          evidence_el: str = "", evidence_en: str = "", cluster: str | None = None,
          gold: Iterable[str] | None = None, suffix: str = "") -> Item:
    return Item(
        iid=f"{tid}:{r.slug}{suffix}", qtype=qtype, rtype=r.rtype, template=tid,
        gold_rids=list(gold or [r.rid]), cluster=cluster or r.attrs.get("dup_group") or r.rid,
        question={"el": q_el, "en": q_en}, nuggets=nuggets,
        answer_ref={"el": ref_el, "en": ref_en}, evidence={"el": evidence_el, "en": evidence_en},
        source_url=r.url, origin="template",
    )


# ---------------------------------------------------------------------------
# products
# ---------------------------------------------------------------------------

def t_product_producer(r: Record) -> list[Item]:
    p_el, p_en = r.attr_text("producer", "el"), r.attr_text("producer", "en")
    if not p_el:
        return []
    return [_item(r, "P1-producer", "factoid",
                  f"Ποιος παράγει το προϊόν «{_t(r, 'el')}»;", f"Who produces “{_t(r, 'en')}”?",
                  [N.text("producer", [p_el], [p_en, tn.to_elot(p_el)])],
                  f"Παραγωγός: {p_el}", f"Producer: {p_en or p_el}",
                  f"Από {p_el}", f"By {p_en or p_el}", cluster=f"family:{product_family(r)}")]


def t_product_price(r: Record) -> list[Item]:
    if r.price_visible and r.price is not None:
        nug = N.number("price", r.price)
        ref_el, ref_en = f"Τιμή: {r.price:.2f} €", f"Price: €{r.price:.2f}"
        qtype = "numeric"
    else:
        nug = N.price_on_request()
        ref_el, ref_en = "Τιμή κατόπιν επικοινωνίας με τον παραγωγό", "Price on request (contact the producer)"
        qtype = "price_on_request"
    return [_item(r, "P2-price", qtype, f"Πόσο κοστίζει το προϊόν «{_t(r, 'el')}»;",
                  f"How much does “{_t(r, 'en')}” cost?", [nug], ref_el, ref_en,
                  cluster=f"family:{product_family(r)}")]


def t_producer_products(records: list[Record]) -> list[Item]:
    by_p: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        if r.rtype == "product" and r.attr_text("producer", "el"):
            by_p[r.attr_text("producer", "el")].append(r)
    out: list[Item] = []
    for p, recs in sorted(by_p.items()):
        if len(recs) < 2:
            continue
        fams: dict[str, Record] = {}
        for r in recs:
            fams.setdefault(product_family(r), r)
        members = []
        for i, (fk, r) in enumerate(sorted(fams.items())):
            base_el = _SIZE_RE.sub("", r.title.get("el", "")).strip(" -")
            base_en = _SIZE_RE.sub("", pick(r.title, "en")).strip(" -")
            members.append(N.member(f"m{i}", [base_el, r.title.get("el")], [base_en, pick(r.title, "en")]))
        p_en = recs[0].attr_text("producer", "en") or p
        anchor = recs[0]
        slug = anchor.slug      # not the producer's name: identifiers must not carry personal names
        it = _item(anchor, "P5-producer-products", "list",
                   f"Ποια προϊόντα διαθέτει στην πλατφόρμα ο παραγωγός «{p}»;",
                   f"Which products does the producer “{p_en}” offer on the platform?",
                   members, "Προϊόντα: " + "; ".join(m.aliases["el"][0] for m in members),
                   "Products: " + "; ".join((m.aliases["en"] or m.aliases["el"])[0] for m in members),
                   cluster=f"producer:{tn.norm(p)}", gold=[r.rid for r in recs], suffix=f"--{slug}")
        it.iid = f"P5-producer-products:{slug}"
        out.append(it)
    return out


def t_category_producers(records: list[Record]) -> list[Item]:
    by_c: dict[str, dict[str, list[Record]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.rtype == "product":
            c = tn.norm(r.attr_text("category", "el"))
            if c in CATEGORY_PHRASE and r.attr_text("producer", "el"):
                by_c[c][r.attr_text("producer", "el")].append(r)
    out: list[Item] = []
    for c, prods in sorted(by_c.items()):
        ph_el, ph_en = CATEGORY_PHRASE[c]
        members = []
        gold = []
        for i, (p, recs) in enumerate(sorted(prods.items())):
            members.append(N.member(f"m{i}", [p], [recs[0].attr_text("producer", "en"), tn.to_elot(p)]))
            gold += [r.rid for r in recs]
        anchor = sorted(prods.items())[0][1][0]
        it = _item(anchor, "P6-category-producers", "list",
                   f"Ποιοι παραγωγοί πουλάνε {ph_el} στην πλατφόρμα;",
                   f"Which producers sell {ph_en} on the platform?", members,
                   "Παραγωγοί: " + "; ".join(sorted(prods)), "Producers: " + "; ".join(sorted(prods)),
                   cluster=f"category:{c}", gold=gold)
        it.iid = f"P6-category-producers:{re.sub(r'[^a-z]+', '-', tn.norm(tn.to_elot(ph_el)))}"
        out.append(it)
    return out


# ---------------------------------------------------------------------------
# experiences
# ---------------------------------------------------------------------------

def t_experience_duration(r: Record) -> list[Item]:
    h = r.attrs.get("duration_hours")
    if not h:
        return []
    return [_item(r, "E1-duration", "numeric", f"Πόσες ώρες διαρκεί η εμπειρία «{_t(r, 'el')}»;",
                  f"How long does the “{_t(r, 'en')}” experience last?", [N.duration_hours("duration", h)],
                  f"Διάρκεια: {h:g} ώρες".replace(".", ","), f"Duration: {h:g} hours")]


def t_experience_capacity(r: Record) -> list[Item]:
    n = r.attrs.get("max_people")
    if not n:
        return []
    return [_item(r, "E2-capacity", "numeric", f"Για πόσα άτομα το πολύ είναι η εμπειρία «{_t(r, 'el')}»;",
                  f"What is the maximum group size for the “{_t(r, 'en')}” experience?",
                  [N.number("max_people", n)], f"Έως {n} άτομα", f"Up to {n} people")]


def t_experience_location(r: Record) -> list[Item]:
    loc = r.attr_text("location_name", "el")
    if not loc:
        return []
    el_al, en_al = place_aliases(loc)
    return [_item(r, "E3-location", "factoid", f"Πού πραγματοποιείται η εμπειρία «{_t(r, 'el')}»;",
                  f"Where does the “{_t(r, 'en')}” experience take place?",
                  [N.text("location", el_al, en_al)], f"Τοποθεσία: {loc}", f"Location: {en_al[0] if en_al else loc}")]


def t_experience_price(r: Record) -> list[Item]:
    if r.price_visible and r.price is not None:
        nug, qtype = N.number("price", r.price), "numeric"
        ref_el, ref_en = f"Τιμή: {r.price:.2f} €", f"Price: €{r.price:.2f}"
    else:
        nug, qtype = N.price_on_request(), "price_on_request"
        ref_el, ref_en = "Τιμή κατόπιν επικοινωνίας με τον οικοδεσπότη", "Price on request (contact the host)"
    return [_item(r, "E4-price", qtype, f"Πόσο κοστίζει η εμπειρία «{_t(r, 'el')}»;",
                  f"How much does the “{_t(r, 'en')}” experience cost?", [nug], ref_el, ref_en)]


def t_experience_availability(r: Record) -> list[Item]:
    note = r.attr_text("availability_note", "el")
    if not note:
        return []
    ms = tn.months_in(note)
    if ms:
        nug = N.months("availability", ms[0], ms[-1])
    elif "ανοιξ" in tn.norm(note):
        nug = N.text("availability", ["άνοιξη", "ανοιξιάτικους", "Μάρτιο, Απρίλιο, Μάιο"], ["spring"])
    else:
        return []
    return [_item(r, "E5-availability", "temporal", f"Πότε είναι διαθέσιμη η εμπειρία «{_t(r, 'el')}»;",
                  f"When is the “{_t(r, 'en')}” experience available?", [nug], note,
                  r.attr_text("availability_note", "en") or note, note, "")]


def t_experiences_by_place(records: list[Record]) -> list[Item]:
    by_v: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        if r.rtype == "experience" and r.attr_text("location_name", "el"):
            by_v[tn.norm(r.attr_text("location_name", "el"))].append(r)
    out = []
    for v, recs in sorted(by_v.items()):
        if len(recs) < 2 or v not in PLACE_IN_EL:
            continue
        name = recs[0].attr_text("location_name", "el")
        en_name = (PLACE_LATIN.get(v) or [tn.to_elot(name)])[0]
        members = [N.member(f"m{i}", [_t(r, "el")], [_t(r, "en")]) for i, r in enumerate(recs)]
        it = _item(recs[0], "E6-experiences-at-place", "list",
                   f"Ποιες αγροτουριστικές εμπειρίες υπάρχουν {PLACE_IN_EL[v]};",
                   f"Which agritourism experiences are offered in {en_name}?", members,
                   "Εμπειρίες: " + "; ".join(_t(r, "el") for r in recs),
                   "Experiences: " + "; ".join(_t(r, "en") for r in recs),
                   cluster=f"place:{v}", gold=[r.rid for r in recs])
        it.iid = f"E6-experiences-at-place:{re.sub(r'[^a-z]+', '-', tn.norm(en_name))}"
        out.append(it)
    return out


# ---------------------------------------------------------------------------
# crops and calendar
# ---------------------------------------------------------------------------

CROP_INFL = {  # natural phrasing with the crop term in another form (el_infl)
    "ελια": "Πότε μαζεύονται οι ελιές στα Κύθηρα;",
    "ελια κορωνεικη": "Πότε μαζεύονται οι κορωνέικες ελιές στα Κύθηρα;",
    "αμπελι": "Πότε τρυγάμε τα αμπέλια στα Κύθηρα;",
    "θυμαρι": "Πότε μαζεύουμε τα θυμάρια στα Κύθηρα;",
    "ριγανη": "Πότε μαζεύουμε τις ρίγανες στα Κύθηρα;",
}


def _period_nugget(nid: str, el: str, en: str):
    rng = tn.parse_month_range(el) or tn.parse_month_range(en)
    if rng:
        return N.months(nid, *rng)
    if el:
        return N.text(nid, [el], [en] if en else [])
    return None


def t_crop_periods(r: Record) -> list[Item]:
    out = []
    for key, tid, q_el, q_en in (
        ("harvest_period", "C1-harvest",
         "Ποια είναι η περίοδος συγκομιδής για την καλλιέργεια «{t}» στα Κύθηρα;",
         "When is the harvest period for “{t}” on Kythera?"),
        ("planting_period", "C2-planting",
         "Πότε φυτεύεται η καλλιέργεια «{t}» στα Κύθηρα;",
         "When is “{t}” planted on Kythera?"),
    ):
        el, en = r.attr_text(key, "el"), r.attr_text(key, "en")
        nug = _period_nugget(key, el, en)
        if nug is None:
            continue
        it = _item(r, tid, "temporal", q_el.format(t=_t(r, "el")), q_en.format(t=_t(r, "en")), [nug],
                   el, en or el, el, en)
        if tid == "C1-harvest" and tn.norm(_t(r, "el")) in CROP_INFL:
            it.variants["el_infl"] = CROP_INFL[tn.norm(_t(r, "el"))]
        out.append(it)
    sci = r.attrs.get("scientific_name")
    if sci:
        out.append(_item(r, "C3-scientific-name", "factoid",
                         f"Ποιο είναι το επιστημονικό όνομα της καλλιέργειας «{_t(r, 'el')}»;",
                         f"What is the scientific name given for “{_t(r, 'en')}”?",
                         [N.text("sci", [sci], [sci])], sci, sci))
    return out


def t_calendar_month(records: list[Record]) -> list[Item]:
    by_m: dict[int, list[Record]] = defaultdict(list)
    for r in records:
        if r.rtype == "calendar":
            by_m[int(r.attrs.get("month", 0))].append(r)
    out = []
    for m, recs in sorted(by_m.items()):
        members, seen = [], set()
        for r in recs:      # one member per distinct task (May lists flowering for two crops)
            key = tn.norm(r.attr_text("activity", "el"))
            if key in seen:
                continue
            seen.add(key)
            members.append(N.member(f"m{len(members)}", [r.attr_text("activity", "el")],
                                    [r.attr_text("activity", "en")]))
        it = _item(recs[0], "K1-month-tasks", "list",
                   f"Ποιες αγροτικές εργασίες προβλέπει το εποχιακό ημερολόγιο των Κυθήρων για τον "
                   f"{tn.MONTH_NAME['el_acc'][m]};",
                   f"Which farm tasks does Kythera's seasonal calendar list for {tn.MONTH_NAME['en'][m]}?",
                   members,
                   "; ".join(f"{r.attr_text('activity', 'el')} ({r.attr_text('crop_name', 'el')})" for r in recs),
                   "; ".join(f"{r.attr_text('activity', 'en')} ({r.attr_text('crop_name', 'en')})" for r in recs),
                   cluster=f"calendar-month:{m:02d}", gold=[r.rid for r in recs])
        it.iid = f"K1-month-tasks:m{m:02d}"
        out.append(it)
    return out


def t_calendar_activity(r: Record) -> list[Item]:
    m = int(r.attrs.get("month", 0))
    act_el, act_en = r.attr_text("activity", "el"), r.attr_text("activity", "en")
    crop_el, crop_en = r.attr_text("crop_name", "el"), r.attr_text("crop_name", "en")
    if not (m and act_el):
        return []
    return [_item(r, "K2-activity-month", "temporal",
                  f"Σε ποιον μήνα γίνεται η εργασία «{act_el}» για την καλλιέργεια «{crop_el}», "
                  f"σύμφωνα με το εποχιακό ημερολόγιο;",
                  f"In which month does the seasonal calendar schedule “{act_en or act_el}” for "
                  f"“{crop_en or crop_el}”?",
                  [N.months("month", m)], tn.MONTH_NAME["el"][m], tn.MONTH_NAME["en"][m])]


# ---------------------------------------------------------------------------
# training modules
# ---------------------------------------------------------------------------

def t_module(r: Record) -> list[Item]:
    out = []
    a = r.attrs
    if a.get("duration_min"):
        mins = int(a["duration_min"])
        out.append(_item(r, "M1-duration", "numeric", f"Πόση διάρκεια έχει το εκπαιδευτικό πρόγραμμα «{_t(r, 'el')}»;",
                         f"How long is the training programme “{_t(r, 'en')}”?",
                         [N.duration_hours("duration", mins / 60.0)], f"{mins} λεπτά", f"{mins} minutes"))
    if a.get("lesson_count"):
        n = int(a["lesson_count"])
        out.append(_item(r, "M2-lesson-count", "numeric", f"Πόσα μαθήματα περιλαμβάνει το πρόγραμμα «{_t(r, 'el')}»;",
                         f"How many lessons does the training programme “{_t(r, 'en')}” include?",
                         [N.number("lessons", n)], f"{n} μαθήματα", f"{n} lessons"))
    lvl_el, lvl_en = r.attr_text("level", "el"), r.attr_text("level", "en")
    if lvl_el:
        out.append(_item(r, "M3-level", "factoid", f"Σε ποιο επίπεδο δυσκολίας απευθύνεται το πρόγραμμα «{_t(r, 'el')}»;",
                         f"What difficulty level is the training programme “{_t(r, 'en')}” aimed at?",
                         [N.text("level", [lvl_el, "αρχάριους"], [lvl_en, "beginners"])], lvl_el, lvl_en or lvl_el))
    lessons = a.get("lessons") or []
    if lessons:
        members = [N.member(f"m{i}", [pick(les["title"], "el")], [pick(les["title"], "en")])
                   for i, les in enumerate(lessons)]
        out.append(_item(r, "M4-lessons", "list", f"Ποια μαθήματα περιλαμβάνει το πρόγραμμα «{_t(r, 'el')}»;",
                         f"Which lessons does the training programme “{_t(r, 'en')}” include?", members,
                         "; ".join(pick(les["title"], "el") for les in lessons),
                         "; ".join(pick(les["title"], "en") for les in lessons)))
    return out


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

PER_RECORD: dict[str, list[Callable[[Record], list[Item]]]] = {
    "product": [t_product_producer, t_product_price],
    "experience": [t_experience_duration, t_experience_capacity, t_experience_location, t_experience_price,
                   t_experience_availability],
    "crop": [t_crop_periods],
    "calendar": [t_calendar_activity],
    "module": [t_module],
}
COLLECTIONS: list[Callable[[list[Record]], list[Item]]] = [
    t_producer_products, t_category_producers, t_experiences_by_place, t_calendar_month,
]


def _nugget_signature(it: Item) -> str:
    return repr(sorted((n.kind, repr(n.value), tuple(sorted(tn.norm(a) for a in n.aliases.get("el", []))))
                       for n in it.nuggets))


DROPPED: list[dict] = []   # items dropped because near-duplicate records disagree (reported by the builder)


def template_items(records: list[Record]) -> list[Item]:
    """All template items for the public records; one item per product family for per-product templates; one
    per duplicate group for others, and none when the duplicates disagree on the answer (the editors must
    resolve the conflict first)."""
    pub = [r for r in records if r.is_public]
    groups: dict[str, list[Record]] = defaultdict(list)
    for r in pub:
        if r.attrs.get("dup_group"):
            groups[r.attrs["dup_group"]].append(r)
    DROPPED.clear()
    items: list[Item] = []
    seen_family: set[tuple[str, str]] = set()
    seen_dup: set[tuple[str, str]] = set()
    for r in pub:
        for fn in PER_RECORD.get(r.rtype, []):
            dg = r.attrs.get("dup_group")
            if dg and len(groups[dg]) > 1:
                if (fn.__name__, dg) in seen_dup:
                    continue
                seen_dup.add((fn.__name__, dg))
                by_tid: dict[str, list[Item]] = defaultdict(list)
                for m in groups[dg]:
                    for it in fn(m):
                        by_tid[it.template].append(it)
                for tid, its in by_tid.items():
                    if len({_nugget_signature(i) for i in its}) > 1:
                        DROPPED.append({"template": tid, "dup_group": dg, "records": [m.rid for m in groups[dg]],
                                        "reason": "near-duplicate records give different answers"})
                        continue
                    it = its[0]
                    it.gold_rids = sorted(m.rid for m in groups[dg])
                    items.append(it)
                continue
            for it in fn(r):
                if r.rtype == "product":
                    key = (it.template, product_family(r))
                    if key in seen_family:
                        continue
                    seen_family.add(key)
                items.append(it)
    for fn in COLLECTIONS:
        items.extend(fn(pub))
    return items


# Probe builders reused by the freshness experiment ------------------------

FIELD_PROBES: dict[str, Callable[[Record], list[Item]]] = {
    "duration_hours": t_experience_duration,
    "max_people": t_experience_capacity,
    "availability_note": t_experience_availability,
    "harvest_period": lambda r: [i for i in t_crop_periods(r) if i.template == "C1-harvest"],
    "planting_period": lambda r: [i for i in t_crop_periods(r) if i.template == "C2-planting"],
    "duration_min": lambda r: [i for i in t_module(r) if i.template == "M1-duration"],
    "lesson_count": lambda r: [i for i in t_module(r) if i.template == "M2-lesson-count"],
    "producer": t_product_producer,
    "price": t_product_price,
    "month": t_calendar_activity,
}
