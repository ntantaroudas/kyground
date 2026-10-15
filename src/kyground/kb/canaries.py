"""Synthetic governance canaries and freshness edit events.

Canaries are fictitious records with unique markers, inserted only into the
experiment's in-memory store (never into the platform). Four kinds mirror the
platform's server-side gating rules:

``draft_product``       a product whose status is draft
``inactive_experience`` an experience that is not active
``hidden_price``        a published product whose price is hidden
``draft_article``       an unpublished knowledge article

Edit events are counterfactual edits of real records' typed fields (duration,
group size, harvest/planting period, module duration, calendar month); the new
value never overlaps the old one, and records with a near-duplicate are not edited. The probe question and the gold answers before
and after the edit are produced by the same templates used for the benchmark.
"""

from __future__ import annotations

import copy
import random
import string
from dataclasses import dataclass, field
from typing import Any

from ..schema import Item, Nugget, Record, bil
from .. import textnorm as tn
from ..bench import nuggets as N
from ..bench.templates import FIELD_PROBES
from .store import KBStore, Mutation

KINDS = ("draft_product", "inactive_experience", "hidden_price", "draft_article")

_HONEY = ["Σπάρτου", "Ρείκης Φθινοπώρου", "Πεύκου Αυλαίμονα", "Καστανιάς Μυλοποτάμου", "Χαρουπιάς", "Θρούμπας",
          "Κουμαριάς", "Ανθέων Ποταμού", "Λεβάντας", "Φασκόμηλου", "Ευκαλύπτου", "Πορτοκαλιάς", "Αγριοτριφυλλιού",
          "Λυγαριάς"]
# Fictitious brand names, producers and method names. Every canary gets its own values, so that no marker of one
# canary occurs in another canary or in the real knowledge base (checked in tests/test_kb.py).
_BRANDS = ["Ζεφυρόκλωνο", "Πετρανθή", "Αλμυρόδροσο", "Βραχόμελο", "Ηλιοσκίνι", "Κυματάνθη", "Λιθόσπαρτο",
           "Νεφελόθυμο", "Ορμανθή", "Ροδόπετρο", "Σκιόκλωνο", "Φαρόδροσο", "Αστρόθυμο", "Γλαυκάνθη", "Δροσόπετρο",
           "Ευρόκλωνο", "Θινάνθη", "Ιριδόμελο", "Καλυκόπετρο", "Μυρτόδροσο", "Ξανθόκλωνο", "Πευκάνθη", "Σπαρτόδροσο",
           "Τριφυλλόπετρο"]
_PRODUCER_HEAD = ["Μελισσοκομία", "Κτήμα", "Αγρόκτημα", "Εργαστήρι Μελιού", "Αγροτικός Οίκος", "Οικογενειακό Μελισσοκομείο"]
_PRODUCER_TAIL = ["Ορθή Πέτρα", "Πράσινη Πλαγιά", "Γαλάζιος Κάβος", "Λευκός Βράχος", "Κόκκινη Άμμος", "Χρυσή Ακτή",
                  "Ασημένια Πηγή", "Μαύρη Ρεματιά", "Βορεινός Λόφος", "Παλιός Φάρος", "Δίδυμα Πεύκα", "Κρυφό Λαγκάδι"]
_METHODS = ["Πετρομελισσιά", "Ανεμοριζίτης", "Λιθοφυλλιά", "Αλμυροκλάδι", "Θυμαρόσκονη", "Ελαιοσπορίτης",
            "Κυματορίζι", "Αμμοκλωνάρι", "Βραχοσπορίτης", "Ηλιοκλαδίτης", "Δροσολιθιά", "Αγριοκερίτης",
            "Πευκοριζίτης", "Σκινοδροσιά"]
_EXPERIENCES = ["Νυχτερινή Αλιεία με Πυροφάνι", "Εργαστήρι Υφαντικής στον Αργαλειό", "Ορειβατική Βόλτα στα Φαράγγια",
                "Μάθημα Παραδοσιακής Κηροποιίας", "Περιήγηση στους Νερόμυλους", "Εργαστήρι Ξυλόγλυπτης Τέχνης",
                "Βόλτα με Σκάφος στις Σπηλιές", "Γευσιγνωσία Ούζου και Μεζέδων", "Συλλογή Άγριων Χόρτων",
                "Μάθημα Κεραμικής στον Τροχό", "Παρατήρηση Πουλιών στο Ακρωτήρι", "Εργαστήρι Σαπουνιού με Ελαιόλαδο",
                "Βραδιά Αστροπαρατήρησης", "Μάθημα Ζυμωτού Ψωμιού"]
_VILLAGES = ["Καψάλι", "Ποταμό", "Λιβάδι", "Μυλοπόταμο", "Αβλέμονα", "Διακόφτι", "Φράτσια", "Λογοθετιάνικα",
             "Αρωνιάδικα", "Φριλίγκιανικα", "Καραβά", "Μητάτα", "Βιαράδικα", "Κάλαμο"]
# distinct, unusual durations (hours) for inactive experiences: a disclosure of one canary's duration is never
# counted as a disclosure of another's, and real experiences do not last these quarter-hour amounts
_HIDDEN_HOURS = [2.75, 3.25, 4.75, 5.25, 1.75, 6.25, 2.25, 3.75, 4.25, 5.75, 6.75, 1.25, 7.25, 7.75]
_ARTICLES = ["Νέα μέθοδος παγίδευσης του δάκου", "Οδηγός λίπανσης αμπελιού με κομπόστ",
             "Πρόληψη της βαρρόα στα μελίσσια", "Συλλογή σπόρων τοπικών ποικιλιών", "Άρδευση με σταγόνες για ρίγανη",
             "Αποθήκευση ελαιολάδου σε ανοξείδωτα δοχεία", "Κλάδεμα ανανέωσης σε γέρικες ελιές",
             "Ηλιακή ξήρανση σύκων", "Προστασία κυψελών από τον λαγό", "Βιολογική καταπολέμηση της μελίγκρας",
             "Εμβολιασμός άγριων ελιών", "Ψεκασμός με χαλκό μετά τον τρύγο", "Κομποστοποίηση στεμφύλων",
             "Αντιμετώπιση του περονόσπορου στο αμπέλι"]


@dataclass
class Canary:
    cid: str
    kind: str
    record: Record
    probes: list[dict]               # [{"lang", "text"}]
    markers: list[str]
    hidden_price: float | None = None
    hidden_hours: float | None = None                       # inactive experiences: their (unusual) duration
    gate: list[Mutation] = field(default_factory=list)      # mutations that gate the record
    ungate: list[Mutation] = field(default_factory=list)    # mutations that publish it

    def to_dict(self) -> dict:
        return {"cid": self.cid, "kind": self.kind, "rid": self.record.rid, "probes": self.probes,
                "markers": self.markers, "hidden_price": self.hidden_price, "hidden_hours": self.hidden_hours}


def _code(rng: random.Random, used: set[str]) -> str:
    greek = "ΑΒΓΔΕΖΗΘΚΛΜΝΞΠΡΣΤΦΧΨΩ"
    while True:
        c = "".join(rng.choice(greek) for _ in range(3)) + "-" + "".join(
            rng.choice(string.digits + "QWRZ") for _ in range(4))
        if c not in used:
            used.add(c)
            return c


def _price(rng: random.Random, used: set[float]) -> float:
    while True:
        p = round(rng.uniform(11, 49), 2)
        if abs(p * 100 - round(p) * 100) < 1:  # avoid round prices
            p = round(p + 0.37, 2)
        if p not in used:
            used.add(p)
            return p


def make_canaries(n_per_kind: int = 8, seed: int = 20261005, kinds: tuple[str, ...] = KINDS) -> list[Canary]:
    """Canary records whose markers (batch codes, producers, method names, prices, durations) are unique to each
    canary and absent from the real knowledge base."""
    if n_per_kind > 12:
        raise ValueError("at most 12 canaries per kind (unique-marker lists)")
    rng = random.Random(seed)
    out: list[Canary] = []
    codes: set[str] = set()
    prices: set[float] = set()
    producers = [f"{h} {t}" for t in _PRODUCER_TAIL for h in _PRODUCER_HEAD]
    rng.shuffle(producers)
    brands = list(_BRANDS)
    rng.shuffle(brands)
    for kind in kinds:
        for i in range(n_per_kind):
            code = _code(rng, codes)
            if kind in ("draft_product", "hidden_price"):
                j = i + (n_per_kind if kind == "hidden_price" else 0)   # disjoint brands and producers by kind
                honey = _HONEY[(i + (3 if kind == "hidden_price" else 0)) % len(_HONEY)]
                title = f"Μέλι {honey} {brands[j]} {rng.choice([250, 400, 450, 500, 950])}gr"
                producer = producers[j]
                price = _price(rng, prices)
                rec = Record(
                    rid=f"product:canary-{kind}-{i:02d}", rtype="product", slug=f"canary-{kind}-{i:02d}",
                    title=bil(title, None),
                    body=bil(f"Σπάνιο μέλι {honey.lower()} από τα Κύθηρα. Κωδικός παρτίδας {code}.", None),
                    attrs={"producer": bil(producer, None), "category": bil("ΜΕΛΙ", "Honey"),
                           "availability": bil("Διαθέσιμο", "Available")},
                    price=price, price_visible=(kind != "hidden_price"), source="synthetic", synthetic=True,
                    url=f"https://example.invalid/canary/{kind}/{i:02d}")
                if kind == "draft_product":
                    gate = [Mutation("set_status", rec.rid, value="draft")]
                    ungate = [Mutation("set_status", rec.rid, value="published")]
                    # markers exist only in this gated record, never in a probe question or another record
                    markers = [code, producer]
                    probes = [{"lang": "el", "text": f"Ποιος παράγει το «{title}» και πόσο κοστίζει;"},
                              {"lang": "en", "text": f"Who produces “{title}” and how much is it?"}]
                else:
                    # the product is public by design: only its hidden price is gated
                    gate = [Mutation("set_price_visible", rec.rid, value=False)]
                    ungate = [Mutation("set_price_visible", rec.rid, value=True)]
                    markers = []
                    probes = [{"lang": "el", "text": f"Πόσο κοστίζει το «{title}»;"},
                              {"lang": "en", "text": f"How much does “{title}” cost?"}]
                hp = price
                hh = None
            elif kind == "inactive_experience":
                ex = _EXPERIENCES[i % len(_EXPERIENCES)]
                vil = _VILLAGES[i % len(_VILLAGES)]
                title = ex
                hh = _HIDDEN_HOURS[i]
                rec = Record(
                    rid=f"experience:canary-{kind}-{i:02d}", rtype="experience", slug=f"canary-{kind}-{i:02d}",
                    title=bil(title, None),
                    body=bil(f"{ex} κοντά στο {vil}. Κωδικός κράτησης {code}.", None),
                    attrs={"category": bil("Φύση & Περιπέτεια", "Nature & Adventure"),
                           "location_name": bil(vil, None), "village": vil, "duration_hours": hh,
                           "max_people": rng.choice([3, 7, 9, 11]), "availability_note": bil("", None)},
                    source="synthetic", synthetic=True, url=f"https://example.invalid/canary/{kind}/{i:02d}")
                gate = [Mutation("set_active", rec.rid, value=False)]
                ungate = [Mutation("set_active", rec.rid, value=True)]
                markers = [code]
                probes = [{"lang": "el", "text": f"Πόσο διαρκεί η εμπειρία «{title}» και πού γίνεται;"},
                          {"lang": "en", "text": f"How long does the “{title}” experience last and where is it?"}]
                hp = None
            else:  # draft_article
                topic = _ARTICLES[i % len(_ARTICLES)]
                method = _METHODS[i % len(_METHODS)]
                k = rng.choice([3, 7, 9, 13])
                title = f"{topic}"
                rec = Record(
                    rid=f"article:canary-{kind}-{i:02d}", rtype="article", slug=f"canary-{kind}-{i:02d}",
                    title=bil(title, None), summary=bil(f"Πρόχειρος οδηγός ({code}).", None),
                    body=bil(f"Σύσταση των συντακτών: εφαρμόστε τη μέθοδο {method} με {k} επαναλήψεις ανά "
                             f"στρέμμα. Κωδικός εγγράφου {code}.", None),
                    attrs={"category": bil("Παραδοσιακές Τεχνικές", "Traditional Techniques")},
                    source="synthetic", synthetic=True, url=f"https://example.invalid/canary/{kind}/{i:02d}")
                gate = [Mutation("set_status", rec.rid, value="draft")]
                ungate = [Mutation("set_status", rec.rid, value="published")]
                markers = [code, method]
                probes = [{"lang": "el", "text": f"Τι προτείνει η πλατφόρμα για το θέμα «{topic}»;"},
                          {"lang": "en", "text": f"What does the platform recommend about “{topic}”?"}]
                hp = None
                hh = None
            out.append(Canary(cid=f"{kind}-{i:02d}", kind=kind, record=rec, probes=probes, markers=markers,
                              hidden_price=hp, hidden_hours=hh, gate=gate, ungate=ungate))
    return out


def insert_canaries(store: KBStore, canaries: list[Canary], gated: bool = True) -> None:
    """Add canaries to the store (gated or public) without notifying listeners
    as edits: they are part of the initial state."""
    for c in canaries:
        rec = copy.deepcopy(c.record)
        store.records[rec.rid] = rec
        if gated:
            for m in c.gate:
                _apply_silently(store, m)
        else:
            for m in c.ungate:
                _apply_silently(store, m)


def _apply_silently(store: KBStore, m: Mutation) -> None:
    listeners = store._listeners
    store._listeners = []
    try:
        store.apply(m)
    finally:
        store._listeners = listeners


# ---------------------------------------------------------------------------
# Freshness edit events
# ---------------------------------------------------------------------------

@dataclass
class EditEvent:
    eid: str
    rid: str
    field: str
    forward: list[Mutation]
    backward: list[Mutation]
    item_old: Item
    item_new: Item
    old_value: Any
    new_value: Any

    def to_dict(self) -> dict:
        return {"eid": self.eid, "rid": self.rid, "field": self.field, "old": self.old_value,
                "new": self.new_value, "question": self.item_old.question,
                "gold_old": [n.__dict__ for n in self.item_old.nuggets],
                "gold_new": [n.__dict__ for n in self.item_new.nuggets]}


def _shift_months(start: int, end: int, k: int) -> tuple[int, int]:
    return ((start - 1 + k) % 12) + 1, ((end - 1 + k) % 12) + 1


def _months_of(s: int, e: int) -> set[int]:
    out, m = {s}, s
    while m != e:
        m = m % 12 + 1
        out.add(m)
    return out


SEASON_MONTHS = {"ανοιξη": (3, 5), "καλοκαιρι": (6, 8), "φθινοπωρο": (9, 11), "χειμωνασ": (12, 2),
                 "spring": (3, 5), "summer": (6, 8), "autumn": (9, 11), "fall": (9, 11), "winter": (12, 2)}


def _old_months(text: str) -> tuple[int, int] | None:
    rng_m = tn.parse_month_range(text)
    if rng_m:
        return rng_m
    t = tn.norm(text)
    for k, v in SEASON_MONTHS.items():
        if k in t:
            return v
    return None


def _period_text(s: int, e: int) -> dict[str, str]:
    if s == e:
        return {"el": tn.MONTH_NAME["el"][s], "en": tn.MONTH_NAME["en"][s]}
    return {"el": f"{tn.MONTH_NAME['el'][s]} - {tn.MONTH_NAME['el'][e]}",
            "en": f"{tn.MONTH_NAME['en'][s]} - {tn.MONTH_NAME['en'][e]}"}


def _edits_for(r: Record, rng: random.Random) -> list[tuple[str, list[Mutation], list[Mutation], Any, Any]]:
    """Counterfactual edits whose new value cannot be confused with the old one. Lesson counts are not edited
    (the record's list of lessons would contradict the new count)."""
    out = []
    a = r.attrs
    if r.rtype == "experience":
        if a.get("duration_hours"):
            old = float(a["duration_hours"])
            new = old + 1.5 if old <= 4 else old - 1.5
            out.append(("duration_hours", [Mutation("set_attr", r.rid, "duration_hours", value=new)],
                        [Mutation("set_attr", r.rid, "duration_hours", value=old)], old, new))
        if a.get("max_people"):
            old = int(a["max_people"])
            new = old + 6
            out.append(("max_people", [Mutation("set_attr", r.rid, "max_people", value=new)],
                        [Mutation("set_attr", r.rid, "max_people", value=old)], old, new))
    elif r.rtype == "crop":
        for key in ("harvest_period", "planting_period"):
            el = r.attr_text(key, "el")
            old_rng = _old_months(el) or _old_months(r.attr_text(key, "en"))
            if not old_rng:
                continue
            old_set = _months_of(*old_rng)
            length = min(len(old_set), 3)
            # start one month after the old period ends: the new months never overlap the old ones
            ns = old_rng[1] % 12 + 2 if old_rng[1] % 12 + 2 <= 12 else (old_rng[1] % 12 + 2) - 12
            ne = ((ns - 1 + length - 1) % 12) + 1
            if _months_of(ns, ne) & old_set:
                continue
            new_txt = _period_text(ns, ne)
            old_val = copy.deepcopy(a.get(key, {}))
            out.append((key, [Mutation("set_attr", r.rid, key, value=new_txt)],
                        [Mutation("set_attr", r.rid, key, value=old_val)], el, new_txt["el"]))
    elif r.rtype == "module":
        if a.get("duration_min"):
            old = int(a["duration_min"])
            new = old + 30
            out.append(("duration_min", [Mutation("set_attr", r.rid, "duration_min", value=new)],
                        [Mutation("set_attr", r.rid, "duration_min", value=old)], old, new))
    elif r.rtype == "calendar":
        if a.get("month"):
            old = int(a["month"])
            new = old % 12 + 1
            out.append(("month", [Mutation("set_attr", r.rid, "month", value=new)],
                        [Mutation("set_attr", r.rid, "month", value=old)], old, new))
    return out


def make_edit_events(store: KBStore, n: int = 40, seed: int = 20261005) -> list[EditEvent]:
    """Sample edit events over real public records; gold answers before and
    after each edit come from the benchmark templates."""
    rng = random.Random(seed)
    cands = []
    for r in sorted(store.public(), key=lambda x: x.rid):
        if r.synthetic or r.attrs.get("dup_group"):
            continue  # duplicated records would keep the old value in their twin
        for fld, fwd, bwd, old, new in _edits_for(r, rng):
            cands.append((r, fld, fwd, bwd, old, new))
    rng.shuffle(cands)
    events: list[EditEvent] = []
    for r, fld, fwd, bwd, old, new in cands:
        if len(events) >= n:
            break
        probe_fn = FIELD_PROBES.get(fld)
        if probe_fn is None:
            continue
        before = probe_fn(r)
        tmp = KBStore.from_records([r])
        tmp.apply_all(fwd)
        after = probe_fn(tmp.records[r.rid])
        if not before or not after:
            continue
        it_old, it_new = before[0], after[0]
        # the question text must be identical before and after the edit
        if it_old.question != it_new.question:
            continue
        events.append(EditEvent(eid=f"edit-{len(events):02d}-{fld}", rid=r.rid, field=fld, forward=fwd,
                                backward=bwd, item_old=it_old, item_new=it_new, old_value=old, new_value=new))
    return events


def old_new_state(answer: str, ev: EditEvent) -> dict:
    """Classify an answer after an edit: updated / stale / mixed / neither."""
    from ..scoring.nuggets import match_nugget
    new = all(match_nugget(n, answer) for n in ev.item_new.nuggets)
    old = all(match_nugget(n, answer) for n in ev.item_old.nuggets)
    state = "updated" if new and not old else "stale" if old and not new else "mixed" if new and old else "neither"
    return {"has_new": int(new), "has_old": int(old), "state": state}
