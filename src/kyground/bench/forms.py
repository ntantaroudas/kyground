"""Surface forms of a question (robustness experiment).

=============  ===========================================================
``el``         canonical Greek (accented, natural)
``en``         English
``el_noacc``   Greek without accents (very common when typing on phones)
``el_caps``    Greek in capitals without accents (common informal style)
``grk_elot``   Greeklish, ELOT 743 / ISO 843-style transcription
``grk_phon``   Greeklish, informal phonetic (x for χ, i for η/ι/υ/ει/οι)
``grk_visual`` Greeklish, informal visual (8 for θ, w for ω, h for η, 3 for ξ)
``el_infl``    key term in another grammatical form (hand-written)
``el_colloq``  colloquial / regional (Kytherian) wording (hand-written or
               produced by the lexicon below, outside quoted titles)
=============  ===========================================================
"""

from __future__ import annotations

import re

from ..schema import Item, Query
from .. import textnorm as tn

FORMS = ("el", "en", "el_noacc", "el_caps", "grk_elot", "grk_phon", "grk_visual", "el_infl", "el_colloq")

# Colloquial / regional substitutions (applied outside «...» titles), longest
# first. Pairs are phrase-level so that articles and agreement stay correct.
COLLOQUIAL = [
    ("στα Κύθηρα", "στο Τσιρίγο"),
    ("των Κυθήρων", "του Τσιρίγου"),
    ("τα Κύθηρα", "το Τσιρίγο"),
    ("στο ελαιοτριβείο", "στο λιοτρίβι"),
    ("το ελαιοτριβείο", "το λιοτρίβι"),
    ("ελαιοτριβείο", "λιοτρίβι"),
    ("τα ελαιόδεντρα", "τα λιόδεντρα"),
    ("των ελαιοδέντρων", "των λιόδεντρων"),
    ("τη συγκομιδή της ελιάς", "το λιομάζωμα"),
    ("συγκομιδή ελιάς", "λιομάζωμα"),
    ("Πόσο κοστίζει", "Πόσο κάνει"),
    ("πραγματοποιείται", "γίνεται"),
    ("Πόσες ώρες διαρκεί", "Πόσες ώρες κρατάει"),
    ("Ιανουάριο", "Γενάρη"),
    ("Φεβρουάριο", "Φλεβάρη"),
    ("Μάιο", "Μάη"),
    ("Σεπτέμβριο", "Σεπτέμβρη"),
    ("Οκτώβριο", "Οκτώβρη"),
    ("Νοέμβριο", "Νοέμβρη"),
    ("Δεκέμβριο", "Δεκέμβρη"),
    ("ελαιόλαδο", "λάδι"),
]

_QUOTED = re.compile(r"(«[^»]*»)")


def colloquial(text: str) -> str | None:
    """Apply the colloquial lexicon outside quoted titles; None if unchanged."""
    parts = _QUOTED.split(text)
    changed = False
    for i, part in enumerate(parts):
        if part.startswith("«"):
            continue
        for a, b in COLLOQUIAL:
            new = re.sub(r"(?<!\w)" + re.escape(a) + r"(?!\w)", b, part)
            if new != part:
                part = new
                changed = True
        parts[i] = part
    out = "".join(parts)
    return out if changed else None


def _latin_quotes(s: str) -> str:
    return s.replace("«", '"').replace("»", '"').replace("“", '"').replace("”", '"')


def realise(item: Item, variants: dict[str, str] | None = None,
            forms: tuple[str, ...] = FORMS) -> list[Query]:
    """All surface-form queries for an item."""
    variants = variants or {}
    el = item.question["el"]
    en = item.question.get("en", "")
    texts: dict[str, tuple[str, str]] = {}
    texts["el"] = (el, "el")
    if en:
        texts["en"] = (en, "en")
    texts["el_noacc"] = (tn.strip_accents(el), "el")
    texts["el_caps"] = (_latin_quotes(tn.caps_no_accents(el)), "el")
    texts["grk_elot"] = (_latin_quotes(tn.to_elot(el)), "el")
    texts["grk_phon"] = (_latin_quotes(tn.to_greeklish_phon(el)), "el")
    texts["grk_visual"] = (_latin_quotes(tn.to_greeklish_visual(el)), "el")
    if variants.get("el_infl"):
        texts["el_infl"] = (variants["el_infl"], "el")
    col = variants.get("el_colloq") or colloquial(el)
    if col:
        texts["el_colloq"] = (col, "el")
    out = []
    for f in forms:
        if f in texts:
            t, lang = texts[f]
            out.append(Query(qid=f"{item.iid}#{f}", iid=item.iid, form=f, lang=lang, text=t))
    return out
