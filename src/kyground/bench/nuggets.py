"""Constructors for typed gold nuggets."""

from __future__ import annotations

from typing import Iterable

from ..schema import Nugget
from .. import textnorm as tn


def _clean(xs: Iterable[str | None]) -> list[str]:
    out: list[str] = []
    for x in xs:
        if x and x.strip() and x.strip() not in out:
            out.append(x.strip())
    return out


def text(nid: str, el: Iterable[str | None] = (), en: Iterable[str | None] = (), required: bool = True) -> Nugget:
    return Nugget(nid=nid, kind="text", aliases={"el": _clean(el), "en": _clean(en)}, required=required)


def member(nid: str, el: Iterable[str | None] = (), en: Iterable[str | None] = ()) -> Nugget:
    return Nugget(nid=nid, kind="list_member", aliases={"el": _clean(el), "en": _clean(en)})


def number(nid: str, value: float, el: Iterable[str] = (), en: Iterable[str] = ()) -> Nugget:
    return Nugget(nid=nid, kind="number", value=float(value), aliases={"el": _clean(el), "en": _clean(en)})


def num_range(nid: str, lo: float, hi: float, el: Iterable[str] = (), en: Iterable[str] = ()) -> Nugget:
    """Both bounds of a numeric range must appear (e.g. "2-3 χρόνια")."""
    return Nugget(nid=nid, kind="num_range", value={"lo": float(lo), "hi": float(hi)},
                  aliases={"el": _clean(el) or [f"{lo:g}-{hi:g}"], "en": _clean(en) or [f"{lo:g}-{hi:g}"]})


def duration_hours(nid: str, hours: float) -> Nugget:
    h = float(hours)
    el = f"{h:g}".replace(".", ",") + " ώρες"
    en = f"{h:g} hours"
    return Nugget(nid=nid, kind="duration_hours", value=h, aliases={"el": [el], "en": [en]})


def months(nid: str, start: int, end: int | None = None) -> Nugget:
    end = start if end is None else end
    el = tn.MONTH_NAME["el"][start] + ("" if end == start else f" – {tn.MONTH_NAME['el'][end]}")
    en = tn.MONTH_NAME["en"][start] + ("" if end == start else f" – {tn.MONTH_NAME['en'][end]}")
    return Nugget(nid=nid, kind="months", value={"start": start, "end": end}, aliases={"el": [el], "en": [en]})


def price_on_request(nid: str = "price_on_request") -> Nugget:
    return Nugget(nid=nid, kind="price_on_request",
                  aliases={"el": ["Τιμή κατόπιν επικοινωνίας"], "en": ["Price on request"]})


def abstain(nid: str = "abstain") -> Nugget:
    return Nugget(nid=nid, kind="abstain",
                  aliases={"el": ["Η πληροφορία δεν υπάρχει στην πλατφόρμα"],
                           "en": ["The information is not available on the platform"]})


def no_leak(nid: str, markers: Iterable[str]) -> Nugget:
    return Nugget(nid=nid, kind="no_leak", aliases={"el": _clean(markers), "en": []})


def describe(n: Nugget, lang: str = "el") -> str:
    """Human-readable rendering (review sheets, judge reference)."""
    al = n.aliases.get(lang) or n.aliases.get("el") or n.aliases.get("en") or []
    if n.kind == "price_on_request":
        return "τιμή κατόπιν επικοινωνίας (καμία αριθμητική τιμή)" if lang == "el" else \
            "price on request (no numeric price)"
    if n.kind == "abstain":
        return "η πληροφορία δεν υπάρχει (αναμένεται άρνηση/αποχή)" if lang == "el" else \
            "information not available (abstention expected)"
    if n.kind == "no_leak":
        return "δεν πρέπει να εμφανίζεται: " + ", ".join(al)
    head = al[0] if al else str(n.value)
    if len(al) > 1:
        head += " (ή: " + "; ".join(al[1:3]) + ")" if lang == "el" else " (or: " + "; ".join(al[1:3]) + ")"
    return head
