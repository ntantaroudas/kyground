"""Deterministic, Greek-aware nugget matching.

The primary outcome of the study is computed here, without any model in the
loop, so that it is reproducible and cannot favour one model family. The LLM
judge (``scoring/judge.py``) is a secondary outcome validated against human
annotation.
"""

from __future__ import annotations

import re
from typing import Any

from ..schema import Item, Nugget
from .. import textnorm as tn


def _alias_match(alias: str, answer: str, ans_norm: str, ans_latin: str) -> bool:
    a = tn.norm(alias)
    if not a:
        return False
    # 1. normalised substring (word-bounded for short aliases)
    if len(a) <= 4:
        if re.search(rf"(?<!\w){re.escape(a)}(?!\w)", ans_norm):
            return True
    elif a in ans_norm:
        return True
    # 2. Greek inflection: stem-prefix containment of every content token
    if len(tn.content_tokens(alias, min_len=2)) >= 1 and tn.stem_match(alias, answer):
        return True
    # 3. cross-script: Greek alias vs Latin answer (or vice versa)
    al = tn.latin_key(alias)
    if len(al) >= 5 and al in ans_latin:
        return True
    return False


# a number nugget about these quantities must be stated with its unit (or bare), never as another quantity
UNIT_CUES = {
    "max_people": ["ατομ", "person", "people", "guest", "επισκεπτ", "συμμετεχ", "participant", "pax", "ομαδ", "group"],
    "lessons": ["μαθημ", "lesson", "ενοτητ", "unit"],
}


def _approx(x: float, y: float) -> bool:
    return abs(x - y) <= max(1e-6, 0.005 * abs(y))


def match_nugget(n: Nugget, answer: str, *, _cache: dict | None = None) -> bool:
    answer = answer or ""
    c = _cache if _cache is not None else {}
    ans_norm = c.setdefault("norm", tn.norm(answer))
    ans_latin = c.setdefault("latin", tn.latin_key(answer))
    aliases = list(n.aliases.get("el", [])) + list(n.aliases.get("en", []))
    k = n.kind
    if k in ("text", "list_member"):
        return any(_alias_match(a, answer, ans_norm, ans_latin) for a in aliases)
    if k == "number":
        v = float(n.value)
        if n.nid.startswith("price"):
            hit = any(_approx(x, v) for x in tn.currency_amounts(answer))
        elif n.nid in UNIT_CUES:
            hit = tn.number_with_unit(v, answer, UNIT_CUES[n.nid])
        else:
            hit = any(_approx(x, v) for x in c.setdefault("nums", tn.numbers_in(answer)))
        if hit:
            return True
        return any(_alias_match(a, answer, ans_norm, ans_latin) for a in aliases if not re.search(r"\d", a))
    if k == "num_range":
        nums = c.setdefault("nums", tn.numbers_in(answer))
        lo, hi = float(n.value["lo"]), float(n.value["hi"])
        return any(_approx(x, lo) for x in nums) and any(_approx(x, hi) for x in nums)
    if k == "duration_hours":
        durs = c.setdefault("durs", tn.durations_hours(answer))
        if any(_approx(x, float(n.value)) for x in durs):
            return True
        # "4ωρη", "τετράωρη" are not parsed: fall back to number next to a unit-like stem - but not when minutes
        # follow ("3 ώρες και 30 λεπτά" is 3.5 hours, not 3; fixed after the freeze, protocol deviation D12)
        v = float(n.value)
        return bool(re.search(rf"(?<![\d.,]){re.escape(f'{v:g}')}(?:[.,]0)?\s*(?:ωρ\w*\b|hours?\b|h\b)"
                              r"(?!\s*(?:και|and|&|,)?\s*\d+\s*(?:λεπτ|min))", ans_norm))
    if k == "months":
        ms = c.setdefault("months", tn.months_in(answer))
        s, e = int(n.value["start"]), int(n.value["end"])
        return (s in ms) and (e in ms)
    if k == "price_on_request":
        return (not tn.currency_amounts(answer)) and tn.mentions_on_request(answer)
    if k == "abstain":
        return tn.is_abstention(answer)
    if k == "no_leak":
        return not any(tn.norm(m) and tn.norm(m) in ans_norm for m in aliases)
    raise ValueError(f"unknown nugget kind {k}")


def score_answer(item: Item, answer: str) -> dict[str, Any]:
    """Item-level scores.

    - ``correct``: all required nuggets matched (for lists: every member)
    - ``recall``: fraction of required nuggets matched
    - ``lenient``: recall >= 0.5 (lists) or correct (others)
    - ``abstained``: the answer declines / says it is not available
    - ``fabricated_price``: a currency amount appears where none is published
    - ``hallucinated``: unanswerable item answered without abstaining, or with invented specifics
    """
    cache: dict = {}
    req = [n for n in item.nuggets if n.required]
    matched = [n.nid for n in req if match_nugget(n, answer, _cache=cache)]
    recall = len(matched) / len(req) if req else 0.0
    correct = bool(req) and len(matched) == len(req)
    abstained = tn.is_abstention(answer or "")
    fabricated_price = False
    if any(n.kind == "price_on_request" for n in req):
        fabricated_price = bool(tn.currency_amounts(answer or ""))
    is_list = any(n.kind == "list_member" for n in req)
    out = {
        "correct": int(correct),
        "recall": round(recall, 4),
        "lenient": int(recall >= 0.5) if is_list else int(correct),
        "matched": matched,
        "abstained": int(abstained),
        "fabricated_price": int(fabricated_price),
        "empty": int(not (answer or "").strip()),
    }
    if item.qtype == "unanswerable":
        # not abstaining, or abstaining but still stating a phone number, time, date, price, duration or calories
        out["hallucinated"] = int(not abstained or tn.has_specifics(answer or ""))
        if out["hallucinated"]:
            out["correct"] = 0
    return out
