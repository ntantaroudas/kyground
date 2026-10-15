"""Controlled answer suite: validation of the scorers without human annotators.

For every benchmark item and both languages, answers whose correctness is known by construction are generated from
the gold facts. Correct answers state the facts in the reference wording and in varied realisations (another alias,
colloquial month names, minutes instead of hours, decimal comma or point, list members in another order,
surrounding text, Greek without accents); unanswerable items are answered with abstentions in several wordings.
Incorrect answers change exactly one thing: a different name, number, duration or month, one list member left out,
an abstention on an answerable question, an amount for a price on request, or, for unanswerable items, an invented
answer or an abstention that still states an invented specific (a phone number, time, price, duration, date or
calorie value).

The deterministic scorer and, optionally, the LLM judge grade every answer; sensitivity (correct answers graded
correct) and specificity (incorrect answers graded incorrect) are reported overall and by language, variant and
fact type. The suite tests whether the scoring rules behave as specified across controlled variation; it cannot
show how well they cover the phrasing of real answers, which is assessed by agreement with the LLM judge on the
experiments' own answers.

Usage (CLI): ``kyground scorer-suite --run runs/<id> [--judge]``
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .. import textnorm as tn
from ..schema import Item, Nugget
from .nuggets import UNIT_CUES, match_nugget, score_answer

COLLOQUIAL_MONTH = {1: "Γενάρη", 2: "Φλεβάρη", 3: "Μάρτη", 4: "Απρίλη", 5: "Μάη", 6: "Ιούνη", 7: "Ιούλη",
                    8: "Αύγουστο", 9: "Σεπτέμβρη", 10: "Οκτώβρη", 11: "Νοέμβρη", 12: "Δεκέμβρη"}
UNIT_WORD = {"max_people": ("άτομα", "people"), "lessons": ("μαθήματα", "lessons")}
FRAME = {"el": ("Σύμφωνα με τα στοιχεία της πλατφόρμας: {}.", " Για περισσότερες λεπτομέρειες δείτε τη σχετική σελίδα."),
         "en": ("According to the platform's records: {}.", " See the listing for more details.")}
ABSTAIN = {
    "el": ["Δεν βρήκα σχετική πληροφορία στην πλατφόρμα.", "Λυπάμαι, αυτό δεν αναφέρεται στα στοιχεία της πλατφόρμας.",
           "Η πλατφόρμα δεν έχει καταχωρημένη τέτοια πληροφορία.",
           "Δεν μπορώ να εντοπίσω κάτι τέτοιο στην πλατφόρμα· ίσως να επικοινωνήσετε με τον παραγωγό."],
    "en": ["I couldn't find any information about this on the platform.",
           "Sorry, the platform's records don't mention this.", "There is no such entry on the platform.",
           "That isn't listed on the platform; you may want to contact the producer directly."],
}
INVENTED = {"el": "Ναι, σύμφωνα με την πλατφόρμα αυτό το προσφέρει ένας τοπικός παραγωγός στον Ποταμό.",
            "en": "Yes, according to the platform this is offered by a local producer in Potamos."}
SPECIFICS = {  # an invented specific of each kind, after an abstention
    "phone": ("27360 00000", "27360 00000"), "time": ("στις 10:30", "at 10:30"), "price": ("12 €", "€12"),
    "duration": ("3 ώρες", "3 hours"), "date": ("στις 15 Μαΐου", "on 15 May"), "kcal": ("120 θερμίδες", "120 kcal"),
}
ABSTAIN_THEN = {"el": "Δεν βρήκα αυτή την πληροφορία στην πλατφόρμα, αλλά συνήθως είναι {}.",
                "en": "I couldn't find this on the platform, but it is usually {}."}
NOT_FOUND = {"el": "Δεν βρήκα αυτή την πληροφορία στην πλατφόρμα.",
             "en": "I couldn't find this information on the platform."}
PRICE_ON_REQUEST = {"el": "Η τιμή δεν αναγράφεται· δίνεται κατόπιν επικοινωνίας με τον παραγωγό",
                    "en": "The price isn't listed; it is available on request from the producer"}
WRONG_PRICE = {"el": "Κοστίζει {} €", "en": "It costs €{}"}


def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def _num(v: float, lang: str) -> str:
    s = f"{v:g}"
    return s.replace(".", ",") if lang == "el" else s


def _price(v: float, lang: str) -> str:
    s = f"{v:.2f}"
    return s.replace(".", ",") if lang == "el" else s


def _aliases(n: Nugget, lang: str) -> list[str]:
    return list(n.aliases.get(lang) or []) or list(n.aliases.get("el" if lang == "en" else "en") or [])


def _unit(n: Nugget, lang: str) -> str:
    """The unit written after the number in the fact's alias ("2-3 χρόνια" -> "χρόνια")."""
    for a in _aliases(n, lang):
        m = re.match(r"^\s*[\d.,]+(?:\s*[-–]\s*[\d.,]+)?\s*(.*)$", a)
        if m and re.search(r"\d", a):
            return m.group(1).strip()
    return ""


def _months(start: int, end: int, lang: str, colloquial: bool) -> str:
    if lang == "el":
        name = COLLOQUIAL_MONTH if colloquial else tn.MONTH_NAME["el_acc"]
        return f"τον {name[start]}" if start == end else f"από τον {name[start]} ως τον {name[end]}"
    en = tn.MONTH_NAME["en"]
    return f"in {en[start]}" if start == end else f"between {en[start]} and {en[end]}"


def _duration(v: float, lang: str, alt: bool) -> str:
    h = int(v)
    m = int(round((v - h) * 60))
    if alt or h == 0:
        if m == 0:
            return f"{h * 60} λεπτά" if lang == "el" else f"{h * 60} minutes"
        if h == 0:
            return f"{m} λεπτά" if lang == "el" else f"{m} minutes"
        if lang == "el":
            return f"{h} {'ώρα' if h == 1 else 'ώρες'} και {m} λεπτά"
        return f"{h} {'hour' if h == 1 else 'hours'} and {m} minutes"
    return (_num(v, "el") + " ώρες") if lang == "el" else (_num(v, "en") + " hours")


def realise(n: Nugget, lang: str, alt: bool) -> str | None:
    """One fact in words; ``alt`` picks another realisation than the reference's."""
    k = n.kind
    if k in ("text", "list_member"):
        al = _aliases(n, lang)
        return (al[-1] if alt else al[0]) if al else None
    if k == "number":
        v = float(n.value)
        if n.nid.startswith("price"):
            return f"{_price(v, lang)} €" if lang == "el" else f"€{_price(v, lang)}"
        if n.nid in UNIT_WORD:
            return f"{_num(v, lang)} {UNIT_WORD[n.nid][0 if lang == 'el' else 1]}"
        u = _unit(n, lang)
        return f"{_num(v, lang)}{'' if u.startswith('%') else ' '}{u}".strip() if alt else (_aliases(n, lang) or [None])[0]
    if k == "num_range":
        lo, hi, u = float(n.value["lo"]), float(n.value["hi"]), _unit(n, lang)
        if not alt:
            return (_aliases(n, lang) or [None])[0]
        return (f"από {_num(lo, lang)} έως {_num(hi, lang)} {u}" if lang == "el" else
                f"from {_num(lo, lang)} to {_num(hi, lang)} {u}").strip()
    if k == "duration_hours":
        return _duration(float(n.value), lang, alt)
    if k == "months":
        return _months(int(n.value["start"]), int(n.value["end"]), lang, colloquial=alt)
    if k == "price_on_request":
        return PRICE_ON_REQUEST[lang]
    return None


def wrong(n: Nugget, lang: str, pool: dict[tuple[str, str], list[str]]) -> tuple[str, str] | None:
    """A realisation of the fact with a wrong value: (text, perturbation)."""
    k = n.kind
    if k == "text":
        cands = [d for d in pool.get((n.nid, lang), []) + pool.get(("*", lang), []) if not match_nugget(n, d)]
        if not cands:
            return None
        return cands[_h(n.nid + lang + str(n.aliases)) % len(cands)], "wrong_text"
    if k == "number":
        v = float(n.value)
        if n.nid.startswith("price"):
            return (f"{_price(v + 5, lang)} €" if lang == "el" else f"€{_price(v + 5, lang)}"), "wrong_number"
        w = v + 1 if 1900 <= v <= 2100 or v < 4 else v + max(1, round(0.25 * v))
        if n.nid in UNIT_WORD:
            return f"{_num(w, lang)} {UNIT_WORD[n.nid][0 if lang == 'el' else 1]}", "wrong_number"
        u = _unit(n, lang)
        return f"{_num(w, lang)}{'' if u.startswith('%') else ' '}{u}".strip(), "wrong_number"
    if k == "num_range":
        lo, hi, u = float(n.value["lo"]) + 1, float(n.value["hi"]) + 1, _unit(n, lang)
        return ((f"από {_num(lo, lang)} έως {_num(hi, lang)} {u}" if lang == "el" else
                 f"from {_num(lo, lang)} to {_num(hi, lang)} {u}").strip(), "wrong_range")
    if k == "duration_hours":
        return _duration(float(n.value) + 1, lang, alt=False), "wrong_duration"
    if k == "months":
        s, e = (int(n.value["start"]) + 1) % 12 + 1, (int(n.value["end"]) + 1) % 12 + 1
        return _months(s, e, lang, colloquial=False), "wrong_month"
    if k == "price_on_request":
        return WRONG_PRICE[lang].format(_price(15.0, lang) if lang == "el" else "15.00"), "amount_for_price_on_request"
    return None


def _frame(text: str, lang: str, item: Item, extra: bool) -> str:
    head, tail = FRAME[lang]
    return head.format(text) + (tail if extra else "")


def _distractor_pool(items: Iterable[Item]) -> dict[tuple[str, str], list[str]]:
    pool: dict[tuple[str, str], list[str]] = defaultdict(list)
    for it in items:
        for n in it.nuggets:
            if n.kind != "text":
                continue
            for lang in ("el", "en"):
                for a in (n.aliases.get(lang) or [])[:1]:
                    if a not in pool[(n.nid, lang)]:
                        pool[(n.nid, lang)].append(a)
                    if a not in pool[("*", lang)]:
                        pool[("*", lang)].append(a)
    return pool


def build_suite(items: list[Item]) -> list[dict[str, Any]]:
    """Answers with known correctness for every item and language."""
    pool = _distractor_pool(items)
    rows: list[dict[str, Any]] = []

    def add(it: Item, lang: str, variant: str, label: int, answer: str, kind: str = "") -> None:
        rows.append({"sid": f"{it.iid}|{lang}|{variant}", "iid": it.iid, "qtype": it.qtype, "origin": it.origin,
                     "lang": lang, "variant": variant, "fact_kind": kind, "label": label,
                     "question": it.question.get(lang) or it.question.get("el", ""), "answer": answer})

    for it in items:
        req = [n for n in it.nuggets if n.required]
        for lang in ("el", "en"):
            if it.qtype == "unanswerable":
                k = _h(it.iid + lang)
                for j in (k % 4, (k + 1) % 4):
                    add(it, lang, f"abstention_{j + 1}", 1, ABSTAIN[lang][j], "abstain")
                add(it, lang, "invented_answer", 0, INVENTED[lang], "abstain")
                spec = sorted(SPECIFICS)[k % len(SPECIFICS)]
                add(it, lang, f"abstention_with_{spec}", 0,
                    ABSTAIN_THEN[lang].format(SPECIFICS[spec][0 if lang == "el" else 1]), "abstain")
                continue
            kinds = "+".join(sorted({n.kind for n in req}))
            if it.answer_ref.get(lang):
                add(it, lang, "reference", 1, it.answer_ref[lang], kinds)
            facts = [realise(n, lang, alt=True) for n in req]
            if all(facts):
                lst = [f for n, f in zip(req, facts) if n.kind == "list_member"]
                other = [f for n, f in zip(req, facts) if n.kind != "list_member"]
                lst = lst[::-1]
                joined_list = (", ".join(lst[:-1]) + (" και " if lang == "el" else " and ") + lst[-1]) if len(lst) > 1 \
                    else "".join(lst)
                text = "; ".join([x for x in [joined_list] + other if x])
                alt = _frame(text, lang, it, extra=True)
                add(it, lang, "reworded", 1, alt, kinds)
                if lang == "el":
                    add(it, lang, "reworded_no_accents", 1, tn.strip_accents(alt), kinds)
                # one wrong fact (the first fact that can be falsified)
                for i, n in enumerate(req):
                    w = wrong(n, lang, pool) if n.kind != "list_member" else None
                    if w:
                        bad = list(facts)
                        bad[i] = w[0]
                        lst_b = [f for m, f in zip(req, bad) if m.kind == "list_member"][::-1]
                        oth_b = [f for m, f in zip(req, bad) if m.kind != "list_member"]
                        jb = (", ".join(lst_b[:-1]) + (" και " if lang == "el" else " and ") + lst_b[-1]) \
                            if len(lst_b) > 1 else "".join(lst_b)
                        add(it, lang, w[1], 0, _frame("; ".join([x for x in [jb] + oth_b if x]), lang, it, True),
                            n.kind)
                        break
                if len(lst) > 1:   # one list member left out
                    add(it, lang, "list_member_missing", 0,
                        _frame("; ".join([x for x in [", ".join(lst[1:])] + other if x]), lang, it, True),
                        "list_member")
            add(it, lang, "abstention_on_answerable", 0, NOT_FOUND[lang], kinds)
    return rows


def judge_binary(label: str, qtype: str) -> int:
    """Binary correctness from the judge's four classes (abstention is correct on unanswerable items)."""
    return int(label == "correct" or (qtype == "unanswerable" and label == "abstained"))


def score_suite(rows: list[dict], items: dict[str, Item]) -> list[dict]:
    out = []
    for r in rows:
        s = score_answer(items[r["iid"]], r["answer"])
        out.append({**r, "det_correct": int(s["correct"])})
    return out


def _rates(rows: list[dict], key: str) -> dict[str, Any]:
    pos = [r for r in rows if r["label"] == 1 and r.get(key) is not None]
    neg = [r for r in rows if r["label"] == 0 and r.get(key) is not None]
    sens = sum(r[key] for r in pos) / len(pos) if pos else None
    spec = sum(1 - r[key] for r in neg) / len(neg) if neg else None
    return {"n_pos": len(pos), "n_neg": len(neg), "sensitivity": sens, "specificity": spec}


def summarise(rows: list[dict]) -> dict[str, Any]:
    """Sensitivity and specificity of each scorer, overall and by language, variant, fact kind and origin."""
    out: dict[str, Any] = {"n": len(rows)}
    for key, name in (("det_correct", "deterministic"), ("judge_correct", "judge")):
        if not any(r.get(key) is not None for r in rows):
            continue
        res = {"overall": _rates(rows, key)}
        for dim in ("lang", "variant", "fact_kind", "origin"):
            groups: dict[str, list[dict]] = defaultdict(list)
            for r in rows:
                groups[str(r[dim])].append(r)
            res[f"by_{dim}"] = {g: _rates(v, key) for g, v in sorted(groups.items())}
        res["errors"] = [{"sid": r["sid"], "label": r["label"], "answer": r["answer"]}
                         for r in rows if r.get(key) is not None and r[key] != r["label"]][:200]
        out[name] = res
    return out


def write_suite(rows: list[dict], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "scorer_suite.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarise(rows)
    (out_dir / "scorer_suite.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary
