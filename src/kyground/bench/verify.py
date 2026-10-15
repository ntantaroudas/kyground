"""Automatic verification of the gold answers against the frozen snapshot.

Every item is checked against the records that the systems see (the gated, bilingual text of
:func:`kyground.kb.serialize.record_text`), without any model in the loop:

- ``records_public``: every gold record exists in the snapshot and is published, active and not excluded;
- ``rederived`` (template items): regenerating the templates from the snapshot gives the same question, gold facts,
  gold records and reference answer;
- ``evidence_quoted`` (items that quote evidence): the quote occurs in the gold records' text (after Unicode, case,
  accent and final-sigma normalisation);
- ``facts_in_records``: every required gold fact is found in the gold records' text by the deterministic scorer,
  as if the records were an answer; a price on request requires that no gold record shows its price;
- ``reference_scores_correct``: the reference answers (Greek and English) are scored correct by the scorer;
- ``derived`` (aggregation items with a ``derive`` rule): recomputing the aggregate from the snapshot (for example
  the experience with the longest duration, or every experience available only in May) selects exactly the
  records that the gold facts name;
- ``absence_verified`` (unanswerable items): the terms that the question asks about occur in no published record
  (or, for a missing attribute of a real record, not in that record), and no phone number, clock time, date,
  calorie value, duration or price occurs where the specification says it must not.

Absence is specified on the item's ``abstain`` fact, e.g. ``value: {absent: ["Πέτρινος Ανεμόμυλος"]}``,
``value: {absent: ["θερμίδ", "kcal"], patterns: [kcal], scope: gold}`` (only the gold records) or
``value: {patterns: [time], within: ["ελαιοτριβ"]}`` (only the records that mention an olive mill);
``no_records: [cooperative]`` requires that no record of that type is published. Evidence quotes
may elide text with "...": each part must occur. Items that pass every applicable check are
marked ``verified``; the others ``needs_fix``, with reasons. Real runs use verified items only.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .. import textnorm as tn
from ..kb.serialize import record_text
from ..schema import Item, Record
from ..scoring.nuggets import match_nugget, score_answer
from .templates import template_items

PATTERNS = {
    "phone": tn.has_phone, "time": tn.has_time, "kcal": tn.has_kcal, "date": tn.has_date,
    "price": lambda t: bool(tn.currency_amounts(t)), "duration": lambda t: bool(tn.durations_hours(t)),
}


def _nugget_key(n) -> tuple:
    return (n.nid, n.kind, json.dumps(n.value, sort_keys=True, ensure_ascii=False),
            tuple(n.aliases.get("el", [])), tuple(n.aliases.get("en", [])), n.required)


def _contains(text_norm: str, text_latin: str, term: str) -> bool:
    t = tn.norm(term)
    if t and t in text_norm:
        return True
    lat = tn.latin_key(term)
    return len(lat) >= 5 and lat in text_latin


def derive_records(spec: dict, recs: dict[str, Record]) -> list[Record]:
    """Records selected by an aggregation rule: ``rtype``, optional ``parent``, ``any_terms`` (in ``fields``,
    default title), ``months_only`` (the months named in ``field`` are exactly these), ``argmax`` (attribute;
    ties kept) and ``report: parent`` (return the parents of the selected records)."""
    cands = [r for r in recs.values() if r.is_public and r.rtype == spec["rtype"]]
    if spec.get("parent"):
        cands = [r for r in cands if r.parent == spec["parent"]]
    if spec.get("any_terms"):
        fields = spec.get("fields", ["title"])

        def txt(r: Record) -> str:
            return tn.norm(" ".join(r.text(f, lang) for f in fields for lang in ("el", "en")))
        cands = [r for r in cands if any(tn.norm(t) in txt(r) for t in spec["any_terms"])]
    if spec.get("months_only"):
        want = set(spec["months_only"])
        cands = [r for r in cands if set(tn.months_in(r.attr_text(spec.get("field", "availability_note"), "el")))
                 == want]
    if spec.get("argmax") and cands:
        vals = [(float(r.attrs.get(spec["argmax"]) or 0), r) for r in cands]
        top = max(v for v, _ in vals)
        cands = [r for v, r in vals if v == top]
    if spec.get("report") == "parent":
        out: list[Record] = []
        for r in cands:
            p = recs.get(r.parent or "")
            if p is not None and p not in out:
                out.append(p)
        cands = out
    return cands


def _derived_ok(it: Item, chosen: list[Record]) -> bool:
    """The rule's records and the gold facts correspond one to one (by title, in either language)."""
    facts = [n for n in it.nuggets if n.required and n.kind in ("text", "list_member")]
    if not chosen or len(facts) != len(chosen):
        return False
    titles = [" / ".join(v for v in (r.title.get("el"), r.title.get("en")) if v) for r in chosen]
    return all(any(match_nugget(n, t) for n in facts) for t in titles) and \
        all(any(match_nugget(n, t) for t in titles) for n in facts)


def _quote_found(quote: str, text_norm: str) -> bool:
    parts = [q for q in re.split(r"\.\.\.|…", quote) if tn.norm(q)]
    return bool(parts) and all(tn.norm(q) in text_norm for q in parts)


def verify_item(it: Item, recs: dict[str, Record], derived: dict[str, Item], kb_text: str,
                kb_latin: str) -> dict[str, Any]:
    checks: dict[str, bool | None] = {}
    fails: list[str] = []
    gold = [recs.get(g) for g in it.gold_rids]
    ok_pub = all(r is not None and r.is_public for r in gold)
    checks["records_public"] = ok_pub
    if not ok_pub:
        fails.append("gold record missing or not public: " +
                     ", ".join(g for g, r in zip(it.gold_rids, gold) if r is None or not r.is_public))
    texts = [record_text(r) for r in gold if r is not None]
    gold_text = "\n\n".join(texts)
    gold_norm, gold_latin = tn.norm(gold_text), tn.latin_key(gold_text)

    if it.origin == "template":
        d = derived.get(it.iid)
        same = d is not None and d.question == it.question and sorted(d.gold_rids) == sorted(it.gold_rids) and \
            [_nugget_key(n) for n in d.nuggets] == [_nugget_key(n) for n in it.nuggets] and \
            d.answer_ref == it.answer_ref
        checks["rederived"] = same
        if not same:
            fails.append("template output differs from the item" if d else "template no longer generates the item")

    if it.origin != "template" and any(it.evidence.get(lang) for lang in ("el", "en")):
        missing = [lang for lang in ("el", "en") if it.evidence.get(lang) and
                   not _quote_found(it.evidence[lang], gold_norm)]
        checks["evidence_quoted"] = not missing
        if missing:
            fails.append("evidence quote not found in the gold records: " + ", ".join(missing))

    if it.qtype != "unanswerable":
        bad = []
        for n in it.nuggets:
            if not n.required:
                continue
            if n.kind == "price_on_request":
                priced = [r for r in gold if r is not None and r.rtype in ("product", "experience")]
                if not priced or any(r.price_visible and r.price is not None for r in priced):
                    bad.append(n.nid)
            elif not match_nugget(n, gold_text):
                bad.append(n.nid)
        checks["facts_in_records"] = not bad
        if bad:
            fails.append("gold facts not found in the gold records: " + ", ".join(bad))

    if it.derive:
        chosen = derive_records(it.derive, recs)
        checks["derived"] = _derived_ok(it, chosen)
        if not checks["derived"]:
            fails.append("the aggregation rule selects " + (", ".join(r.rid for r in chosen) or "no record") +
                         ", which does not match the gold facts")

    wrong = [lang for lang in ("el", "en") if it.answer_ref.get(lang) and
             not score_answer(it, it.answer_ref[lang])["correct"]]
    checks["reference_scores_correct"] = not wrong and bool(it.answer_ref)
    if wrong or not it.answer_ref:
        fails.append("reference answer not scored correct: " + (", ".join(wrong) or "no reference answer"))

    if it.qtype == "unanswerable":
        spec = next((n.value for n in it.nuggets if n.kind == "abstain" and isinstance(n.value, dict)), None)
        if not spec or not (spec.get("absent") or spec.get("patterns") or spec.get("no_records")):
            checks["absence_verified"] = False
            fails.append("no absence specification (value: {absent: [...]}) on the abstain fact")
        else:
            scope_gold = spec.get("scope") == "gold"
            if scope_gold and not texts:
                checks["absence_verified"] = False
                fails.append("scope 'gold' but the item has no gold records")
            else:
                if scope_gold:
                    raw = gold_text
                elif spec.get("within"):
                    pub = [record_text(r) for r in recs.values() if r.is_public]
                    raw = "\n\n".join(t for t in pub if any(tn.norm(w) in tn.norm(t) for w in spec["within"]))
                else:
                    raw = kb_text
                t_norm, t_lat = tn.norm(raw), tn.latin_key(raw)
                found = [a for a in spec.get("absent", []) if _contains(t_norm, t_lat, a)]
                found += [f"pattern:{p}" for p in spec.get("patterns", []) if PATTERNS[p](raw)]
                found += [f"records:{t}" for t in spec.get("no_records", [])
                          if any(r.is_public and r.rtype == t for r in recs.values())]
                checks["absence_verified"] = not found
                if found:
                    where = "gold records" if scope_gold else ("records within scope" if spec.get("within") else
                                                               "knowledge base")
                    fails.append(f"asked-about information occurs in the {where}: " + ", ".join(found))
    return {"iid": it.iid, "origin": it.origin, "qtype": it.qtype, "checks": checks, "failures": fails,
            "status": "verified" if not fails else "needs_fix"}


def verify_items(items: list[Item], records: list[Record]) -> list[dict]:
    recs = {r.rid: r for r in records}
    derived = {d.iid: d for d in template_items(records)}
    pub = [r for r in records if r.is_public]
    kb_text = "\n\n".join(record_text(r) for r in pub)
    kb_latin = tn.latin_key(kb_text)
    return [verify_item(it, recs, derived, kb_text, kb_latin) for it in items]


def apply_verification(items: list[Item], results: list[dict]) -> list[Item]:
    by = {r["iid"]: r for r in results}
    for it in items:
        r = by.get(it.iid)
        if r:
            it.status = r["status"]
    return items


def write_report(results: list[dict], out_dir: Path, snapshot: str = "") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "snapshot": snapshot, "items": len(results),
        "verified": sum(r["status"] == "verified" for r in results),
        "needs_fix": sum(r["status"] != "verified" for r in results),
        "by_origin": dict(Counter(f"{r['origin']}:{r['status']}" for r in results)),
        "checks": {},
    }
    for name in ("records_public", "rederived", "evidence_quoted", "facts_in_records", "derived",
                 "reference_scores_correct", "absence_verified"):
        vals = [r["checks"][name] for r in results if r["checks"].get(name) is not None]
        summary["checks"][name] = {"applied": len(vals), "passed": sum(bool(v) for v in vals)}
    (out_dir / "gold_verification.json").write_text(
        json.dumps({"summary": summary, "items": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# Gold verification", "", f"Snapshot: `{snapshot}`" if snapshot else "", "",
             f"{summary['verified']} of {summary['items']} items verified; {summary['needs_fix']} need a fix.", "",
             "| Check | Applied to | Passed |", "|---|---:|---:|"]
    lines += [f"| {k} | {v['applied']} | {v['passed']} |" for k, v in summary["checks"].items()]
    bad = [r for r in results if r["status"] != "verified"]
    if bad:
        lines += ["", "## Items that need a fix", ""]
        lines += [f"- `{r['iid']}`: " + "; ".join(r["failures"]) for r in bad]
    (out_dir / "gold_verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
