"""Pseudonymised release of the benchmark and run outputs (Zenodo, CC BY 4.0).

The platform's public content names producers and hosts, some of whom are natural persons. Before release:

1. ``kyground pseudonymise --candidates`` writes ``data/release/pseudonyms.yaml``: every producer, contact
   person and host name in the snapshot, with a suggested role label and a guess whether it names a natural
   person (business words such as "Συνεταιρισμός" or "Seeds" suggest not). The authors check every entry; when in
   doubt, an entry is treated as a natural person.
2. ``kyground pseudonymise --map data/release/pseudonyms.yaml --runs runs/main`` copies the benchmark, the run
   outputs (not the model cache) and the result tables into ``release/``, replacing every entry marked
   ``natural_person: true`` by its label - in Greek with or without accents and in any case, and in Latin
   transliteration - in questions, gold answers, model answers and retrieved text alike, so that the benchmark
   stays consistent (the gold answer of "Who produces X?" becomes the label).
3. ``release/pseudonymisation_report.md`` lists the remaining occurrences of every original name (should be 0).

Run outputs store the identifiers of the records and chunks that reached the model, not their text, and the model
cache (which holds the full prompts, including retrieved platform content) is never copied, so the release does
not redistribute the knowledge base; the snapshot itself stays with the authors (data availability statement).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from . import textnorm as tn
from .schema import Record

BUSINESS_WORDS = ("μελισσοκομ", "συνεταιρισμ", "κτημα", "οινοποιει", "αγροκτημα", "εργαστηρι", "seeds", "farm",
                  "cooperative", "winery", "apiar", "ελαιουργικ", "εταιρ", "ike", "ο.ε", "ε.ε", "ltd")

_GREEK = {"α": "αάΑΆ", "ε": "εέΕΈ", "η": "ηήΗΉ", "ι": "ιίϊΐΙΊΪ", "ο": "οόΟΌ", "υ": "υύϋΰΥΎΫ", "ω": "ωώΩΏ",
          "σ": "σςΣ"}
_LATIN = {"a": "aáàâäAÁÀÂÄ", "e": "eéèêëEÉÈÊË", "i": "iíìîïIÍÌÎÏ", "o": "oóòôöOÓÒÔÖ", "u": "uúùûüUÚÙÛÜ"}
# letters that look alike in Greek and Latin capitals: platform text sometimes mixes them ("ΦΑΝΤAΣΤΙΚΟΥ")
_HOMOGLYPH = {"α": "aA", "a": "αΑ", "β": "B", "b": "Β", "ε": "eE", "e": "εΕ", "ζ": "Z", "z": "Ζ", "η": "H", "h": "Η",
              "ι": "iI", "i": "ιΙ", "κ": "kK", "k": "κΚ", "μ": "M", "m": "Μ", "ν": "N", "n": "Ν", "ο": "oO",
              "o": "οΟ", "ρ": "pP", "p": "ρΡ", "τ": "T", "t": "Τ", "υ": "yY", "y": "υΥ", "χ": "xX", "x": "χΧ"}
# nominative ending -> endings of the other cases, so that "του Μάρκου Ανυπαρκτέ" matches "Μάρκος Ανυπαρκτές"
_EL_ENDINGS = [("οσ", ("ου", "ο", "ε")), ("ησ", ("η",)), ("ασ", ("α",)), ("εσ", ("ε",)), ("α", ("ασ",)),
               ("η", ("ησ",))]
_LAT_ENDINGS = [("os", ("ou", "o", "e")), ("is", ("i",)), ("as", ("a",)), ("es", ("e",)), ("a", ("as",)),
                ("i", ("is",))]
_SEP = r"[\s_-]+"
_LEFT, _RIGHT = r"(?<![^\W_])", r"(?![^\W_])"     # word boundaries that treat "_" and "-" as separators


def _char_class(ch: str) -> str:
    base = tn.strip_accents(ch.lower()).replace("ς", "σ")
    chars = _GREEK.get(base) or _LATIN.get(base) or (base + base.upper() if base.isalpha() else "")
    if not chars:
        return r"\s+" if ch.isspace() else re.escape(ch)
    return "[" + "".join(dict.fromkeys(chars + _HOMOGLYPH.get(base, ""))) + "]"


def _token_forms(tok: str) -> set[str]:
    """The token and its other case forms (Greek or Latin transliteration), unaccented and lower-cased."""
    base = tn.strip_accents(tok.lower()).replace("ς", "σ")
    forms = {base}
    for end, others in (_EL_ENDINGS if tn.is_greek(tok) else _LAT_ENDINGS):
        if base.endswith(end) and len(base) - len(end) >= 3:
            forms |= {base[: -len(end)] + o for o in others}
            break
    return forms


def _token_pattern(tok: str) -> str:
    forms = sorted(_token_forms(tok), key=len, reverse=True)
    return "(?:" + "|".join("".join(_char_class(c) for c in f) for f in forms) + ")"


def name_regex(names: list[str]) -> re.Pattern:
    """One pattern for all spellings of a name: accent-, case- and final-sigma-insensitive, in any grammatical
    case, with spaces, hyphens or underscores between its words, in Greek or Latin script; the last word of a
    multi-word name (normally the surname) is also matched on its own when it has at least five letters."""
    alts: list[str] = []
    for n in sorted({n.strip() for n in names if n and n.strip()}, key=len, reverse=True):
        toks = n.split()
        alts.append(_SEP.join(_token_pattern(t) for t in toks))
        if len(toks) > 1 and len(tn.strip_accents(toks[-1])) >= 5:
            alts.append(_token_pattern(toks[-1]))
    return re.compile(rf"{_LEFT}(?:{'|'.join(alts)}){_RIGHT}", re.IGNORECASE)


def candidates(records: list[Record]) -> dict:
    """Names that may identify natural persons, with suggested labels, for the authors to review."""
    found: dict[str, dict] = {}
    counters = {"producer": 0, "contact": 0, "host": 0}
    prefix = {"producer": "Producer", "contact": "Contact person", "host": "Host"}

    def add(name: str, role: str, en: str = "") -> None:
        name = (name or "").strip()
        if not name or tn.norm(name) in {tn.norm(k) for k in found}:
            return
        counters[role] += 1
        label = f"{prefix[role]} {chr(64 + counters[role]) if counters[role] <= 26 else counters[role]}"
        business = any(w in tn.norm(name) for w in BUSINESS_WORDS)
        found[name] = {"label": label, "role": role, "natural_person": not business,
                       "variants": sorted({v for v in (en, tn.to_elot(name)) if v and v != name})}

    for r in records:
        a = r.attrs
        if r.rtype == "product":
            p = a.get("producer") or {}
            add(p.get("el", "") if isinstance(p, dict) else str(p), "producer",
                p.get("en", "") if isinstance(p, dict) else "")
            add(str(a.get("producer_contact_person") or ""), "contact")
        if r.rtype == "experience":
            h = a.get("host") or ""
            add(h.get("el", "") if isinstance(h, dict) else str(h), "host",
                h.get("en", "") if isinstance(h, dict) else "")
    return {"note": "Check every entry (when in doubt, set natural_person: true); only natural_person: true is "
                    "replaced.", "names": found}


def _replace(text: str, rules: list[tuple[re.Pattern, str]]) -> str:
    for pat, label in rules:
        text = pat.sub(label, text)
    return text


def _rules(mapping: dict) -> list[tuple[re.Pattern, str]]:
    rules = []
    for name, spec in (mapping.get("names") or {}).items():
        if spec.get("natural_person"):
            rules.append((name_regex([name, *spec.get("variants", [])]), spec["label"]))
    return rules


def _walk(obj, rules):
    if isinstance(obj, str):
        return _replace(obj, rules)
    if isinstance(obj, list):
        return [_walk(x, rules) for x in obj]
    if isinstance(obj, dict):
        return {k: _walk(v, rules) for k, v in obj.items()}
    return obj


def pseudonymise(mapping_path: Path, sources: list[Path], out_dir: Path, root: Path) -> dict:
    mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    rules = _rules(mapping)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for src in sources:
        files = [src] if src.is_file() else [p for p in src.rglob("*") if p.is_file() and "cache" not in p.parts]
        for f in files:
            if f.suffix not in (".jsonl", ".json", ".csv", ".md", ".yaml", ".txt"):
                continue
            rel = f.relative_to(root) if root in f.parents else Path(f.name)
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if f.suffix == ".jsonl":
                lines = [json.dumps(_walk(json.loads(ln), rules), ensure_ascii=False)
                         for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
                dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
            elif f.suffix == ".json":
                dst.write_text(json.dumps(_walk(json.loads(f.read_text(encoding="utf-8")), rules),
                                          ensure_ascii=False, indent=1), encoding="utf-8")
            else:
                dst.write_text(_replace(f.read_text(encoding="utf-8"), rules), encoding="utf-8")
            written.append(str(rel))
    # residual checks: the name in any spelling, and (stricter) any fragment of its last word
    texts = [p.read_text(encoding="utf-8") for p in out_dir.rglob("*") if p.is_file()
             and p.suffix in (".jsonl", ".json", ".csv", ".md", ".yaml", ".txt")]
    flat = [tn.norm(t) for t in texts]
    flat_latin = [tn.norm(tn.to_elot(t)) for t in texts]
    residual, fragments = {}, {}
    for name, spec in (mapping.get("names") or {}).items():
        if not spec.get("natural_person"):
            continue
        pat = name_regex([name, *spec.get("variants", [])])
        residual[spec["label"]] = sum(len(pat.findall(t)) for t in texts)
        stems = {tn.norm(w)[:-2] for w in [name.split()[-1], *[v.split()[-1] for v in spec.get("variants", [])]]
                 if len(tn.norm(w)) >= 6}
        fragments[spec["label"]] = sum(t.count(st) for st in stems for t in (flat + flat_latin) if st)
    lines = ["# Pseudonymisation report", "", f"{len(written)} files written; {len(rules)} names replaced.", "",
             "| Label | Remaining occurrences of the name | Remaining fragments of its last word |", "|---|---:|---:|"]
    lines += [f"| {k} | {v} | {fragments[k]} |" for k, v in residual.items()]
    lines += ["", "Both columns should be 0; a fragment count above 0 needs a look (it may also be an unrelated word)."]
    (out_dir / "pseudonymisation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"files": len(written), "names_replaced": len(rules), "residual": residual, "fragments": fragments}
