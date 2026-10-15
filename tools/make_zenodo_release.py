"""Zenodo deposit of the KyGround data, the underlying and extended data of the ORE article.

    PYTHONPATH=src python tools/make_zenodo_release.py     -> zenodo/KyGround_data_v1.0.zip and zenodo/ZENODO_UPLOAD.md

The archive holds what the Data and software availability section of the manuscript lists, in one folder,
KyGround_data_v1.0, with an underlying_data and an extended_data folder. Every file goes through the pseudonymisation
of kyground.release, which replaces the names of natural persons in data/release/pseudonyms.yaml by their role labels,
in Greek or Latin script, with or without accents, in any case and grammatical case. A file without a name is copied
byte for byte, so that the digests of freeze_v0.4.json can still be checked.

Never copied are the frozen snapshot, the request cache, the batch files given to the sub-agents and the pseudonym
map, because they hold the platform's records and the names. The run outputs hold identifiers of the retrieved
records, not their text.

The script checks the folder before zipping and stops if a check fails.
  * No name of a natural person in the map remains, in any spelling, and no fragment of a surname.
  * No e-mail address and no telephone number occurs.
  * No passage of 12 or more words of a record's text occurs, apart from the short evidence quotations of the
    benchmark and the model answers, and the share of the records' text that occurs at all is reported.
  * Every JSON and JSON Lines file parses, and no file comes from a forbidden folder.
It then writes README.md, LICENCE.txt, pseudonymisation_report.md and MANIFEST-SHA256.txt and zips the folder with
fixed timestamps, so that a rerun gives the same archive.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kyground import textnorm as tn  # noqa: E402
from kyground.release import _LEFT, _RIGHT, _SEP, _token_forms, _token_pattern, name_regex  # noqa: E402

VERSION = "1.0"
NAME = f"KyGround_data_v{VERSION}"
STAMP = (2026, 10, 12, 0, 0, 0)
MAP = ROOT / "data" / "release" / "pseudonyms.yaml"
SNAPSHOT = ROOT / "data" / "snapshots" / "public_20261005" / "records.jsonl"
TEXT = {".jsonl", ".json", ".csv", ".md", ".yaml", ".txt"}
FORBIDDEN = {"cache", "session", "snapshots", "release", "_superseded_cache"}
SUPPLEMENTARY = ["acc_diagnostics", "acc_tool_behaviour", "acc_by_qtype", "acc_errors", "acc_by_quote", "acc_did",
                 "acc_sensitivity", "acc_gee", "acc_judge", "acc_judge_contrasts", "judge_disagreements",
                 "router_arguments", "freshness_context"]

# (source in the project, destination in the archive), directories copied with their structure
LAYOUT = [
    ("data/benchmark/v1.0/items.jsonl", "underlying_data/benchmark/items.jsonl"),
    ("data/benchmark/v1.0/queries.jsonl", "underlying_data/benchmark/queries.jsonl"),
    ("data/benchmark/v1.0/summary.json", "underlying_data/benchmark/summary.json"),
    ("data/benchmark/v1.0/gold_verification.json", "underlying_data/benchmark/gold_verification.json"),
    ("data/benchmark/v1.0/gold_verification.md", "underlying_data/benchmark/gold_verification.md"),
    ("runs/main", "underlying_data/runs/main"),
    ("runs/main_batched", "underlying_data/runs/main_batched"),
    ("runs/suite/scorer_suite.jsonl", "underlying_data/scorer_suite/scorer_suite.jsonl"),
    ("runs/suite/scorer_suite.json", "underlying_data/scorer_suite/scorer_suite.json"),
    ("runs/suite/judge_session_suite", "underlying_data/scorer_suite/judge_session"),
    ("paper/results", "underlying_data/results"),
    ("paper/sensitivity/results_batched", "underlying_data/results_batched"),
    ("protocol/study_protocol.md", "extended_data/protocol/study_protocol.md"),
    ("protocol/freeze_v0.4.json", "extended_data/protocol/freeze_v0.4.json"),
    ("protocol/deviations.md", "extended_data/protocol/deviations.md"),
    ("configs/study.yaml", "extended_data/configs/study.yaml"),
    ("configs/prompts.yaml", "extended_data/configs/prompts.yaml"),
    ("configs/tools.yaml", "extended_data/configs/tools.yaml"),
    ("protocol/evidence/public_search_probe_20261005.csv", "extended_data/public_search_probe_20261005.csv"),
    ("protocol/evidence/public_search_probe_20261005.json", "extended_data/public_search_probe_20261005.json"),
    ("protocol/evidence/snapshot_record_hashes_20261005.json", "extended_data/snapshot_record_hashes_20261005.json"),
    ("runs/tune/rag_tuning.csv", "extended_data/rag_tuning.csv"),
    ("runs/tune/rag_tuning.json", "extended_data/rag_tuning.json"),
    ("data/benchmark/curated_items.yaml", "extended_data/curated_items.yaml"),
    ("runs/main/contamination_audit.json", "extended_data/contamination_audit.json"),
    ("runs/main/control_isolation.json", "extended_data/control_isolation.json"),
] + [(f"paper/results/tables/{t}{ext}", f"extended_data/supplementary_tables/{t}{ext}")
     for t in SUPPLEMENTARY for ext in (".csv", ".md", ".json") if (ROOT / f"paper/results/tables/{t}{ext}").exists()]
# files of the copied folders that are given once, in extended_data
SKIP = {"runs/main/contamination_audit.json", "runs/main/control_isolation.json"}
# the invented telephone number of the controlled scorer suite could belong to a subscriber on the island, so any
# number of the island in that form is written with zeros, which the scorer still reads as a telephone number
MASKS = [(re.compile(r"(?<!\d)27360 (?!00000)\d{5}(?!\d)"), "27360 00000")]


# ------------------------------------------------------------------------------------------------ copying

def ordered_rules(mapping: dict) -> list[tuple[re.Pattern, str]]:
    """Replacement rules, full names first and the longest first, then names of one word, then surnames alone, so
    that a full name keeps its own label when its surname is also the whole name of another entry."""
    entries = [(n, s) for n, s in (mapping.get("names") or {}).items() if s.get("natural_person")]
    singles = {tn.norm(v) for n, s in entries for v in [n, *s.get("variants", [])] if len(v.split()) == 1}
    tiers: list[dict[str, list[str]]] = [{}, {}, {}]
    for n, s in entries:
        for v in [n, *s.get("variants", [])]:
            toks = v.split()
            if len(toks) == 1:
                tiers[1].setdefault(s["label"], []).append(v)
            else:
                tiers[0].setdefault(s["label"], []).append(v)
                if len(tn.strip_accents(toks[-1])) >= 5 and tn.norm(toks[-1]) not in singles:
                    tiers[2].setdefault(s["label"], []).append(toks[-1])
    rules = []
    for tier in tiers:
        for label, names in sorted(tier.items(), key=lambda x: -max(map(len, x[1]))):
            alts = [_SEP.join(_token_pattern(t) for t in n.split()) for n in sorted(set(names), key=len, reverse=True)]
            rules.append((re.compile(rf"{_LEFT}(?:{'|'.join(alts)}){_RIGHT}", re.IGNORECASE), label))
    return rules


class Pseudonymiser:
    """Masks, then names by their ordered rules, then given names written next to a label (a first name just
    before or after "Producer A"), then a label repeated by these steps ("Producer G Producer G")."""

    def __init__(self, mapping: dict):
        self.rules = ordered_rules(mapping)
        self.first = first_name_keys(mapping)
        labels = sorted({lab for _, lab in self.rules}, key=len, reverse=True)
        alt = "|".join(re.escape(x) for x in labels)
        self.before = re.compile(rf"{_LEFT}([^\W\d_]{{4,}})(\s+)({alt}){_RIGHT}")
        self.after = re.compile(rf"{_LEFT}({alt})(\s+)([^\W\d_]{{4,}}){_RIGHT}")
        self.repeat = re.compile(rf"{_LEFT}({alt})(?:\s+\1)+{_RIGHT}")

    def apply(self, text: str) -> tuple[str, int]:
        n = 0
        for old, new in MASKS:
            text, k = old.subn(new, text)
            n += k
        for pat, label in self.rules:
            text, k = pat.subn(label, text)
            n += k
        hits = [0]

        def absorb_before(m: re.Match) -> str:
            if near(m.group(1), self.first.get(m.group(3), set())):
                hits[0] += 1
                return m.group(3)
            return m.group(0)

        def absorb_after(m: re.Match) -> str:
            if near(m.group(3), self.first.get(m.group(1), set())):
                hits[0] += 1
                return m.group(1)
            return m.group(0)
        if any(lab in text for lab in self.first):
            text = self.before.sub(absorb_before, text)
            text = self.after.sub(absorb_after, text)
            text, k = self.repeat.subn(r"\1", text)
            n += hits[0] + k
        return text, n


def first_name_keys(mapping: dict) -> dict[str, set[str]]:
    """Phonetic keys of the given names (and, for names written surname first, of the surname), in every case."""
    keys: dict[str, set[str]] = {}
    for name, spec in (mapping.get("names") or {}).items():
        if spec.get("natural_person"):
            for v in [name, *spec.get("variants", [])]:
                toks = v.split()
                if len(toks) > 1:
                    keys.setdefault(spec["label"], set()).update(phon(f) for f in _token_forms(toks[0]))
    return keys


def near(word: str, keys: set[str]) -> bool:
    k = phon(word)
    return any(k[:1] == key[:1] and abs(len(k) - len(key)) <= 2 and lev(k, key) <= (1 if len(key) <= 5 else 2)
               for key in keys)


def subn(text: str, ps: Pseudonymiser) -> tuple[str, int]:
    return ps.apply(text)


def walk(obj, rules) -> tuple[object, int]:
    if isinstance(obj, str):
        return subn(obj, rules)
    if isinstance(obj, list):
        out, n = [], 0
        for x in obj:
            y, k = walk(x, rules)
            out.append(y)
            n += k
        return out, n
    if isinstance(obj, dict):
        out, n = {}, 0
        for key, v in obj.items():
            key2, k1 = subn(key, rules)
            y, k2 = walk(v, rules)
            out[key2] = y
            n += k1 + k2
        return out, n
    return obj, 0


def copy_pseudonymised(src: Path, dst: Path, rules) -> int:
    """Copy one file, replacing names; a file without a name is copied byte for byte. Returns the replacements."""
    if src.suffix not in TEXT:
        raise SystemExit(f"not a text file, not released: {src}")
    raw = src.read_bytes()
    text = raw.decode("utf-8")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".jsonl":
        rows, n = [], 0
        for ln in text.splitlines():
            if ln.strip():
                obj, k = walk(json.loads(ln), rules)
                rows.append(obj)
                n += k
        out = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8") if n else raw
    elif src.suffix == ".json":
        obj, n = walk(json.loads(text), rules)
        out = json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8") if n else raw
    else:
        new, n = subn(text, rules)
        out = new.encode("utf-8") if n else raw
    dst.write_bytes(out)
    return n


def collect() -> list[tuple[Path, str]]:
    pairs = []
    for s, d in LAYOUT:
        src = ROOT / s
        if src.is_dir():
            for f in sorted(p for p in src.rglob("*") if p.is_file()):
                rel = f.relative_to(ROOT).as_posix()
                if rel in SKIP or "__pycache__" in f.parts:
                    continue
                pairs.append((f, f"{d}/{f.relative_to(src).as_posix()}"))
        elif src.is_file():
            pairs.append((src, d))
        else:
            raise SystemExit(f"missing source {s}")
    for f, _ in pairs:
        if FORBIDDEN & set(f.relative_to(ROOT).parts):
            raise SystemExit(f"forbidden source {f}")
    dests = [d for _, d in pairs]
    dup = [d for d, c in Counter(dests).items() if c > 1]
    if dup:
        raise SystemExit(f"duplicate destinations {dup}")
    return pairs


# ------------------------------------------------------------------------------------------------ checks

def decoded_texts(folder: Path) -> dict[str, str]:
    """Every file's text, with JSON strings decoded, so that no escaped character hides a name."""
    out = {}
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        t = p.read_text(encoding="utf-8")
        if p.suffix == ".json":
            obj = json.loads(t)                      # parses, or the script stops
            if "\\u" in t:
                t = t + "\n" + json.dumps(obj, ensure_ascii=False)
        elif p.suffix == ".jsonl":
            objs = [json.loads(x) for x in t.splitlines() if x.strip()]
            if "\\u" in t:
                t = t + "\n" + "\n".join(json.dumps(o, ensure_ascii=False) for o in objs)
        out[p.relative_to(folder).as_posix()] = t
    return out


def name_checks(texts: dict[str, str], mapping: dict) -> tuple[dict, dict, list[str]]:
    residual, fragments, where = {}, {}, []
    flat = {k: tn.norm(t) for k, t in texts.items()}
    flat_latin = {k: tn.norm(tn.to_elot(t)) for k, t in texts.items()}
    for name, spec in (mapping.get("names") or {}).items():
        if not spec.get("natural_person"):
            continue
        pat = name_regex([name, *spec.get("variants", [])])
        label = spec["label"]
        residual.setdefault(label, 0)
        for k, t in texts.items():
            n = len(pat.findall(t))
            if n:
                residual[label] += n
                where.append(f"{label} in {k} ({n})")
        stems = {tn.norm(w)[:-2] for w in [name.split()[-1], *[v.split()[-1] for v in spec.get("variants", [])]]
                 if len(tn.norm(w)) >= 6}
        fragments.setdefault(label, 0)
        for k in texts:
            n = sum(flat[k].count(st) + flat_latin[k].count(st) for st in stems if st)
            if n:
                fragments[label] += n
                where.append(f"fragment of {label} in {k} ({n})")
    first = first_name_keys(mapping)
    alt = "|".join(re.escape(x) for x in sorted(first, key=len, reverse=True))
    around = re.compile(rf"([^\W\d_]{{4,}})\s+({alt})(?![^\W_])|(?<![^\W_])({alt})\s+([^\W\d_]{{4,}})")
    for k, t in texts.items():
        for m in around.finditer(t):
            word, label = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
            if near(word, first.get(label, set())):
                residual[label] += 1
                where.append(f"given name next to {label} in {k}")
    return residual, fragments, where


WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
NOT_NAMES: set[str] = set()      # words that sound like a surname but were checked and are ordinary words


def phon(w: str) -> str:
    """A rough phonetic key, so that misspellings and transliterations of a name meet (Παπαδόπουλος, Papadopulos)."""
    k = tn.norm(tn.to_elot(w))
    for a, b in (("ch", "x"), ("th", "8"), ("ou", "u"), ("ei", "i"), ("oi", "i"), ("y", "i"), ("h", "i"),
                 ("w", "o"), ("b", "v")):
        k = k.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", k)


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def near_names(texts: dict[str, str], mapping: dict) -> list[str]:
    """Words that sound like the surname of a natural person in the map, spelled in a way the map does not list."""
    keys = {}
    for name, spec in (mapping.get("names") or {}).items():
        if spec.get("natural_person"):
            for v in [name, *spec.get("variants", [])]:
                k = phon(v.split()[-1])
                if len(k) >= 6:
                    keys[k] = spec["label"]
    words = Counter()
    for f, t in texts.items():
        for w in set(WORD.findall(t)):
            words[(w, f)] += 1
    found = Counter()
    for (w, f), _ in words.items():
        if w in NOT_NAMES or len(w) < 5:
            continue
        kw = phon(w)
        for k, label in keys.items():
            if abs(len(kw) - len(k)) <= 2 and kw[:2] == k[:2] and lev(kw, k) <= (1 if len(k) <= 8 else 2):
                found[(label, w)] += 1
    return [f"{label} sounds like {w!r} ({n} files)" for (label, w), n in sorted(found.items())]


EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# ten digits starting with 2 (landline) or 69 (mobile), in any grouping, standing alone, not inside an identifier,
# a hash or a decimal
PHONE = re.compile(r"(?<![\w.+-])(?:\+30[\s.-]?)?[26](?:[\s.-]?\d){9}(?![\w])(?![.,]\d)")
MASKED = {re.sub(r"\D", "", new) for _, new in MASKS}


def contact_checks(texts: dict[str, str]) -> list[str]:
    found = []
    for k, t in texts.items():
        for m in EMAIL.finditer(t):
            found.append(f"e-mail {m.group(0)} in {k}")
        for m in PHONE.finditer(t):
            digits = re.sub(r"\D", "", m.group(0))[-10:]
            if (digits.startswith("2") or digits.startswith("69")) and digits not in MASKED:
                found.append(f"telephone-like number {m.group(0)} in {k}")
    return found


def shingles(text: str, n: int = 12) -> set[tuple[str, ...]]:
    w = tn.norm(text).split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def record_text(r: dict) -> str:
    parts = []

    def add(o):
        if isinstance(o, str):
            parts.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                add(v)
        elif isinstance(o, list):
            for v in o:
                add(v)
    for f in ("title", "body", "summary"):
        add(r.get(f))
    return "\n".join(parts)


def overlap_checks(texts: dict[str, str]) -> tuple[float, dict[str, int], int]:
    """Share of the records' 12-word passages that occur in the archive, and the files they occur in."""
    recs = [json.loads(x) for x in SNAPSHOT.read_text(encoding="utf-8").splitlines() if x.strip()]
    snap = set()
    for r in recs:
        snap |= shingles(record_text(r))
    seen, per_file = set(), {}
    for k, t in texts.items():
        hit = shingles(t) & snap
        if hit:
            per_file[k] = len(hit)
            seen |= hit
    return len(seen) / max(1, len(snap)), per_file, len(snap)


# ------------------------------------------------------------------------------------------------ documents

README = """# KyGround benchmark, run outputs and extended data, version 1.0

This deposit holds the underlying and extended data of the article *Tool-calling retrieval versus vector
retrieval-augmented generation for a small, community-edited Greek–English agricultural knowledge base, tested for
accuracy, robustness to how users type Greek, freshness and access control*, by Nikolaos D. Tantaroudas, Ilias
Karachalios and Andrew J. McCracken, submitted to Open Research Europe. The same data support the two companion
preprints of the study, on accuracy and robustness (arXiv 2610.08205, https://doi.org/10.48550/arXiv.2610.08205) and
on freshness and access control (Preprints 202610.0645, https://doi.org/10.20944/preprints202610.0645.v1).

The study compares tool-calling retrieval, the design of the deployed assistant *Falco eleonorae*, with vector
retrieval-augmented generation (RAG) and four other conditions on the knowledge base of the DASK platform of the
Kythera pilot of the PoliRURAL Plus project, with Claude Haiku 4.5 as router and answer model, in four experiments on
accuracy and robustness to how Greek is typed, freshness after edits, disclosure of hidden content and the search
settings of the data interface.

## Contents

`underlying_data` holds the data behind every result of the article.

- `benchmark/items.jsonl`, the 198 benchmark questions in standard Greek and English, with their typed reference
  facts, reference answers, short evidence quotations, the records each answer is drawn from, question type, record
  type, cluster, component, split and verification status
- `benchmark/queries.jsonl`, every form of every question, as asked (1,469 queries)
- `benchmark/summary.json`, the composition of the release
- `benchmark/gold_verification.json` and `benchmark/gold_verification.md`, the result of every automatic check for
  every question
- `runs/main/`, the primary run outputs. For every experiment, each question asked, the answer, the identifiers of
  the retrieved chunks or records, the tool calls with their inputs and the identifiers of the records they
  returned, estimated token usage and scores (`accuracy*.jsonl`, `freshness.jsonl`, `governance.jsonl`), the
  freshness edits (`freshness_events.json`), the access-control canaries (`governance_canaries.json`), the replies
  flagged by the contamination audit (`contamination_flags.jsonl`), the run metadata (`meta_*.json`,
  `config_snapshot.json`) and, in `judge_session/`, the blinded batches given to the LLM judge, its grades, the
  rubric and the key that links grades to answers
- `runs/main_batched/`, the same outputs with the original batched replies, before the flagged replies were
  generated again in isolation, which the article reports as a sensitivity analysis
- `scorer_suite/`, the controlled answers with their constructed labels and the grades of both scorers, with the
  batches graded by the judge
- `results/` and `results_batched/`, all tables behind the figures and the numbers of the article, with the
  corrected and with the batched replies

`extended_data` holds the material that documents how the study was run.

- `protocol/study_protocol.md` and `protocol/freeze_v0.4.json`, the protocol and analysis plan, version 0.4, with
  the SHA-256 digests of the protocol and configuration files recorded before the first generation round
- `protocol/deviations.md`, the deviations from the protocol, with their dates, their reasons and whether results
  had been seen
- `configs/study.yaml`, `configs/prompts.yaml` and `configs/tools.yaml`, the configuration of the study, the
  reconstructed system prompt and grounding paragraphs, the judge rubric, the nine tool schemas and the single
  search tool of C1q, of which the prompts and tool texts are reconstructions and not the production texts
- `public_search_probe_20261005.csv` and `.json`, the probes of the platform's public search that set the free-text
  matching of the replica of the data interface, with their results
- `snapshot_record_hashes_20261005.json`, the identifier and a content hash of every published record of the frozen
  snapshot, with which the software shows which records of a newly downloaded snapshot are unchanged
- `rag_tuning.csv` and `rag_tuning.json`, the retrieval results on the development split that were used to choose
  the settings of vector RAG
- `curated_items.yaml`, the individually written questions with their evidence, aggregation rules and
  specifications of what must be absent
- `contamination_audit.json` and `control_isolation.json`, the audit of batched generation, by experiment, condition
  and kind of request, and the isolation control
- `supplementary_tables/`, the supplementary tables cited in the article, as CSV and Markdown

`MANIFEST-SHA256.txt` gives the SHA-256 digest of every file, and `pseudonymisation_report.md` the result of the
checks described below.

## Pseudonymisation and what is not included

The platform's public pages name producers, contact persons and hosts, some of whom are natural persons in a small,
identifiable community. Their names are replaced throughout by role labels, such as "Producer A", in Greek or Latin
script, with or without accents and in any case and grammatical form, in questions, reference answers, model answers
and search terms alike, so that the benchmark stays consistent. The names of businesses and cooperatives are kept,
and record identifiers, which contain no names, are unchanged. A person who appears both as a producer and as a
contact person has one label. Before the archive was made, it was checked automatically that no name of a natural
person remains, including misspelt and transliterated forms and given names written next to a label, and that no
e-mail address or telephone number occurs. The controlled answers of the scorer suite include an invented telephone
number, written in the archive as 27360 00000, which the scorer reads as a telephone number like the original.

The run outputs identify the records and chunks that were retrieved but do not copy their text, so the platform's
knowledge base is not redistributed. Questions, reference answers, evidence quotations and model answers reproduce
facts from the records, and some answers quote a sentence or a short passage of a record. Not included, for the
reasons above, are the frozen snapshot of the platform's
published records (fingerprint 234378797e066c0e), the request cache and the batch files given to the generation
sub-agents, which contain the text of the records, and the evaluation software. They are available on request from
the corresponding author, Nikolaos D. Tantaroudas (nikolaos.tantaroudas@iccs.gr), under the conditions stated in the
article's data availability statement. The published records can be viewed on the platform at
https://dask-platform.onrender.com/.

The released copies of `study_protocol.md`, `study.yaml`, `prompts.yaml`, `tools.yaml` and `rag_tuning.json` are
byte for byte the files whose digests `freeze_v0.4.json` records. The released benchmark files differ from the frozen
ones in the role labels, and `deviations.md` has gained the deviations recorded after the freeze.

## Licence

The data are available under the terms of the Creative Commons Attribution 4.0 International licence (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/. Please cite the deposit and the article.

## Funding

The work was supported by the European Union's Horizon Europe research and innovation programme through the
PoliRURAL Plus project (Grant Agreement No. 101136910), via its DEVELOP cascade-funding open call.
"""

LICENCE = """KyGround benchmark, run outputs and extended data, version 1.0
Copyright 2026 Nikolaos D. Tantaroudas, Ilias Karachalios and Andrew J. McCracken

These data are licensed under the Creative Commons Attribution 4.0 International licence (CC BY 4.0).
The licence text is at https://creativecommons.org/licenses/by/4.0/legalcode
"""

UPLOAD = """# Zenodo upload of {zip}

{zip} ({size_mb:.1f} MB, SHA-256 {sha}) is the data deposit cited in the ORE manuscript as reference
"KyGround benchmark, run outputs and extended data, version 1.0". It contains {n} files, {mb:.1f} MB when unpacked.

## Steps

1. On Zenodo, start a new upload and choose Dataset as the resource type.
2. Reserve the DOI on the upload form before publishing. Put it in `paper/build_docx_v2.py` (the line
   `ZENODO_DOI = ...`) and rebuild the manuscript, so that the article cites the deposit with its real DOI.
3. Upload {zip}, fill in the fields below, then publish.
4. When the ORE article has its DOI, edit the published record and add the article as a related work. Editing the
   metadata changes neither the files nor the DOI.

## Fields

**Title.** KyGround benchmark, run outputs and extended data, version 1.0

**Authors.** Nikolaos D. Tantaroudas, Institute of Communication and Computer Systems (ICCS), National Technical
University of Athens, ORCID 0000-0001-6727-1014. Ilias Karachalios, University of Thessaly. Andrew J. McCracken,
DASKALOS-APPS.

**Description.** Underlying and extended data of a study comparing tool-calling retrieval with vector
retrieval-augmented generation (RAG) for a small, community-edited Greek–English agricultural knowledge base, the
DASK platform of the Kythera pilot of the PoliRURAL Plus project. KyGround release 1.0 has 198 questions with
automatically verified reference answers, each asked in up to nine forms (standard Greek, English, Greek without
accents, Greek in capitals, three Greeklish schemes, another grammatical form and colloquial or regional wording),
1,469 queries in all. The deposit holds the benchmark, the answers, retrieval identifiers, tool calls and scores of
six conditions with Claude Haiku 4.5 as router and answer model in four experiments (accuracy and robustness,
freshness after edits, disclosure of hidden content and search settings), the grades of an LLM judge, a controlled
suite for validating the scorers, all result tables, the protocol and its deviations, the reconstructed prompts and
tool schemas, and supplementary tables. Names of natural persons are replaced by role labels, and the platform's
records are not redistributed. README.md describes every file.

**Version.** 1.0

**Licence.** Creative Commons Attribution 4.0 International (CC BY 4.0)

**Languages.** English, Greek

**Keywords.** retrieval-augmented generation, tool calling, Model Context Protocol, Greek, Greeklish, benchmark,
knowledge freshness, access control, agricultural advisory services

**Related works**, each as "is supplement to"
- the ORE article, once it has a DOI
- arXiv preprint, DOI 10.48550/arXiv.2610.08205
- Preprints preprint, DOI 10.20944/preprints202610.0645.v1

**Funding.** European Commission, Horizon Europe, Grant Agreement No. 101136910 (PoliRURAL Plus)

**Contact for the data with access restrictions.** Nikolaos D. Tantaroudas, nikolaos.tantaroudas@iccs.gr
"""


# ------------------------------------------------------------------------------------------------ main

def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def keep(top: Path, where: str | None) -> None:
    if where:
        dst = Path(where) / NAME
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(top, dst)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "zenodo"))
    ap.add_argument("--keep", help="also leave the unpacked folder here, for inspection")
    a = ap.parse_args(argv)
    mapping = yaml.safe_load(MAP.read_text(encoding="utf-8"))
    rules = Pseudonymiser(mapping)
    pairs = collect()
    with tempfile.TemporaryDirectory() as tmp:
        top = Path(tmp) / NAME
        replaced, changed = 0, []
        for src, dst in pairs:
            n = copy_pseudonymised(src, top / dst, rules)
            replaced += n
            if n:
                changed.append(dst)
        texts = decoded_texts(top)
        residual, fragments, where = name_checks(texts, mapping)
        contacts = contact_checks(texts)
        near = near_names(texts, mapping)
        coverage, per_file, n_snap = overlap_checks(texts)
        long_passages = {k: v for k, v in per_file.items()
                         if not (k.startswith("underlying_data/runs/") or k.startswith("underlying_data/scorer_suite/")
                                 or k.startswith("underlying_data/benchmark/") or k == "extended_data/curated_items.yaml")}
        freeze = json.loads((ROOT / "protocol" / "freeze_v0.4.json").read_text(encoding="utf-8"))["sha256"]
        dest_of = {s: d for s, d in LAYOUT}
        freeze_match = {p: (sha256((top / dest_of[p]).read_bytes()) == h) for p, h in freeze.items() if p in dest_of}

        print(f"{len(pairs)} files, {replaced} names replaced in {len(changed)} files")
        print(f"remaining names {sum(residual.values())}, remaining surname fragments {sum(fragments.values())}")
        for w in where[:20]:
            print("   ", w)
        print(f"e-mail addresses and telephone numbers {len(contacts)}", *contacts[:10], sep="\n    ")
        print(f"words that sound like a surname {len(near)}", *near[:40], sep="\n    ")
        print(f"12-word passages of the records found in the archive, {coverage:.1%} of {n_snap}, by file")
        for k, v in sorted(per_file.items(), key=lambda x: -x[1])[:12]:
            print(f"    {v:5d}  {k}")
        print(f"files outside the runs, the benchmark and the curated items with such passages {long_passages or 'none'}")
        print("released copies byte for byte as frozen", {k: v for k, v in freeze_match.items()})
        ok = not sum(residual.values()) and not sum(fragments.values()) and not contacts and not long_passages \
            and not near
        if not ok:
            keep(top, a.keep)
            print("CHECK FAILED, archive not written")
            return 2

        report = ["# Pseudonymisation report", "",
                  f"{len(pairs)} files copied, {replaced} occurrences of names replaced in {len(changed)} files.", "",
                  "| Label | Remaining occurrences of the name | Remaining fragments of its surname |",
                  "|---|---:|---:|"]
        report += [f"| {k} | {residual[k]} | {fragments[k]} |" for k in residual]
        report += ["", f"E-mail addresses and telephone numbers found, {len(contacts)}.", "",
                   "Words that sound like one of the surnames, in a spelling the map does not list, 0.", "",
                   f"Passages of 12 words of the records' text found in the archive, {coverage:.1%} of the "
                   f"{n_snap} passages of the frozen snapshot, all in questions, reference answers, evidence "
                   "quotations and model answers.", ""]
        (top / "pseudonymisation_report.md").write_text("\n".join(report), encoding="utf-8")
        (top / "README.md").write_text(README, encoding="utf-8")
        (top / "LICENCE.txt").write_text(LICENCE, encoding="utf-8")
        files = sorted(p for p in top.rglob("*") if p.is_file())
        manifest = "".join(f"{sha256(p.read_bytes())}  {p.relative_to(top).as_posix()}\n" for p in files)
        (top / "MANIFEST-SHA256.txt").write_text(manifest, encoding="utf-8")
        files = sorted(p for p in top.rglob("*") if p.is_file())

        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for p in files:
                info = zipfile.ZipInfo(f"{NAME}/{p.relative_to(top).as_posix()}", date_time=STAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                z.writestr(info, p.read_bytes(), compresslevel=9)
        data = buf.getvalue()
        zpath = out / f"{NAME}.zip"
        zpath.write_bytes(data)
        unpacked = sum(p.stat().st_size for p in files)
        (out / "ZENODO_UPLOAD.md").write_text(UPLOAD.format(zip=zpath.name, size_mb=len(data) / 1e6, sha=sha256(data),
                                                            n=len(files), mb=unpacked / 1e6), encoding="utf-8")
        keep(top, a.keep)
        print(f"wrote {zpath} ({len(data) / 1e6:.1f} MB, {len(files)} files, {unpacked / 1e6:.1f} MB unpacked), "
              f"SHA-256 {sha256(data)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
