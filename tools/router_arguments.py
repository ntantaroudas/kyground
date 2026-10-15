"""Exploratory analysis (not pre-specified): the script of the free-text arguments that the router wrote in C2.

For every C2 answer of the main run (test split, answerable items) and every tool call, the free-text arguments
(query, term, crop_type, category) are classified as Greek or Latin script by their letters, and the share of calls
that returned no record is computed per surface form and script. Questions are also grouped by whether any of their
free-text arguments was written in Latin script, with the share answered correctly.

    PYTHONPATH=src python tools/router_arguments.py   # -> paper/results/tables/router_arguments.{csv,md,json}
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREE_TEXT = ("query", "term", "crop_type", "category")
FORMS = ["el", "en", "el_noacc", "el_caps", "grk_elot", "grk_phon", "grk_visual"]
FORM_NAMES = {"el": "Greek", "en": "English", "el_noacc": "Greek, no accents", "el_caps": "Greek, capitals",
              "grk_elot": "Greeklish (ELOT)", "grk_phon": "Greeklish (phonetic)", "grk_visual": "Greeklish (visual)"}
GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")
LATIN = re.compile(r"[A-Za-z]")


def script(text: str) -> str:
    g, lat = len(GREEK.findall(text)), len(LATIN.findall(text))
    if g == 0 and lat == 0:
        return "none"
    return "Greek" if g >= lat else "Latin"


def main() -> None:
    calls = defaultdict(Counter)       # form -> counts
    questions = defaultdict(Counter)
    for line in open(ROOT / "runs/main/accuracy_scored.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if (r["condition"] != "C2" or r["label"] != "" or r["split"] != "test" or r["qtype"] == "unanswerable"
                or r["form"] not in FORMS):
            continue
        f = r["form"]
        any_latin = False
        for tc in r["tool_calls"] or []:
            args = tc.get("args") or {}
            text = " ".join(str(args[k]) for k in FREE_TEXT if args.get(k))
            s = script(text) if text else "no free-text argument"
            empty = int(tc.get("n_results", 0) == 0)
            calls[f][f"calls|{s}"] += 1
            calls[f][f"empty|{s}"] += empty
            any_latin |= s == "Latin"
        g = "some argument in Latin script" if any_latin else "no argument in Latin script"
        questions[f][f"n|{g}"] += 1
        questions[f][f"correct|{g}"] += int(r["correct"])
    rows = []
    for f in FORMS:
        c, q = calls[f], questions[f]
        row = {"form": f, "form_name": FORM_NAMES[f]}
        for s in ("Greek", "Latin", "no free-text argument"):
            n = c[f"calls|{s}"]
            row[f"calls_{s}"] = n
            row[f"empty_{s}"] = round(c[f"empty|{s}"] / n, 3) if n else None
        free = row["calls_Greek"] + row["calls_Latin"]
        row["latin_share_of_free_text"] = round(row["calls_Latin"] / free, 3) if free else None
        for g, key in (("some argument in Latin script", "latin_q"), ("no argument in Latin script", "greek_q")):
            n = q[f"n|{g}"]
            row[f"{key}_n"] = n
            row[f"{key}_correct"] = round(q[f"correct|{g}"] / n, 3) if n else None
        rows.append(row)
    out = ROOT / "paper/results/tables"
    with open(out / "router_arguments.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def pct(x):
        return "" if x is None else f"{100 * x:.1f}"
    md = ["| Form | Tool calls with free-text arguments | Written in Latin script, % | No result, Greek-script "
          "arguments, % | No result, Latin-script arguments, % | Correct, questions with a Latin-script argument, % "
          "(n) | Correct, other questions, % (n) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['form_name']} | {r['calls_Greek'] + r['calls_Latin']} | {pct(r['latin_share_of_free_text'])} | "
                  f"{pct(r['empty_Greek'])} | {pct(r['empty_Latin'])} | {pct(r['latin_q_correct'])} ({r['latin_q_n']}) | "
                  f"{pct(r['greek_q_correct'])} ({r['greek_q_n']}) |")
    (out / "router_arguments.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "router_arguments.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
