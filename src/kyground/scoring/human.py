"""Optional blinded audit by the authors (secondary validation of the automatic scorers).

The scorers are validated without annotators (``scoring/suite.py`` and the agreement between the deterministic
scorer and the LLM judge, ``analysis/report.py``). If the authors wish to adjudicate, a sample of answers -
stratified by condition and deterministic outcome, or only answers on which the two scorers disagree
(``kyground human-export --disagreements``) - is exported to a workbook without condition labels, in random order.
One or two authors label each answer independently; agreement (Cohen's kappa) is computed between them, and between
the reference label and (a) the deterministic scorer and (b) the LLM judge.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from ..analysis.stats import cohen_kappa
from ..schema import Item
from ..scoring.judge import reference_text

LABELS = ("correct", "partially_correct", "incorrect", "abstained")


def sample_rows(df: pd.DataFrame, n: int = 150, seed: int = 20261005) -> pd.DataFrame:
    """Stratified sample (condition x deterministic outcome) of exactly ``n`` answers to canonical Greek and
    English test questions from the six main conditions (first repetition only; no ablation, live or open-model
    rows). Strata get equal shares; the remainder is filled at random from the rows not yet drawn."""
    main = df[df["form"].isin(["el", "en"]) & ~df["condition"].astype(str).str.contains("@")]
    if "split" in main:
        main = main[main["split"] == "test"]
    if "rep" in main:
        main = main[main["rep"].astype(int) == 0]
    main = main.copy()
    main["stratum"] = main["condition"].astype(str) + "|" + main["correct"].astype(str)
    per = max(1, n // max(1, main["stratum"].nunique()))
    parts = [g.sample(min(len(g), per), random_state=seed) for _, g in main.groupby("stratum")]
    out = pd.concat(parts) if parts else main.head(0)
    if len(out) < n:
        rest = main.drop(index=out.index)
        out = pd.concat([out, rest.sample(min(len(rest), n - len(out)), random_state=seed)])
    out = out.sample(frac=1.0, random_state=seed).head(n)
    out["ann_id"] = [hashlib.sha1(f"{r.condition}|{r.qid}|{r.rep}".encode()).hexdigest()[:10]
                     for r in out.itertuples()]
    return out


def export_annotation(sample: pd.DataFrame, items: dict[str, Item], path: Path, annotator: str) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = "Annotation"
    heads = ["ann_id", "Question", "Reference answer", "Answer to grade", "LABEL", "Unsupported specifics? (yes/no)",
             "Comment"]
    widths = [12, 50, 50, 70, 18, 16, 30]
    yellow = PatternFill("solid", fgColor="FFF2CC")
    for j, (h, w) in enumerate(zip(heads, widths), start=1):
        c = ws.cell(row=1, column=j, value=h)
        c.font = Font(name="Arial", bold=True)
        if j >= 5:
            c.fill = yellow
        ws.column_dimensions[c.column_letter].width = w
    for i, r in enumerate(sample.itertuples(), start=2):
        vals = [r.ann_id, r.question, reference_text(items[r.iid]), r.answer, "", "", ""]
        for j, v in enumerate(vals, start=1):
            c = ws.cell(row=i, column=j, value=v)
            c.font = Font(name="Arial", size=10)
            c.alignment = Alignment(wrap_text=True, vertical="top")
            if j >= 5:
                c.fill = yellow
    last = len(sample) + 1
    dv = DataValidation(type="list", formula1='"' + ",".join(LABELS) + '"', allow_blank=True)
    dv2 = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
    ws.add_data_validation(dv)
    ws.add_data_validation(dv2)
    dv.add(f"E2:E{last}")
    dv2.add(f"F2:F{last}")
    ws.freeze_panes = "B2"
    info = wb.create_sheet("Instructions")
    info["A1"] = (f"Annotator: {annotator}. Grade each answer against the reference. Labels: correct (all reference "
                  "facts, none contradicted), partially_correct (some facts, none contradicted), incorrect (a fact "
                  "missing or contradicted), abstained (says the information is unavailable). For items whose "
                  "reference says the information does not exist, 'abstained' is the desired outcome. Do not try to "
                  "guess which system produced an answer.")
    info["A1"].alignment = Alignment(wrap_text=True)
    info.column_dimensions["A"].width = 120
    wb.save(path)
    return path


def read_annotation(path: Path) -> pd.DataFrame:
    wb = load_workbook(path, data_only=True)
    ws = wb["Annotation"]
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0]:
            rows.append({"ann_id": r[0], "label": (r[4] or "").strip(), "unsupported": (r[5] or "").strip()})
    return pd.DataFrame(rows)


def human_correct(label: str, qtype: str) -> int:
    """Binary 'correct' from a 4-class label; abstention is the correct behaviour on unanswerable items."""
    return int(label == "correct" or (qtype == "unanswerable" and label == "abstained"))


def agreement(sample: pd.DataFrame, ann_a: pd.DataFrame, ann_b: pd.DataFrame | None = None,
              adjudicated: pd.DataFrame | None = None) -> dict:
    """Cohen's kappa between annotators (4 classes) and between the human reference and the automatic scorers.

    The human reference is the adjudicated label where available, otherwise annotator A's label.
    """
    m = sample.merge(ann_a.rename(columns={"label": "label_a"})[["ann_id", "label_a"]], on="ann_id")
    m = m[m["label_a"].isin(LABELS)]
    if ann_b is not None:
        m = m.merge(ann_b.rename(columns={"label": "label_b"})[["ann_id", "label_b"]], on="ann_id", how="left")
    if adjudicated is not None:
        m = m.merge(adjudicated.rename(columns={"label": "label_ref"})[["ann_id", "label_ref"]], on="ann_id",
                    how="left")
        m["label_ref"] = m["label_ref"].where(m["label_ref"].isin(LABELS), m["label_a"])
    else:
        m["label_ref"] = m["label_a"]
    out: dict = {"n": int(len(m)), "reference": "adjudicated" if adjudicated is not None else "annotator_A"}
    if m.empty:
        return out
    ref_bin = [human_correct(lab, q) for lab, q in zip(m["label_ref"], m["qtype"])]
    out["human_correct_rate"] = float(sum(ref_bin) / len(ref_bin))
    out["kappa_deterministic_vs_human"] = cohen_kappa(m["correct"].astype(int).tolist(), ref_bin)
    if ann_b is not None and m["label_b"].isin(LABELS).any():
        mb = m[m["label_b"].isin(LABELS)]
        out["n_double"] = int(len(mb))
        out["kappa_A_vs_B"] = cohen_kappa(mb["label_a"].tolist(), mb["label_b"].tolist())
        out["kappa_A_vs_B_binary"] = cohen_kappa([human_correct(x, q) for x, q in zip(mb["label_a"], mb["qtype"])],
                                                 [human_correct(x, q) for x, q in zip(mb["label_b"], mb["qtype"])])
    if "judge_correctness" in m and m["judge_correctness"].isin(LABELS).any():
        mj = m[m["judge_correctness"].isin(LABELS)]
        out["n_judged"] = int(len(mj))
        out["kappa_judge_vs_human"] = cohen_kappa(mj["judge_correctness"].tolist(), mj["label_ref"].tolist())
        out["kappa_judge_vs_human_binary"] = cohen_kappa(
            [human_correct(x, q) for x, q in zip(mj["judge_correctness"], mj["qtype"])],
            [human_correct(x, q) for x, q in zip(mj["label_ref"], mj["qtype"])])
    return out
