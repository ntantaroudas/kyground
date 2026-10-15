"""End-to-end smoke test with the mock model (no network, no cost)."""

import json
from pathlib import Path

import pandas as pd

from kyground.bench.build import build
from kyground.bench.forms import FORMS, colloquial, realise
from kyground.experiments.accuracy import run_accuracy, score_accuracy
from kyground.experiments.factory import Env, load_config
from kyground.experiments.freshness import run_freshness
from kyground.experiments.governance import run_governance
from kyground.analysis.report import make_report
from kyground.analysis import stats as S
from kyground.schema import load_items, read_jsonl, write_jsonl
from test_kb import _kb


def test_forms():
    it = load_items_from_kb()[0]
    qs = realise(it)
    assert {q.form for q in qs} >= {"el", "en", "el_noacc", "el_caps", "grk_elot", "grk_phon", "grk_visual"}
    assert colloquial("Πού είναι το ελαιοτριβείο στα Κύθηρα;") == "Πού είναι το λιοτρίβι στο Τσιρίγο;"
    assert colloquial("Ο «Μάιος» είναι ωραίος") is None


def load_items_from_kb():
    from kyground.bench.templates import template_items
    return template_items(_kb().public())


def test_end_to_end_mock(tmp_path: Path):
    snap = tmp_path / "records.jsonl"
    write_jsonl(snap, _kb().all(include_gated=True))
    bdir = tmp_path / "bench"
    summary = build(list(_kb().all(include_gated=True)), None, bdir, dev_frac=0.0)
    assert summary["items"] > 0
    cfg = load_config(overrides={"paths": {"snapshot": str(snap), "benchmark_dir": str(bdir),
                                           "runs_dir": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache")}})
    env = Env.build(cfg, mock=True)
    run = tmp_path / "runs" / "r"
    run_accuracy(env, run, ["C0", "C1", "C2", "C4"], split="test", forms=["el", "en", "grk_phon"], workers=2)
    out = score_accuracy(run, bdir / "items.jsonl")
    df = pd.DataFrame(list(read_jsonl(out)))
    assert set(df["condition"]) == {"C0", "C1", "C2", "C4"} and (df["error"] == "").all()
    # labelled runs (E4 ablation, open model) must stay separate from the main conditions
    run_accuracy(env, run, ["C2"], split="test", forms=["el"], workers=2, search_mode="phrase_unaccent",
                 label="search=phrase_unaccent")
    df = pd.DataFrame(list(read_jsonl(score_accuracy(run, bdir / "items.jsonl"))))
    assert "C2@search=phrase_unaccent" in set(df["condition"])
    assert (df[df["label"] == ""]["condition"].isin(["C0", "C1", "C2", "C4"])).all()
    run_freshness(env, run, ["C2", "C1-batch"], n_events=2, workers=1)
    run_governance(env, run, ["C2", "C1-batch", "C1-raw"], n_per_kind=1, workers=1)
    # second scorer: judge on canonical Greek and English answers, and the controlled answer suite
    from kyground.scoring.judge import judge_rows
    from kyground.scoring.suite import build_suite, score_suite, write_suite
    items = {i.iid: i for i in load_items(bdir / "items.jsonl")}
    scored = [r for r in read_jsonl(run / "accuracy_scored.jsonl") if r["form"] in ("el", "en")]
    write_jsonl(run / "accuracy_judged.jsonl", judge_rows(env.clients["judge"], env.prompts, items, scored, workers=2))
    rows = score_suite(build_suite(list(items.values())), items)
    for r in rows:
        r["judge_correct"] = 1                      # the mock judge grades everything "correct"
    summ = write_suite(rows, run)
    assert summ["deterministic"]["overall"]["sensitivity"] == 1.0 and summ["judge"]["overall"]["specificity"] == 0
    key = make_report([run], bdir / "items.jsonl", tmp_path / "report", None, cfg)
    assert key["MOCK"] is True
    assert "ACC_C2@search=phrase_unaccent_el" in key and "H5_DIFF" in key
    for k in ("N_JUDGED", "AGREE_DETERMINISTIC_VS_JUDGE", "KAPPA_DETERMINISTIC_VS_JUDGE", "JDIFF_C2_C1",
              "SUITE_SENS_DET", "SUITE_SPEC_JUDGE"):
        assert k in key, k
    assert (tmp_path / "figures" / "fig_accuracy_by_form.png").exists()
    assert (tmp_path / "report" / "tables" / "ms_hypotheses.md").exists()


def test_stats():
    df = pd.DataFrame({"qid": list(range(20)) * 2, "rep": 0, "cluster": [i // 2 for i in range(20)] * 2,
                       "condition": ["A"] * 20 + ["B"] * 20,
                       "correct": [1] * 18 + [0] * 2 + [1] * 10 + [0] * 10})
    d = S.cluster_bootstrap_diff(df, "A", "B", n_boot=2000)
    assert abs(d["diff"] - 0.4) < 1e-9 and d["ci_lo"] > 0
    # only 4 informative clusters -> the smallest attainable exact p is 2/16
    assert abs(S.cluster_signflip_test(df, "A", "B") - 0.125) < 1e-9
    big = pd.DataFrame({"qid": list(range(60)) * 2, "rep": 0, "cluster": [i // 2 for i in range(60)] * 2,
                        "condition": ["A"] * 60 + ["B"] * 60,
                        "correct": [1] * 50 + [0] * 10 + [1] * 30 + [0] * 30})
    assert S.cluster_signflip_test(big, "A", "B") < 0.01
    assert S.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    lo, hi = S.clopper_pearson(0, 30)
    assert lo == 0 and abs(hi - 0.1157) < 1e-3
    assert abs(S.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) - 1.0) < 1e-12
