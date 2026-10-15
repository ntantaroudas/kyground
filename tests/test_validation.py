import pandas as pd

from kyground.scoring.human import agreement, human_correct


def test_human_correct_maps_abstention_on_unanswerable():
    assert human_correct("correct", "factoid") == 1
    assert human_correct("abstained", "factoid") == 0
    assert human_correct("abstained", "unanswerable") == 1
    assert human_correct("partially_correct", "list") == 0


def test_agreement_uses_adjudicated_reference_and_judge():
    sample = pd.DataFrame({
        "ann_id": ["a", "b", "c", "d"],
        "qtype": ["factoid", "list", "unanswerable", "numeric"],
        "correct": [1, 0, 1, 0],
        "judge_correctness": ["correct", "partially_correct", "abstained", "incorrect"],
    })
    ann_a = pd.DataFrame({"ann_id": ["a", "b", "c", "d"], "label": ["correct", "incorrect", "abstained", "correct"]})
    ann_b = pd.DataFrame({"ann_id": ["a", "b", "c", "d"], "label": ["correct", "partially_correct", "abstained",
                                                                    "incorrect"]})
    adj = pd.DataFrame({"ann_id": ["d"], "label": ["incorrect"]})
    rep = agreement(sample, ann_a, ann_b, adj)
    assert rep["n"] == 4 and rep["reference"] == "adjudicated"
    # with the adjudicated label for d, the human reference equals the deterministic scorer exactly
    assert rep["kappa_deterministic_vs_human"] == 1.0
    assert rep["kappa_judge_vs_human_binary"] == 1.0
    assert rep["n_double"] == 4 and rep["kappa_A_vs_B"] < 1.0


def test_power_simulation_matches_design_effect_for_equal_clusters():
    from kyground.analysis import stats as S
    # 40 clusters of 4: with ICC 0 the simulated power equals the unclustered McNemar approximation
    sizes = [4] * 40
    p0 = S.simulate_power(sizes, 0.175, 0.05, 0.0, n_sim=1500)
    assert abs(p0 - S.mcnemar_power(160, 0.175, 0.05)) < 0.06
    # positive ICC lowers power, close to the design-effect approximation
    p3 = S.simulate_power(sizes, 0.175, 0.05, 0.3, n_sim=1500)
    approx = S.mcnemar_power(int(160 / S.design_effect(sizes, 0.3)), 0.175, 0.05)
    assert p3 < p0 and abs(p3 - approx) < 0.08
