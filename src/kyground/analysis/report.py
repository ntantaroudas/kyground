"""Tables, figures, key numbers and the filled manuscript, from run outputs.

Usage: ``kyground report --runs runs/<id> [runs/<id2> ...]``
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..schema import Item, load_items, read_jsonl
from . import figures as F
from . import stats as S

PRIMARY = [("C2", "C1"), ("C2", "C1q"), ("C2", "C0")]
SECONDARY = [("C1", "C0"), ("C1q", "C1"), ("C3", "C2"), ("C4", "C2")]


def _load(runs: list[Path], name: str) -> pd.DataFrame:
    frames = [pd.DataFrame(list(read_jsonl(r / name))) for r in runs if (r / name).exists()]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_(no data)_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                vals.append("" if np.isnan(v) else format(v, floatfmt))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def cost_usd(usage_rows: list[dict], pricing: dict) -> float:
    total = 0.0
    for u in usage_rows:
        for model, m in (u or {}).items():
            p = None
            for key, val in pricing.items():
                if model.startswith(key):
                    p = val
            if not p:
                continue
            inp = m.get("input_tokens", 0) - m.get("cached_input_tokens", 0)
            total += inp * p.get("input", 0) / 1e6
            total += m.get("cached_input_tokens", 0) * p.get("cached_input", p.get("input", 0)) / 1e6
            total += m.get("output_tokens", 0) * p.get("output", 0) / 1e6
    return total


def accuracy_tables(df: pd.DataFrame, n_boot: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ans = df[df["qtype"] != "unanswerable"].copy()
    ans["condition"] = ans["condition"].astype(str)
    conds = [c for c in F.COND_ORDER if c in set(ans["condition"])] + sorted(
        c for c in set(ans["condition"]) if c not in F.COND_ORDER)
    # by condition x form
    rows = []
    for c in conds:
        for f in F.FORM_LABEL:
            sub = ans[(ans["condition"] == c) & (ans["form"] == f)]
            if sub.empty:
                continue
            est, lo, hi = S.cluster_bootstrap_mean(sub, "correct", n_boot=n_boot, seed=seed)
            rec, _, _ = S.cluster_bootstrap_mean(sub, "recall", n_boot=200, seed=seed)
            rows.append({"condition": c, "form": f, "n": len(sub), "acc": est, "lo": lo, "hi": hi,
                         "recall": rec})
    out["by_form"] = pd.DataFrame(rows)
    # primary / secondary contrasts on canonical Greek
    con = []
    el = ans[ans["form"] == "el"]
    for (a, b) in PRIMARY + SECONDARY:
        if a not in conds or b not in conds:
            continue
        d = S.cluster_bootstrap_diff(el, a, b, "correct", n_boot=n_boot, seed=seed)
        if not d.get("n_pairs"):
            continue
        d["p_signflip"] = S.cluster_signflip_test(el, a, b, "correct", seed=seed)
        d.update(S.mcnemar_exact(el, a, b, "correct"))
        d["family"] = "primary" if (a, b) in PRIMARY else "secondary"
        con.append(d)
    cdf = pd.DataFrame(con)
    if not cdf.empty:
        cdf["p_holm"] = np.nan
        prim = cdf["family"] == "primary"
        if prim.any():
            cdf.loc[prim, "p_holm"] = S.holm(cdf.loc[prim, "p_signflip"].tolist())
    out["contrasts"] = cdf
    # robustness gaps relative to canonical Greek (difference-in-differences vs C1)
    gaps = []
    for f in [x for x in F.FORM_LABEL if x not in ("el",)]:
        for c in conds:
            a = ans[(ans["condition"] == c) & (ans["form"].isin(["el", f]))].copy()
            if a.empty or f not in set(a["form"]):
                continue
            a["base_qid"] = a["qid"].str.split("#").str[0]
            w = a.pivot_table(index=["base_qid", "rep", "cluster"], columns="form", values="correct",
                              aggfunc="first").dropna().reset_index()
            if w.empty or f not in w:
                continue
            w["gap"] = w[f] - w["el"]
            est, lo, hi = S.cluster_bootstrap_mean(w, "gap", n_boot=n_boot, seed=seed)
            gaps.append({"condition": c, "form": f, "n_pairs": len(w), "gap": est, "lo": lo, "hi": hi})
    out["robustness_gaps"] = pd.DataFrame(gaps)
    # difference-in-differences: (C2 gap) - (C1 gap) per form, paired by item, cluster bootstrap
    did = []
    pair = ans[ans["condition"].isin(["C1", "C2"])]
    for f in [x for x in F.FORM_LABEL if x != "el"]:
        sub = pair[pair["form"].isin(["el", f])]
        if sub.empty or f not in set(sub["form"]):
            continue
        w = sub.pivot_table(index=["iid", "rep", "cluster"], columns=["condition", "form"], values="correct",
                            aggfunc="first")
        need = [("C2", f), ("C2", "el"), ("C1", f), ("C1", "el")]
        if not all(c in w.columns for c in need):
            continue
        w = w[need].dropna()
        if w.empty:
            continue
        d = pd.DataFrame({"cluster": w.index.get_level_values("cluster"),
                          "did": (w[("C2", f)] - w[("C2", "el")] - (w[("C1", f)] - w[("C1", "el")])).to_numpy()})
        est, lo, hi = S.cluster_bootstrap_mean(d, "did", n_boot=n_boot, seed=seed)
        did.append({"form": f, "n_items": len(d), "did": est, "lo": lo, "hi": hi})
    out["did"] = pd.DataFrame(did)
    # exploratory: questions that quote an exact title in «» (literal matching is easy) vs the rest
    if "question" in el:
        qq = el.assign(quotes_title=el["question"].astype(str).str.contains("«"))
        out["by_quote"] = qq.groupby(["condition", "quotes_title"]).agg(n=("correct", "size"),
                                                                        acc=("correct", "mean")).reset_index()
    # by question type (canonical Greek)
    qt = []
    for c in conds:
        for q in sorted(set(el["qtype"])):
            sub = el[(el["condition"] == c) & (el["qtype"] == q)]
            if len(sub):
                qt.append({"condition": c, "qtype": q, "n": len(sub), "acc": sub["correct"].mean()})
    out["by_qtype"] = pd.DataFrame(qt).pivot_table(index="qtype", columns="condition", values="acc").reset_index() \
        if qt else pd.DataFrame()
    # diagnostics (el + en canonical)
    dg = []
    can = ans[ans["form"].isin(["el", "en"])]
    for c in conds:
        sub = can[can["condition"] == c]
        if sub.empty:
            continue
        n = len(sub)
        corr = sub["correct"] == 1
        abst = (sub["abstained"] == 1) & ~corr
        gic = sub["gold_in_context"].fillna(0).astype(int) == 1
        dg.append({"condition": c, "n": n, "correct": corr.mean(), "abstained": abst.mean(),
                   "wrong_gold_in_context": (~corr & ~abst & gic).mean(),
                   "wrong_gold_missing": (~corr & ~abst & ~gic).mean(),
                   "gold_retrieved": gic.mean()})
    out["diagnostics"] = pd.DataFrame(dg)
    # tool behaviour
    if "n_tool_calls" in can:
        tb = can[can["condition"].isin(["C2", "C3"])].groupby("condition").agg(
            mean_tool_calls=("n_tool_calls", "mean"), any_tool_call=("n_tool_calls", lambda x: (x > 0).mean()),
            empty_tool_result=("tool_empty", "mean")).reset_index()
        out["tool_behaviour"] = tb
    return out


def error_table(df: pd.DataFrame) -> pd.DataFrame:
    """API errors, empty answers and routing flags by condition (failures count as incorrect)."""
    rows = []
    for c in sorted(set(df["condition"])):
        s = df[df["condition"] == c]
        flags = s["flags"] if "flags" in s else pd.Series([[]] * len(s))
        flat = [f for fl in flags for f in (fl or [])]
        rows.append({"condition": c, "answers": len(s), "api_error": float((s["error"].fillna("") != "").mean()),
                     "empty_answer": float(s["empty"].mean()) if "empty" in s else np.nan,
                     "router_no_tool": flat.count("router_no_tool") / max(1, len(s)),
                     "router_error": flat.count("router_error") / max(1, len(s)),
                     "rewrite_no_query": flat.count("rewrite_no_query") / max(1, len(s))})
    return pd.DataFrame(rows)


def safety_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    can = df[df["form"].isin(["el", "en"])]
    ua = can[can["qtype"] == "unanswerable"]
    rows = []
    for c in sorted(set(ua["condition"])):
        s = ua[ua["condition"] == c]
        k, n = int(s["hallucinated"].sum()), len(s)
        lo, hi = S.wilson_ci(k, n)
        rows.append({"condition": c, "n": n, "hallucination_rate": k / n if n else np.nan, "lo": lo, "hi": hi})
    out["unanswerable"] = pd.DataFrame(rows)
    por = can[can["qtype"] == "price_on_request"]
    rows = []
    for c in sorted(set(por["condition"])):
        s = por[por["condition"] == c]
        k, n = int(s["fabricated_price"].sum()), len(s)
        lo, hi = S.clopper_pearson(k, n)
        rows.append({"condition": c, "n": n, "fabricated_price_rate": k / n if n else np.nan, "lo": lo, "hi": hi})
    out["fabricated_price"] = pd.DataFrame(rows)
    return out


def _usage_sum(u: dict | None, field: str) -> float:
    return float(sum((v or {}).get(field, 0) for v in (u or {}).values()))


def cost_table(df: pd.DataFrame, pricing: dict) -> pd.DataFrame:
    """Per condition: model calls and tokens per question (with session generation the tokens are estimates of what
    each request would cost as a separate call; no price applies), and cost and latency when calls were priced."""
    rows = []
    for c in sorted(set(df["condition"])):
        s = df[df["condition"] == c]
        usd = cost_usd(s["usage"].tolist(), pricing)
        lat = s["latency_s"].astype(float)
        calls = [_usage_sum(u, "calls") for u in s["usage"]]
        tin = [_usage_sum(u, "input_tokens") for u in s["usage"]]
        tout = [_usage_sum(u, "output_tokens") for u in s["usage"]]
        rows.append({"condition": c, "n": len(s), "cost_per_1k_usd": 1000 * usd / max(1, len(s)),
                     "latency_median": float(lat.median()), "latency_p90": float(lat.quantile(0.9)),
                     "calls_per_q": float(np.mean(calls)), "in_tokens_per_q": float(np.mean(tin)),
                     "in_tokens_p90": float(np.quantile(tin, 0.9)), "out_tokens_per_q": float(np.mean(tout))})
    return pd.DataFrame(rows)


def freshness_table(fr: pd.DataFrame) -> pd.DataFrame:
    """Share of answers giving the new value. For the same-conversation phases the rate is computed among the
    conversations whose first answer gave the old value (the case the experiment is about); ``rate_all`` keeps
    every conversation."""
    rows = []
    pre = fr[fr["phase"] == "pre_edit"]
    for (v, ph), s in fr.groupby(["variant", "phase"], sort=False):
        if ph == "pre_edit":
            n_pre = len(s)
            rows.append({"variant": v, "condition": s["condition"].iloc[0], "policy": s["policy"].iloc[0],
                         "phase": ph, "n": n_pre, "rate": float((s["state"] == "correct_pre").mean()),
                         "lo": np.nan, "hi": np.nan, "stale": np.nan, "mixed": np.nan})
            continue
        k_all, n_all = int((s["state"] == "updated").sum()), len(s)
        sub = s
        if ph.startswith("in_thread") and "pre_gave_old" in s:
            sub = s[s["pre_gave_old"].fillna(0).astype(int) == 1]
        k, n = int((sub["state"] == "updated").sum()), len(sub)
        lo, hi = S.wilson_ci(k, n)
        rows.append({"variant": v, "condition": s["condition"].iloc[0], "policy": s["policy"].iloc[0], "phase": ph,
                     "n": n, "rate": k / n if n else np.nan, "lo": lo, "hi": hi,
                     "rate_all": k_all / n_all if n_all else np.nan, "n_all": n_all,
                     "stale": float((sub["state"] == "stale").mean()) if n else np.nan,
                     "mixed": float((sub["state"] == "mixed").mean()) if n else np.nan})
    return pd.DataFrame(rows)


def leakage_table(gv: pd.DataFrame) -> pd.DataFrame:
    """Leak rate per variant and phase. ``rate_cond_visible`` restricts the
    revocation phases to canaries the system did surface while public (G0),
    i.e. the effectiveness of revocation where it matters."""
    rows = []
    vis = gv[gv["phase"] == "G0_visible"].set_index(["variant", "canary"])["leak"].to_dict()
    for (v, ph), s in gv.groupby(["variant", "phase"], sort=False):
        k, n = int(s["leak"].sum()), len(s)
        lo, hi = S.clopper_pearson(k, n)
        row = {"variant": v, "condition": s["condition"].iloc[0], "policy": s["policy"].iloc[0], "phase": ph,
               "n": n, "leaks": k, "rate": k / n if n else np.nan, "lo": lo, "hi": hi,
               "context_exposed": float(s["context_exposed"].mean()) if "context_exposed" in s else np.nan}
        if ph.startswith(("G2", "G3")):
            sv = s[[bool(vis.get((v, c), 0)) for c in s["canary"]]]
            row["n_visible"] = len(sv)
            row["rate_cond_visible"] = float(sv["leak"].mean()) if len(sv) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_test(df: pd.DataFrame, a: str, b: str, value: str, n_boot: int, seed: int) -> dict | None:
    """Paired comparison of a binary outcome between two variants (cluster bootstrap CI, cluster sign-flip
    p-value, exact McNemar) plus an unpaired Fisher exact test as sensitivity."""
    if not {a, b} <= set(df["condition"]):
        return None
    d = S.cluster_bootstrap_diff(df, a, b, value, n_boot=n_boot, seed=seed)
    if not d.get("n_pairs"):
        return None
    d["p_signflip"] = S.cluster_signflip_test(df, a, b, value, seed=seed)
    d.update(S.mcnemar_exact(df, a, b, value))
    sa, sb = df[df["condition"] == a][value], df[df["condition"] == b][value]
    ka, kb = int(sa.sum()), int(sb.sum())
    d["p_fisher"] = float(sps.fisher_exact([[ka, len(sa) - ka], [kb, len(sb) - kb]])[1])
    return d


def hypothesis_tests(fr: pd.DataFrame, gv: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """H4 (freshness) and H5 (governance), paired by edit and by canary probe; clusters are the edited
    record (H4) and the canary (H5). Decision rules are in protocol section 7."""
    rows = []
    if not fr.empty:
        f = fr[fr["phase"] == "fresh_before_refresh"].copy()
        f = f.assign(condition=f["variant"], qid=f["eid"].astype(str) + "|" + f["lang"].astype(str), rep=0,
                     cluster=f["rid"].astype(str), updated=(f["state"] == "updated").astype(int))
        d = _paired_test(f, "C2", "C1-batch", "updated", n_boot, seed)
        if d:
            d.update({"hypothesis": "H4", "phase": "fresh_before_refresh", "outcome": "updated",
                      "supported": bool(d["diff"] > 0 and d["p_signflip"] < 0.05)})
            rows.append(d)
    if not gv.empty:
        g = gv[gv["phase"] == "G2_revoked_before_refresh"].copy()
        g = g.assign(condition=g["variant"], qid=g["canary"].astype(str) + "|" + g["lang"].astype(str), rep=0,
                     cluster=g["canary"].astype(str), leak=g["leak"].astype(int))
        d = _paired_test(g, "C2", "C1-batch", "leak", n_boot, seed)
        if d:
            d.update({"hypothesis": "H5", "phase": "G2_revoked_before_refresh", "outcome": "leak",
                      "supported": bool(d["ci_hi"] <= 0)})
            rows.append(d)
    cols = ["hypothesis", "phase", "outcome", "a", "b", "n_pairs", "n_clusters", "acc_a", "acc_b", "diff", "ci_lo",
            "ci_hi", "p_signflip", "a_wins", "b_wins", "p_mcnemar", "p_fisher", "supported"]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame()


def sensitivity_table(acc: pd.DataFrame, items: list[Item], n_boot: int, seed: int) -> pd.DataFrame:
    """Primary contrasts re-estimated (i) with record-connected components as clusters, (ii) on editor-verified
    items only and (iii) with development items added (first repetition), next to the primary analysis."""
    status = {i.iid: i.status for i in items}
    comp = {i.iid: i.component or i.cluster for i in items}
    el = acc[(acc["qtype"] != "unanswerable") & (acc["form"] == "el") & (~acc["condition"].str.contains("@"))]
    el = el.assign(component=el["iid"].map(comp).fillna(el["cluster"]))
    sets = {"primary (test)": el[el["split"] == "test"],
            "editor-verified items only": el[(el["split"] == "test") & (el["iid"].map(status) == "verified")],
            "test + development (rep 0)": el[(el["split"] == "test") | ((el["split"] == "dev") & (el["rep"] == 0))]}
    rows = []
    for name, df in sets.items():
        if name != "primary (test)" and len(df) == len(sets["primary (test)"]):
            continue  # nothing verified yet / no development rows: identical to the primary analysis
        for a, b in PRIMARY:
            d = S.cluster_bootstrap_diff(df, a, b, "correct", n_boot=n_boot, seed=seed)
            if not d.get("n_pairs"):
                continue
            d["p_signflip"] = S.cluster_signflip_test(df, a, b, "correct", seed=seed)
            d.update(S.mcnemar_exact(df, a, b, "correct"))
            rows.append({"analysis": name, **d})
    prim = sets["primary (test)"]
    for a, b in PRIMARY:  # record-connected components as the unit of resampling
        d = S.cluster_bootstrap_diff(prim, a, b, "correct", cluster="component", n_boot=n_boot, seed=seed)
        if d.get("n_pairs"):
            d["p_signflip"] = S.cluster_signflip_test(prim, a, b, "correct", cluster="component", seed=seed)
            d.update(S.mcnemar_exact(prim, a, b, "correct"))
            rows.append({"analysis": "clusters = record-connected components", **d})
    return pd.DataFrame(rows)


def gee_table(acc: pd.DataFrame) -> pd.DataFrame:
    """GEE logistic regression of correctness on condition x form for the primary pair (C2 vs C1), answerable
    test items, exchangeable correlation within clusters; odds ratios relative to C1 and canonical Greek."""
    sub = acc[(acc["split"] == "test") & (acc["qtype"] != "unanswerable") & acc["condition"].isin(["C1", "C2"])]
    if sub.empty or sub["form"].nunique() < 2 or sub["condition"].nunique() < 2:
        return pd.DataFrame()
    import warnings
    sub = sub.assign(correct=sub["correct"].astype(int))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = S.gee_logit(sub, "correct ~ C(condition, Treatment('C1')) * C(form, Treatment('el'))")
        except Exception as exc:  # noqa: BLE001 - e.g. perfect separation on small or mock data
            return pd.DataFrame([{"term": "model not estimable", "note": repr(exc)[:200]}])
    out = out.reset_index().rename(columns={"index": "term"})[["term", "OR", "OR_lo", "OR_hi", "p"]]
    bad = ~np.isfinite(out[["OR", "OR_lo", "OR_hi"]].astype(float)).all(axis=1) | (out["OR"] == 0)
    out["note"] = np.where(bad, "not estimable (separation: no variation in this cell)", "")
    if caught:
        out.attrs["warnings"] = sorted({str(w.message)[:120] for w in caught})
    return out


def rep_agreement(acc: pd.DataFrame) -> pd.DataFrame:
    """Run-to-run variation, from the queries that were run more than once (development split)."""
    rows = []
    n_rep = acc.groupby(["condition", "qid"])["rep"].nunique()
    multi = n_rep[n_rep > 1]
    for c in sorted({c for c, _ in multi.index}):
        qids = [q for cc, q in multi.index if cc == c]
        s = acc[(acc["condition"] == c) & (acc["qid"].isin(qids))]
        agree = s.groupby("qid")["correct"].nunique().eq(1).mean()
        per_rep = s.groupby("rep")["correct"].mean()
        rows.append({"condition": c, "n_queries": len(qids), "reps": int(s["rep"].nunique()),
                     "all_reps_agree": float(agree),
                     "acc_sd_across_reps": float(per_rep.std(ddof=1)) if len(per_rep) > 1 else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# Presentation tables for the manuscript (string cells, readable headers); the raw tables stay in tables/*.csv
# ---------------------------------------------------------------------------------------------------------------

def _sgn(x: float, nd: int = 1) -> str:
    """Signed number with a typographic minus, e.g. '−8.6' or '+2.0'."""
    return f"{x:+.{nd}f}".replace("-", "−")


def _p_eq(p) -> str:
    """'= 0.066' or '< 0.001', for sentences of the form 'p {{P_...}}'."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "= n/a"
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def _p1(x) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def _ci_cell(est, lo, hi) -> str:
    return f"{_p1(est)} ({_p1(lo)}–{_p1(hi)})" if _p1(est) else ""


def _pp_cell(est, lo, hi) -> str:
    if est is None or (isinstance(est, float) and np.isnan(est)):
        return ""
    return f"{_sgn(100 * est)} ({_sgn(100 * lo)} to {_sgn(100 * hi)})"


def _pfmt(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _cond_label(c: str) -> str:
    base, _, lab = c.partition("@")
    return F.COND_LABEL.get(base, base) + (f" [{lab}]" if lab else "")


def manuscript_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    bf = tables.get("acc_by_form")
    if bf is not None and not bf.empty:
        def grid(sub: pd.DataFrame) -> pd.DataFrame:
            forms = [f for f in F.FORM_LABEL if f in set(sub["form"])]
            conds = [c for c in F.COND_ORDER if c in set(sub["condition"])] + sorted(
                c for c in set(sub["condition"]) if c not in F.COND_ORDER)
            rows = []
            for c in conds:
                r = {"Condition": _cond_label(c)}
                for f in forms:
                    m = sub[(sub["condition"] == c) & (sub["form"] == f)]
                    r[F.FORM_LABEL[f]] = _ci_cell(m["acc"].iloc[0], m["lo"].iloc[0], m["hi"].iloc[0]) if len(m) else ""
                rows.append(r)
            return pd.DataFrame(rows)
        base = bf[~bf["condition"].str.contains("@")]
        out["ms_accuracy"] = grid(base)
        abl = bf[bf["condition"].str.contains("@search=")]
        if not abl.empty:
            out["ms_ablation"] = grid(abl)
        oth = bf[bf["condition"].str.contains("@") & ~bf["condition"].str.contains("@search=")]
        if not oth.empty:
            out["ms_exploratory"] = grid(oth)
    con = tables.get("acc_contrasts")
    if con is not None and not con.empty:
        out["ms_contrasts"] = pd.DataFrame([{
            "Contrast": f"{r['a']} − {r['b']}", "Family": r["family"],
            "Pairs (clusters)": f"{int(r['n_pairs'])} ({int(r['n_clusters'])})",
            f"Accuracy, % (first)": _p1(r["acc_a"]), "Accuracy, % (second)": _p1(r["acc_b"]),
            "Difference, pp (95% CI)": _pp_cell(r["diff"], r["ci_lo"], r["ci_hi"]),
            "p (sign-flip)": _pfmt(r["p_signflip"]), "p (Holm)": _pfmt(r.get("p_holm")),
            "p (McNemar)": _pfmt(r["p_mcnemar"])} for _, r in con.iterrows()])
    gaps = tables.get("acc_robustness_gaps")
    if gaps is not None and not gaps.empty:
        g = gaps[~gaps["condition"].str.contains("@")]
        forms = [f for f in F.FORM_LABEL if f in set(g["form"])]
        rows = []
        for c in [c for c in F.COND_ORDER if c in set(g["condition"])]:
            r = {"Condition": _cond_label(c)}
            for f in forms:
                m = g[(g["condition"] == c) & (g["form"] == f)]
                r[F.FORM_LABEL[f]] = _pp_cell(m["gap"].iloc[0], m["lo"].iloc[0], m["hi"].iloc[0]) if len(m) else ""
            rows.append(r)
        did = tables.get("acc_did")
        if did is not None and not did.empty:
            r = {"Condition": "C2 − C1 (difference-in-differences)"}
            for f in forms:
                m = did[did["form"] == f]
                r[F.FORM_LABEL[f]] = _pp_cell(m["did"].iloc[0], m["lo"].iloc[0], m["hi"].iloc[0]) if len(m) else ""
            rows.append(r)
        out["ms_gaps"] = pd.DataFrame(rows)
    su, fp = tables.get("safety_unanswerable"), tables.get("safety_fabricated_price")
    if su is not None and not su.empty:
        su = su[~su["condition"].str.contains("@")]
        fp = fp[~fp["condition"].str.contains("@")] if fp is not None and not fp.empty else fp
        rows = []
        for c in [c for c in F.COND_ORDER if c in set(su["condition"])] + sorted(
                c for c in set(su["condition"]) if c not in F.COND_ORDER):
            u = su[su["condition"] == c].iloc[0]
            r = {"Condition": _cond_label(c),
                 "Unanswerable: answered instead of abstaining, % (95% CI)":
                     f"{_ci_cell(u['hallucination_rate'], u['lo'], u['hi'])}; n = {int(u['n'])}"}
            if fp is not None and not fp.empty and c in set(fp["condition"]):
                q = fp[fp["condition"] == c].iloc[0]
                r["Price on request: number given, % (95% CI)"] = \
                    f"{_ci_cell(q['fabricated_price_rate'], q['lo'], q['hi'])}; n = {int(q['n'])}"
            rows.append(r)
        out["ms_safety"] = pd.DataFrame(rows)
    fr = tables.get("freshness")
    if fr is not None and not fr.empty:
        phases = [("in_thread_reask", "Same conversation, re-asked†"), ("in_thread_challenge",
                  "Same conversation, challenged†"), ("fresh_before_refresh", "New conversation, before re-index"),
                  ("fresh_after_refresh", "New conversation, after re-index")]
        rows = []
        for v in F._order_variants(fr["variant"]):
            pre = fr[(fr["variant"] == v) & (fr["phase"] == "pre_edit")]
            r = {"Variant": v, "First answer gave the old value": (
                f"{int(round(pre['rate'].iloc[0] * pre['n'].iloc[0]))}/{int(pre['n'].iloc[0])}" if len(pre) else "")}
            for ph, lab in phases:
                m = fr[(fr["variant"] == v) & (fr["phase"] == ph)]
                r[lab] = (f"{_ci_cell(m['rate'].iloc[0], m['lo'].iloc[0], m['hi'].iloc[0])}; n = {int(m['n'].iloc[0])}"
                          if len(m) and m["n"].iloc[0] else "")
            rows.append(r)
        out["ms_freshness"] = pd.DataFrame(rows)
    lk = tables.get("leakage")
    if lk is not None and not lk.empty:
        phases = [("G1_static", "Gated from the start"), ("G0_visible", "Visible (positive control)"),
                  ("G2_revoked_before_refresh", "Revoked, new conversation, before re-index"),
                  ("G2_revoked_after_refresh", "Revoked, new conversation, after re-index"),
                  ("G3_in_thread", "Revoked, same conversation")]
        rows = []
        for v in F._order_variants(lk["variant"]):
            r = {"Variant": v}
            for ph, lab in phases:
                m = lk[(lk["variant"] == v) & (lk["phase"] == ph)]
                if len(m):
                    x = m.iloc[0]
                    cell = f"{int(x['leaks'])}/{int(x['n'])}"
                    if not np.isnan(x.get("context_exposed", np.nan)):
                        cell += f" (exposed {_p1(x['context_exposed'])}%)"
                    r[lab] = cell
                else:
                    r[lab] = ""
            rows.append(r)
        out["ms_leakage"] = pd.DataFrame(rows)
    if lk is not None and not lk.empty and "rate_cond_visible" in lk:
        rows = []
        for v in F._order_variants(lk["variant"]):
            r = {"Variant": v}
            for ph, lab in (("G2_revoked_before_refresh", "New conversation, before re-index"),
                            ("G2_revoked_after_refresh", "New conversation, after re-index"),
                            ("G3_in_thread", "Same conversation")):
                m = lk[(lk["variant"] == v) & (lk["phase"] == ph)]
                if len(m) and m["n_visible"].iloc[0] == m["n_visible"].iloc[0] and m["n_visible"].iloc[0]:
                    nv = int(m["n_visible"].iloc[0])
                    r[lab] = f"{int(round(m['rate_cond_visible'].iloc[0] * nv))}/{nv}"
                else:
                    r[lab] = ""
            if any(r[k] for k in r if k != "Variant"):
                rows.append(r)
        if rows:
            out["ms_leakage_cond"] = pd.DataFrame(rows)
    cl = tables.get("cost_latency")
    if cl is not None and not cl.empty:
        cl = cl[~cl["condition"].str.contains("@")]
        if (cl["cost_per_1k_usd"] > 0).any():
            out["ms_cost"] = pd.DataFrame([{
                "Condition": _cond_label(r["condition"]), "Answers": int(r["n"]),
                "Cost per 1,000 questions, USD": f"{r['cost_per_1k_usd']:.2f}",
                "Median latency, s": f"{r['latency_median']:.1f}",
                "90th percentile latency, s": f"{r['latency_p90']:.1f}"} for _, r in cl.iterrows()])
        else:       # session generation: no price and no meaningful latency; estimated tokens instead
            out["ms_cost"] = pd.DataFrame([{
                "Condition": _cond_label(r["condition"]), "Answers": int(r["n"]),
                "Model calls per question": f"{r['calls_per_q']:.2f}",
                "Input tokens per question": f"{r['in_tokens_per_q']:,.0f}",
                "Input tokens, 90th percentile": f"{r['in_tokens_p90']:,.0f}",
                "Output tokens per question": f"{r['out_tokens_per_q']:,.0f}"} for _, r in cl.iterrows()])
    rows = []
    if con is not None and not con.empty:
        for h, (a, b) in zip(("H1", "H2", "H3"), PRIMARY):
            m = con[(con["a"] == a) & (con["b"] == b)]
            if not len(m):
                continue
            r = m.iloc[0]
            ok = (r["p_holm"] < 0.05) and (h != "H3" or r["diff"] > 0)
            rows.append({"Hypothesis": h, "Comparison": f"accuracy, {a} − {b} (canonical Greek)",
                         "Estimate, pp (95% CI)": _pp_cell(r["diff"], r["ci_lo"], r["ci_hi"]),
                         "p": f"{_pfmt(r['p_holm'])} (Holm)", "Decision": "supported" if ok else "not supported"})
    ht = tables.get("hypothesis_tests")
    if ht is not None and not ht.empty:
        what = {"H4": "updated answers, new conversation before re-index", "H5": "disclosure after revocation, "
                "new conversation before re-index"}
        for _, r in ht.iterrows():
            rows.append({"Hypothesis": r["hypothesis"], "Comparison": f"{what[r['hypothesis']]}, {r['a']} − {r['b']}",
                         "Estimate, pp (95% CI)": _pp_cell(r["diff"], r["ci_lo"], r["ci_hi"]),
                         "p": f"{_pfmt(r['p_signflip'])} (sign-flip); {_pfmt(r['p_fisher'])} (Fisher)",
                         "Decision": "supported" if r["supported"] else "not supported"})
    if rows:
        out["ms_hypotheses"] = pd.DataFrame(rows)
    return out


def judge_tables(jdf: pd.DataFrame, n_boot: int, seed: int) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Agreement between the deterministic scorer and the LLM judge on the experiments' own answers (canonical
    Greek and English test questions of the main conditions), and the primary contrasts graded by the judge."""
    from ..scoring.suite import judge_binary
    tables: dict[str, pd.DataFrame] = {}
    key: dict[str, Any] = {}
    if jdf.empty or "judge_correctness" not in jdf:
        return tables, key
    j = jdf[jdf["form"].isin(["el", "en"]) & ~jdf["condition"].astype(str).str.contains("@")].copy()
    j = j[j["judge_correctness"].fillna("").astype(str) != ""]
    if j.empty:
        return tables, key
    j["judge_correct"] = [judge_binary(lab, q) for lab, q in zip(j["judge_correctness"], j["qtype"])]
    det, jud = j["correct"].astype(int).tolist(), j["judge_correct"].astype(int).tolist()
    key["N_JUDGED"] = f"{len(j):,}"
    key["AGREE_DETERMINISTIC_VS_JUDGE"] = _pct(float(np.mean([x == y for x, y in zip(det, jud)])))
    key["KAPPA_DETERMINISTIC_VS_JUDGE"] = f"{S.cohen_kappa(det, jud):.2f}"
    rows = []
    for (c, f), g in j.groupby(["condition", "form"]):
        ans = g[g["qtype"] != "unanswerable"]
        rows.append({"condition": c, "form": f, "n": len(g), "acc_deterministic": ans["correct"].mean(),
                     "acc_judge": ans["judge_correct"].mean(),
                     "agreement": float((g["correct"].astype(int) == g["judge_correct"]).mean()),
                     "kappa": S.cohen_kappa(g["correct"].astype(int).tolist(), g["judge_correct"].tolist())})
    tables["acc_judge"] = pd.DataFrame(rows)
    dis = j[j["correct"].astype(int) != j["judge_correct"]]
    if not dis.empty:
        tables["judge_disagreements"] = dis.assign(
            direction=np.where(dis["correct"].astype(int) == 1, "scorer correct, judge not", "judge correct, scorer not")
        ).groupby(["qtype", "direction"]).size().rename("answers").reset_index()
    el = j[(j["form"] == "el") & (j["qtype"] != "unanswerable")]
    con = []
    for a, b in PRIMARY:
        d = S.cluster_bootstrap_diff(el, a, b, "judge_correct", n_boot=n_boot, seed=seed)
        if not d.get("n_pairs"):
            continue
        d["p_signflip"] = S.cluster_signflip_test(el, a, b, "judge_correct", seed=seed)
        con.append(d)
        key[f"JDIFF_{a}_{b}"] = f"{_sgn(100 * d['diff'])} pp"
        key[f"JDIFF_{a}_{b}_CI"] = f"{_sgn(100 * d['ci_lo'])} to {_sgn(100 * d['ci_hi'])} pp"
    if con:
        cdf = pd.DataFrame(con)
        cdf["p_holm"] = S.holm(cdf["p_signflip"].tolist())
        tables["acc_judge_contrasts"] = cdf
    return tables, key


def live_comparison(acc: pd.DataFrame, items: list[Item], diff: dict | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """C2 against the deployed interface versus C2 with the replica, on the same answerable test queries, leaving
    out items whose gold records changed on the platform after the snapshot (``kyground snapshot-diff``)."""
    from ..sources.snapshot_diff import affected_items
    key: dict[str, Any] = {}
    if acc.empty or not acc["condition"].astype(str).eq("C2@live").any():
        return pd.DataFrame(), key
    excl = affected_items(diff, items) if diff else set()
    ans = acc[acc["qtype"] != "unanswerable"]
    live_all = ans[ans["condition"].astype(str) == "C2@live"]
    lv = live_all[~live_all["iid"].isin(excl)]
    rp = ans[ans["condition"].astype(str) == "C2"]
    common = set(lv["qid"]) & set(rp["qid"])
    lv, rp = lv[lv["qid"].isin(common)], rp[rp["qid"].isin(common)]
    key["N_LIVE_ITEMS"] = int(lv["iid"].nunique())
    key["N_LIVE_EXCLUDED"] = len(excl & set(live_all["iid"]))
    rows = []
    for f in sorted(set(lv["form"])):
        a, b = lv[lv["form"] == f], rp[rp["form"] == f]
        m = a.merge(b, on=["qid", "rep"], suffixes=("_live", "_replica"))
        rows.append({"form": f, "n": len(m), "acc_live": a["correct"].mean(), "acc_replica": b["correct"].mean(),
                     "same_outcome": float((m["correct_live"] == m["correct_replica"]).mean()) if len(m) else np.nan})
        key[f"LIVE_ACC_{f}"] = _pct(a["correct"].mean())
        key[f"LIVE_REPLICA_ACC_{f}"] = _pct(b["correct"].mean())
    return pd.DataFrame(rows), key


def suite_keys(summary: dict) -> dict[str, Any]:
    """Key numbers of the controlled answer suite (``kyground scorer-suite``)."""
    key: dict[str, Any] = {}
    det = (summary.get("deterministic") or {}).get("overall") or {}
    if det:
        key["SUITE_N"] = f"{det['n_pos'] + det['n_neg']:,}"
        key["SUITE_N_POS"] = f"{det['n_pos']:,}"
        key["SUITE_N_NEG"] = f"{det['n_neg']:,}"
        key["SUITE_SENS_DET"] = _pct(det["sensitivity"])
        key["SUITE_SPEC_DET"] = _pct(det["specificity"])
    jd = (summary.get("judge") or {}).get("overall") or {}
    if jd:
        key["SUITE_N_JUDGED"] = f"{jd['n_pos'] + jd['n_neg']:,}"
        key["SUITE_SENS_JUDGE"] = _pct(jd["sensitivity"])
        key["SUITE_SPEC_JUDGE"] = _pct(jd["specificity"])
    return key


def _is_mock(runs: list[Path]) -> bool:
    return any(json.loads(m.read_text(encoding="utf-8")).get("mock", False) for r in runs for m in r.glob("meta_*.json"))


def make_report(runs: list[Path], items_path: Path, out_dir: Path, manuscript: Path | None, cfg: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tdir, fdir = out_dir / "tables", out_dir.parent / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    n_boot = int(cfg.get("analysis", {}).get("n_boot", 10000))
    seed = int(cfg.get("seed", 20261005))
    items = load_items(items_path)
    statuses = cfg.get("accuracy", {}).get("item_statuses") or []
    if statuses and not _is_mock(runs):
        items = [i for i in items if i.status in set(statuses)]
    test = [i for i in items if i.split == "test"]
    key: dict[str, Any] = {"N_ITEMS": len(items), "N_CLUSTERS": len({i.cluster for i in items}),
                           "N_TEST_ITEMS": len(test),
                           "N_TEST_ANSWERABLE": sum(1 for i in test if i.qtype != "unanswerable"),
                           "N_TEST_CLUSTERS": len({i.cluster for i in test if i.qtype != "unanswerable"}),
                           "N_QUOTED": sum(1 for i in test if i.qtype != "unanswerable" and "«" in i.question.get("el", ""))}
    tables: dict[str, pd.DataFrame] = {}
    acc_all = _load(runs, "accuracy_scored.jsonl")
    mock = False
    fps: set[str] = set()
    for r in runs:
        for m in r.glob("meta_*.json"):
            meta = json.loads(m.read_text(encoding="utf-8"))
            mock = mock or meta.get("mock", False)
            if meta.get("kb_fingerprint"):
                fps.add(str(meta["kb_fingerprint"]))
    key["MOCK"] = mock
    key["KB_FINGERPRINT"] = ", ".join(sorted(fps)) if fps else "n/a"
    main_meta = {}
    for r in runs:
        if (r / "meta_accuracy.json").exists():
            main_meta = json.loads((r / "meta_accuracy.json").read_text(encoding="utf-8"))
            break
    rep_cfg = main_meta.get("replica") or cfg.get("replica", {})
    key["REPLICA_SEARCH_MODE"] = rep_cfg.get("search_mode", "n/a")
    if main_meta.get("kb_records"):
        key["KB_RECORDS"] = main_meta["kb_records"]
        key["KB_CHARS"] = f"{int(main_meta['kb_chars']):,}"
    acc = acc_all
    if not acc_all.empty:
        split = {i.iid: i.split for i in items}
        acc_all = acc_all.assign(split=acc_all["iid"].map(split).fillna("test"))
        acc = acc_all[acc_all["split"] == "test"]
    if not acc.empty:
        key["N_QUERIES_RUN"] = f"{int(acc['qid'].nunique()):,}"
        at = accuracy_tables(acc, n_boot, seed)
        tables.update({f"acc_{k}": v for k, v in at.items()})
        tables.update({f"safety_{k}": v for k, v in safety_tables(acc).items()})
        tables["cost_latency"] = cost_table(acc[acc["form"].isin(["el", "en"])], cfg.get("pricing", {}))
        tables["acc_errors"] = error_table(acc)
        for _, r in at["did"].iterrows() if "did" in at and not at["did"].empty else []:
            key[f"DID_{r['form']}"] = f"{_sgn(100 * r['did'])} pp"
            key[f"DID_{r['form']}_CI"] = f"{_sgn(100 * r['lo'])} to {_sgn(100 * r['hi'])} pp"
        tables["acc_sensitivity"] = sensitivity_table(acc_all, items, n_boot, seed)
        tables["acc_gee"] = gee_table(acc)
        ra = rep_agreement(acc_all)
        if not ra.empty:
            tables["acc_rep_agreement"] = ra
        bf = at["by_form"]
        for _, r in bf.iterrows():
            key[f"ACC_{r['condition']}_{r['form']}"] = _pct(r["acc"])
            key[f"ACC_{r['condition']}_{r['form']}_CI"] = f"{_pct(r['lo'])}–{_pct(r['hi'])}"
        for _, r in at["contrasts"].iterrows():
            key[f"DIFF_{r['a']}_{r['b']}"] = f"{_sgn(100 * r['diff'])} pp"
            key[f"DIFF_{r['a']}_{r['b']}_CI"] = f"{_sgn(100 * r['ci_lo'])} to {_sgn(100 * r['ci_hi'])} pp"
            ph = r.get("p_holm", np.nan)
            key[f"P_{r['a']}_{r['b']}"] = _p_eq(r["p_signflip"] if ph is None or np.isnan(ph) else ph)
        for _, r in at["robustness_gaps"].iterrows():
            key[f"GAP_{r['condition']}_{r['form']}"] = f"{_sgn(100 * r['gap'])} pp"
            key[f"GAP_{r['condition']}_{r['form']}_CI"] = f"{_sgn(100 * r['lo'])} to {_sgn(100 * r['hi'])} pp"
        sf = safety_tables(acc)
        for _, r in sf["unanswerable"].iterrows():
            key[f"HALLU_{r['condition']}"] = f"{int(round(r['hallucination_rate'] * r['n']))}/{r['n']}"
        for _, r in sf["fabricated_price"].iterrows():
            key[f"FABPRICE_{r['condition']}"] = f"{int(round(r['fabricated_price_rate'] * r['n']))}/{r['n']}"
        for _, r in tables["cost_latency"].iterrows():
            key[f"COST1K_{r['condition']}"] = f"${r['cost_per_1k_usd']:.2f}"
            key[f"LATMED_{r['condition']}"] = f"{r['latency_median']:.1f} s"
            key[f"TOKIN_{r['condition']}"] = f"{r['in_tokens_per_q']:,.0f}"
            key[f"CALLS_{r['condition']}"] = f"{r['calls_per_q']:.2f}"
        diff = next((json.loads((r / "snapshot_diff.json").read_text(encoding="utf-8")) for r in runs
                     if (r / "snapshot_diff.json").exists()), None)
        lt, lk = live_comparison(acc, items, diff)
        if not lt.empty:
            tables["live_vs_replica"] = lt
        key.update(lk)
        sel = acc[acc["form"] == "el"]
        sel = sel[sel["condition"].isin(F.COND_ORDER)]
        bfo = bf[bf["condition"].isin(F.COND_ORDER)]
        if not bfo.empty:
            F.accuracy_by_form(bfo, fdir)
        dg = at["diagnostics"]
        if not dg.empty:
            F.stacked_outcomes(dg[dg["condition"].isin(F.COND_ORDER)], fdir)
        cl = tables["cost_latency"]
        if not cl.empty and cl["condition"].isin(F.COND_ORDER).any():
            if (cl["cost_per_1k_usd"] > 0).any():
                F.cost_latency(cl[cl["condition"].isin(F.COND_ORDER)], fdir)
            else:
                F.tokens_calls(cl[cl["condition"].isin(F.COND_ORDER)], fdir)
    fr = _load(runs, "freshness.jsonl")
    if not fr.empty:
        ft = freshness_table(fr)
        tables["freshness"] = ft
        F.freshness_bars(ft, fdir)
        for _, r in ft.iterrows():
            key[f"FRESH_{r['variant']}_{r['phase']}"] = _pct(r["rate"])
            if r["phase"].startswith("in_thread"):
                key[f"FRESH_{r['variant']}_{r['phase']}_N"] = int(r["n"])
                key[f"FRESH_{r['variant']}_{r['phase']}_ALL"] = _pct(r.get("rate_all", np.nan))
    gv = _load(runs, "governance.jsonl")
    if not gv.empty:
        lt = leakage_table(gv)
        tables["leakage"] = lt
        F.leakage_bars(lt[lt["phase"] != "G0_visible"], fdir)
        for _, r in lt.iterrows():
            key[f"LEAK_{r['variant']}_{r['phase']}"] = f"{r['leaks']}/{r['n']}"
            nv = r.get("n_visible", np.nan)
            if nv == nv and nv:
                key[f"LEAKCOND_{r['variant']}_{r['phase']}"] = f"{int(round(r['rate_cond_visible'] * nv))}/{int(nv)}"
    jt, jk = judge_tables(_load(runs, "accuracy_judged.jsonl").pipe(
        lambda d: d.assign(split=d["iid"].map({i.iid: i.split for i in items})).query("split == 'test'")
        if not d.empty else d), n_boot, seed)
    tables.update(jt)
    key.update(jk)
    for r in runs:  # validation outputs written by `scorer-suite`, `human-agreement` and `conformance`
        if (r / "scorer_suite.json").exists():
            key.update(suite_keys(json.loads((r / "scorer_suite.json").read_text(encoding="utf-8"))))
        if (r / "agreement.json").exists():
            ag = json.loads((r / "agreement.json").read_text(encoding="utf-8"))
            for k, v in ag.items():
                if k.startswith("kappa"):
                    key[k.upper()] = f"{v:.2f}"
            key["N_ANNOTATED"] = ag.get("n")
        if (r / "conformance.json").exists():
            cf = json.loads((r / "conformance.json").read_text(encoding="utf-8"))["summary"]
            key["CONF_N_CALLS"] = cf.get("n_calls")
            key["CONF_N_DISTINCT"] = cf.get("n_distinct_calls", cf.get("n_calls"))
            key["CONF_MEAN_JACCARD"] = cf.get("mean_jaccard")
            key["CONF_EXACT"] = _pct(cf["exact_rate"]) if cf.get("exact_rate") is not None else "n/a"
    ht = hypothesis_tests(fr, gv, n_boot, seed)
    if not ht.empty:
        tables["hypothesis_tests"] = ht
        for _, r in ht.iterrows():
            h = r["hypothesis"]
            key[f"{h}_DIFF"] = f"{_sgn(100 * r['diff'])} pp"
            key[f"{h}_CI"] = f"{_sgn(100 * r['ci_lo'])} to {_sgn(100 * r['ci_hi'])} pp"
            key[f"{h}_P"] = _p_eq(r["p_signflip"])
            key[f"{h}_P_FISHER"] = _p_eq(r["p_fisher"])
            key[f"{h}_SUPPORTED"] = "supported" if r["supported"] else "not supported"
    tables.update(manuscript_tables(tables))
    # write tables
    md_parts = ["# KyGround results", "", f"Mock run: **{mock}** (mock numbers are NOT results).", ""]
    for name, t in tables.items():
        t.to_csv(tdir / f"{name}.csv", index=False)
        (tdir / f"{name}.md").write_text(_md(t), encoding="utf-8")
        md_parts += [f"## {name}", "", _md(t), ""]
    (out_dir / "results.md").write_text("\n".join(md_parts), encoding="utf-8")
    (out_dir / "key_numbers.json").write_text(json.dumps(key, ensure_ascii=False, indent=1, default=str),
                                              encoding="utf-8")
    if manuscript and manuscript.exists():
        fill_manuscript(manuscript, key, tables, manuscript.with_name(manuscript.stem + ".filled.md"))
    return key


def fill_manuscript(src: Path, key: dict, tables: dict[str, pd.DataFrame], dst: Path) -> Path:
    txt = src.read_text(encoding="utf-8")

    def tab(m: re.Match) -> str:
        t = tables.get(m.group(1))
        return _md(t) if t is not None else m.group(0)

    txt = re.sub(r"\{\{TABLE:([a-z_]+)\}\}", tab, txt)
    txt = re.sub(r"\{\{([A-Za-z0-9_@.=-]+)\}\}", lambda m: str(key.get(m.group(1), m.group(0))), txt)
    if key.get("MOCK"):
        txt = "> **MOCK RUN — numbers below are from the deterministic mock model, not results.**\n\n" + txt
    dst.write_text(txt, encoding="utf-8")
    return dst
