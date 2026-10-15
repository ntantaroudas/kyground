"""Statistical procedures (pre-specified in protocol/study_protocol.md, section 7).

Items are clustered: several questions (and all surface forms of a question)
derive from the same record. Inference therefore resamples or permutes whole
clusters:

- cluster bootstrap percentile intervals for accuracies and paired differences
- cluster-level sign-flip permutation test for paired differences (primary)
- GEE logistic regression with exchangeable working correlation (secondary)
- exact McNemar test on canonical-Greek items (sensitivity, ignores clustering)
- Holm correction across the pre-specified primary contrasts
- Clopper-Pearson and Wilson intervals for proportions (leakage, freshness)
- Cohen's kappa for judge-human and human-human agreement
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = sps.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return (lo, hi)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else sps.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else sps.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def _cluster_arrays(df: pd.DataFrame, value: str, cluster: str) -> tuple[np.ndarray, np.ndarray]:
    g = df.groupby(cluster)[value]
    return g.sum().to_numpy(dtype=float), g.count().to_numpy(dtype=float)


def cluster_bootstrap_mean(df: pd.DataFrame, value: str, cluster: str = "cluster", n_boot: int = 10000,
                           seed: int = 20261005, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of ``value`` with a cluster-bootstrap percentile CI."""
    if df.empty:
        return (float("nan"),) * 3
    sums, counts = _cluster_arrays(df, value, cluster)
    rng = np.random.default_rng(seed)
    k = len(sums)
    idx = rng.integers(0, k, size=(n_boot, k))
    boots = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    est = sums.sum() / counts.sum()
    return (float(est), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def paired_frame(df: pd.DataFrame, a: str, b: str, value: str, key: Sequence[str] = ("qid", "rep"),
                 cond_col: str = "condition", cluster: str = "cluster") -> pd.DataFrame:
    sub = df[df[cond_col].isin([a, b])]
    if sub.empty or not {a, b} <= set(sub[cond_col]):
        return pd.DataFrame()
    wide = sub.pivot_table(index=list(key) + [cluster], columns=cond_col, values=value, aggfunc="first")
    wide = wide.dropna(subset=[a, b]).reset_index()
    wide["diff"] = wide[a] - wide[b]
    return wide


def cluster_bootstrap_diff(df: pd.DataFrame, a: str, b: str, value: str = "correct",
                           key: Sequence[str] = ("qid", "rep"), cond_col: str = "condition",
                           cluster: str = "cluster", n_boot: int = 10000, seed: int = 20261005,
                           alpha: float = 0.05) -> dict:
    w = paired_frame(df, a, b, value, key, cond_col, cluster)
    if w.empty:
        return {"a": a, "b": b, "n_pairs": 0}
    sums, counts = _cluster_arrays(w, "diff", cluster)
    rng = np.random.default_rng(seed)
    k = len(sums)
    idx = rng.integers(0, k, size=(n_boot, k))
    boots = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    est = sums.sum() / counts.sum()
    return {"a": a, "b": b, "n_pairs": int(len(w)), "n_clusters": int(k), "acc_a": float(w[a].mean()),
            "acc_b": float(w[b].mean()), "diff": float(est), "ci_lo": float(np.quantile(boots, alpha / 2)),
            "ci_hi": float(np.quantile(boots, 1 - alpha / 2))}


def cluster_signflip_test(df: pd.DataFrame, a: str, b: str, value: str = "correct",
                          key: Sequence[str] = ("qid", "rep"), cond_col: str = "condition",
                          cluster: str = "cluster", n_perm: int = 20000, seed: int = 20261005) -> float:
    """Two-sided p-value for H0: no difference, flipping the sign of whole
    clusters' summed paired differences (exact when clusters <= 16)."""
    w = paired_frame(df, a, b, value, key, cond_col, cluster)
    if w.empty:
        return float("nan")
    sums, counts = _cluster_arrays(w, "diff", cluster)
    total = counts.sum()
    obs = abs(sums.sum() / total)
    sums = sums[sums != 0]          # clusters whose summed difference is zero cannot change the statistic
    k = len(sums)
    if k == 0:
        return 1.0
    if k <= 16:                     # exact enumeration over the informative clusters
        signs = np.array(np.meshgrid(*[[-1, 1]] * k)).T.reshape(-1, k)
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1, 1], size=(n_perm, k))
    null = np.abs((signs * sums).sum(axis=1) / total)
    hits = float(np.sum(null >= obs - 1e-12))
    if k <= 16:
        return hits / len(null)
    return (hits + 1) / (len(null) + 1)


def mcnemar_exact(df: pd.DataFrame, a: str, b: str, value: str = "correct", key: Sequence[str] = ("qid", "rep"),
                  cond_col: str = "condition", cluster: str = "cluster") -> dict:
    w = paired_frame(df, a, b, value, key, cond_col, cluster)
    if w.empty:
        return {"b_wins": 0, "a_wins": 0, "p_mcnemar": float("nan")}
    n01 = int(((w[a] == 0) & (w[b] == 1)).sum())
    n10 = int(((w[a] == 1) & (w[b] == 0)).sum())
    n = n01 + n10
    p = float(sps.binomtest(n10, n, 0.5).pvalue) if n else 1.0
    return {"b_wins": n01, "a_wins": n10, "p_mcnemar": p}


def holm(pvals: Sequence[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adj[i] = running
    return adj.tolist()


def gee_logit(df: pd.DataFrame, formula: str, groups: str = "cluster") -> pd.DataFrame:
    """GEE logistic regression (exchangeable correlation); odds ratios."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = df.copy()
    d[groups] = d[groups].astype("category").cat.codes
    model = smf.gee(formula, groups, d, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
    res = model.fit()
    ci = res.conf_int()
    out = pd.DataFrame({"coef": res.params, "se": res.bse, "p": res.pvalues, "ci_lo": ci[0], "ci_hi": ci[1]})
    out["OR"] = np.exp(out["coef"])
    out["OR_lo"] = np.exp(out["ci_lo"])
    out["OR_hi"] = np.exp(out["ci_hi"])
    return out


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    a = list(a)
    b = list(b)
    labels = sorted(set(a) | set(b), key=str)
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(lv) / n) * (b.count(lv) / n) for lv in labels)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def mcnemar_power(n_pairs: int, p10: float, p01: float, alpha: float = 0.05) -> float:
    """Approximate power of McNemar's test (Connor 1987)."""
    pd_ = p10 + p01
    delta = abs(p10 - p01)
    if pd_ <= 0 or delta <= 0:
        return 0.0
    za = sps.norm.ppf(1 - alpha / 2)
    num = math.sqrt(n_pairs) * delta - za * math.sqrt(pd_)
    den = math.sqrt(max(pd_ - delta ** 2, 1e-12))
    return float(sps.norm.cdf(num / den))


def simulate_power(cluster_sizes: Sequence[int], p10: float, p01: float, icc: float, alpha: float = 0.05,
                   n_sim: int = 2000, seed: int = 20261005) -> float:
    """Power of the two-sided cluster sign-flip test for a paired binary outcome, by simulation.

    Each item's paired difference is +1 (first condition right, second wrong) with probability ``p10``, -1 with
    ``p01`` and 0 otherwise; within a cluster, each item copies a shared cluster-level draw with probability
    sqrt(``icc``), which gives an intra-cluster correlation of ``icc`` between items' differences. The
    randomisation p-value uses the normal approximation to the sign-flip distribution, which is accurate for the
    numbers of informative clusters involved.
    """
    rng = np.random.default_rng(seed)
    sizes = np.asarray(cluster_sizes, dtype=int)
    k, n = len(sizes), int(sizes.sum())
    cl = np.repeat(np.arange(k), sizes)
    vals, probs = np.array([1, -1, 0]), np.array([p10, p01, 1 - p10 - p01])
    z = sps.norm.ppf(1 - alpha / 2)
    hits = 0
    for _ in range(n_sim):
        shared = rng.choice(vals, size=k, p=probs)[cl]
        own = rng.choice(vals, size=n, p=probs)
        d = np.where(rng.random(n) < np.sqrt(icc), shared, own)
        s = np.bincount(cl, weights=d, minlength=k)
        var = float((s ** 2).sum())
        if var > 0 and abs(s.sum()) / np.sqrt(var) > z:
            hits += 1
    return hits / n_sim


def design_effect(cluster_sizes: Sequence[int], icc: float) -> float:
    """Kish design effect for unequal cluster sizes: 1 + (sum m^2 / sum m - 1) * icc."""
    m = np.asarray(cluster_sizes, dtype=float)
    return float(1 + ((m ** 2).sum() / m.sum() - 1) * icc)


def effective_n(n_items: int, mean_cluster_size: float, icc: float) -> float:
    """Kish design effect: n / (1 + (m - 1) * icc)."""
    return n_items / (1 + (mean_cluster_size - 1) * icc)


def required_pairs(p10: float, p01: float, alpha: float = 0.05, power: float = 0.8) -> int:
    for n in range(10, 100000):
        if mcnemar_power(n, p10, p01, alpha) >= power:
            return n
    return -1


def summarize_props(groups: Iterable[tuple[str, int, int]], method: str = "wilson") -> pd.DataFrame:
    rows = []
    for name, k, n in groups:
        lo, hi = (clopper_pearson if method == "cp" else wilson_ci)(k, n)
        rows.append({"group": name, "k": k, "n": n, "rate": k / n if n else float("nan"), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)
