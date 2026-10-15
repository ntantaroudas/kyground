"""Publication figures (Matplotlib, 300 dpi PNG + LZW TIFF, as ORE requires).

Colour assignment is fixed per condition across all figures (validated
categorical palette: adjacent-pair CVD separation >= 8 Delta E), and identity
is never carried by colour alone: every condition is also labelled on an
axis or marked with its own marker shape, and every figure has a table twin.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COND_ORDER = ["C2", "C1", "C1q", "C3", "C4", "C0"]
COND_COLOR = {"C2": "#2a78d6", "C1": "#eb6834", "C1q": "#1baf7a", "C3": "#eda100", "C4": "#e87ba4", "C0": "#008300"}
COND_MARKER = {"C2": "o", "C1": "s", "C1q": "D", "C3": "^", "C4": "v", "C0": "P"}
COND_LABEL = {"C0": "C0 closed book", "C1": "C1 vector RAG", "C1q": "C1q RAG + rewrite", "C2": "C2 tool retrieval",
              "C3": "C3 hybrid", "C4": "C4 full context"}
FORM_LABEL = {"el": "Greek", "en": "English", "el_noacc": "Greek, no accents", "el_caps": "Greek, capitals",
              "grk_elot": "Greeklish (ELOT)", "grk_phon": "Greeklish (phonetic)", "grk_visual": "Greeklish (visual)",
              "el_infl": "Greek, other inflection", "el_colloq": "Greek, colloquial"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
VARIANT_ORDER = ["C2", "C1-upsert", "C1-batch", "C1-append", "C1-raw", "C1q-batch", "C3-batch", "C4", "C0"]


def _order_variants(vs) -> list[str]:
    vs = list(dict.fromkeys(vs))
    return [v for v in VARIANT_ORDER if v in vs] + [v for v in vs if v not in VARIANT_ORDER]


def _pct_axis(ax, quarters: bool = False) -> None:
    ax.set_xlim(-0.02, 1.02)
    ticks = [0, 0.25, 0.5, 0.75, 1.0] if quarters else [0, 0.5, 1.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0" if t == 0 else f"{t:.0%}" for t in ticks])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
})


def save(fig, out: Path, name: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.png", out / f"{name}.tif"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], dpi=300, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return paths


def accuracy_by_form(tab: pd.DataFrame, out: Path, name: str = "fig_accuracy_by_form") -> list[Path]:
    """tab columns: condition, form, acc, lo, hi"""
    conds = [c for c in COND_ORDER if c in set(tab["condition"])]
    forms = [f for f in FORM_LABEL if f in set(tab["form"])]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(forms) + 1.4))
    step = 0.8 / max(1, len(conds))
    for j, c in enumerate(conds):
        sub = tab[tab["condition"] == c].set_index("form")
        ys, xs, lo, hi = [], [], [], []
        for i, f in enumerate(forms):
            if f in sub.index:
                ys.append(i - 0.4 + step * (j + 0.5))
                xs.append(sub.loc[f, "acc"])
                # percentile intervals need not contain the estimate exactly; never draw negative error bars
                lo.append(max(0.0, sub.loc[f, "acc"] - sub.loc[f, "lo"]))
                hi.append(max(0.0, sub.loc[f, "hi"] - sub.loc[f, "acc"]))
        ax.errorbar(xs, ys, xerr=[lo, hi], fmt=COND_MARKER[c], color=COND_COLOR[c], ms=4.5, lw=1.2, capsize=0,
                    mec="white", mew=0.6, label=COND_LABEL[c])
    ax.set_yticks(range(len(forms)))
    ax.set_yticklabels([FORM_LABEL[f] for f in forms])
    ax.invert_yaxis()
    _pct_axis(ax, quarters=True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Answers fully correct (95% cluster-bootstrap CI)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.02 * len(forms) / 7), ncol=3, fontsize=8)
    return save(fig, out, name)


def stacked_outcomes(tab: pd.DataFrame, out: Path, name: str = "fig_error_decomposition") -> list[Path]:
    """tab: condition, correct, wrong_gold_in_context, wrong_gold_missing, abstained (shares)"""
    conds = [c for c in COND_ORDER if c in set(tab["condition"])]
    # outcomes use a neutral lightness ramp (not the condition hues, which identify conditions in every figure)
    parts = [("correct", "Correct", "#3d3c38", "white"),
             ("wrong_gold_in_context", "Wrong, evidence was retrieved", "#7a7974", "white"),
             ("wrong_gold_missing", "Wrong, evidence not retrieved", "#b4b3aa", INK),
             ("abstained", "Abstained", "#e6e5de", INK)]
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(conds) + 1.2))
    t = tab.set_index("condition")
    for i, c in enumerate(conds):
        left = 0.0
        for key, label, col, txt in parts:
            v = float(t.loc[c, key]) if c in t.index else 0.0
            ax.barh(i, v, left=left, color=col, edgecolor="white", linewidth=2, height=0.62,
                    label=label if i == 0 else None)
            if v >= 0.06:
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", color=txt, fontsize=7.5)
            left += v
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([COND_LABEL[c] for c in conds])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_axisbelow(True)
    ax.set_xlabel("Share of answers (Greek and English canonical questions)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
    return save(fig, out, name)


_PHASES = [("in_thread_reask", "Same conversation,\nre-asked"), ("in_thread_challenge", "Same conversation,\nchallenged"),
           ("fresh_before_refresh", "New conversation,\nbefore re-index"),
           ("fresh_after_refresh", "New conversation,\nafter re-index")]


def _bar_panel(ax, sub: pd.DataFrame, variants: list[str], label_fn) -> None:
    """Horizontal bars with CI whiskers and a direct label per bar; 'not run' where a variant has no data."""
    for i, v in enumerate(variants):
        if v not in sub.index:
            ax.text(0.02, i, "not run", va="center", ha="left", fontsize=6.5, color=MUTED)
            continue
        r = sub.loc[v]
        ax.barh(i, r["rate"], color=COND_COLOR.get(r["condition"], MUTED), hatch=_HATCH.get(r["policy"], ""),
                edgecolor="white", linewidth=0.8, height=0.62)
        ax.errorbar(r["rate"], i, xerr=[[max(0.0, r["rate"] - r["lo"])], [max(0.0, r["hi"] - r["rate"])]],
                    fmt="none", ecolor=INK2, lw=0.8)
        if r["rate"] >= 0.3:    # inside the bar, at its start, clear of the whisker at the bar end
            ax.text(0.03, i, label_fn(r), va="center", ha="left", fontsize=6.5, color="white",
                    bbox=dict(facecolor=COND_COLOR.get(r["condition"], MUTED), edgecolor="none", pad=0.8))
        else:
            ax.text(float(r["hi"]) + 0.04, i, label_fn(r), va="center", ha="left", fontsize=6.5, color=INK2)
_HATCH = {None: "", "manual": "", "immediate_upsert": "////", "immediate_append": "....", "never": "xx"}


def freshness_bars(tab: pd.DataFrame, out: Path, name: str = "fig_freshness") -> list[Path]:
    """tab: variant, condition, policy, phase, rate, lo, hi (share 'updated')"""
    variants = _order_variants(tab["variant"])
    fig, axes = plt.subplots(1, len(_PHASES), figsize=(7.2, 0.32 * len(variants) + 1.6), sharey=True,
                             gridspec_kw={"wspace": 0.32})
    for j, (ax, (ph, plabel)) in enumerate(zip(axes, _PHASES)):
        sub = tab[tab["phase"] == ph].set_index("variant")
        _bar_panel(ax, sub, variants, lambda r: f"{r['rate']:.0%}")
        ax.set_title(plabel, fontsize=8, color=INK2)
        _pct_axis(ax)
        ax.set_axisbelow(True)
        ax.grid(axis="y", visible=False)
        if j:
            ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(variants)))
    axes[0].set_yticklabels(variants)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7)
    fig.supxlabel("Share of answers giving the edited (new) value, 95% Wilson CI", fontsize=8, color=INK2, y=-0.06)
    return save(fig, out, name)


def leakage_bars(tab: pd.DataFrame, out: Path, name: str = "fig_leakage") -> list[Path]:
    """tab: variant, condition, policy, phase, rate, lo, hi"""
    phases = [p for p in ["G1_static", "G2_revoked_before_refresh", "G2_revoked_after_refresh", "G3_in_thread"]
              if p in set(tab["phase"])]
    labels = {"G1_static": "Gated from\nthe start", "G2_revoked_before_refresh": "Revoked; new conv.,\nbefore re-index",
              "G2_revoked_after_refresh": "Revoked; new conv.,\nafter re-index",
              "G3_in_thread": "Revoked; same\nconversation"}
    variants = _order_variants(tab["variant"])
    fig, axes = plt.subplots(1, len(phases), figsize=(7.2, 0.32 * len(variants) + 1.6), sharey=True,
                             gridspec_kw={"wspace": 0.32})
    axes = np.atleast_1d(axes)
    for j, (ax, ph) in enumerate(zip(axes, phases)):
        sub = tab[tab["phase"] == ph].set_index("variant")
        _bar_panel(ax, sub, variants, lambda r: f"{int(r['leaks'])}/{int(r['n'])}")
        ax.set_title(labels[ph], fontsize=8, color=INK2)
        _pct_axis(ax)
        ax.set_axisbelow(True)
        ax.grid(axis="y", visible=False)
        if j:
            ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(variants)))
    axes[0].set_yticklabels(variants)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7)
    fig.supxlabel("Share of probes that disclosed gated content, 95% Clopper–Pearson CI", fontsize=8, color=INK2,
                  y=-0.06)
    return save(fig, out, name)


def tokens_calls(tab: pd.DataFrame, out: Path, name: str = "fig_tokens") -> list[Path]:
    """tab: condition, in_tokens_per_q, in_tokens_p90, calls_per_q (two separate panels; no dual axis)"""
    conds = [c for c in COND_ORDER if c in set(tab["condition"])]
    t = tab.set_index("condition")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 0.4 * len(conds) + 1.2), sharey=True)
    for i, c in enumerate(conds):
        a1.barh(i, t.loc[c, "in_tokens_per_q"] / 1000, color=COND_COLOR[c], height=0.6)
        a1.plot([t.loc[c, "in_tokens_p90"] / 1000], [i], marker="|", color=INK2, ms=10)
        a2.barh(i, t.loc[c, "calls_per_q"], color=COND_COLOR[c], height=0.6)
    a1.set_yticks(range(len(conds)))
    a1.set_yticklabels([COND_LABEL[c] for c in conds])
    a1.invert_yaxis()
    a1.set_xlabel("Input tokens per question, thousands\n(bar: mean, tick: 90th percentile)")
    a2.set_xlabel("Model calls per question")
    for a in (a1, a2):
        a.grid(axis="y", visible=False)
    return save(fig, out, name)


def cost_latency(tab: pd.DataFrame, out: Path, name: str = "fig_cost_latency") -> list[Path]:
    """tab: condition, cost_per_1k_usd, latency_median, latency_p90 (two separate panels; no dual axis)"""
    conds = [c for c in COND_ORDER if c in set(tab["condition"])]
    t = tab.set_index("condition")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 0.4 * len(conds) + 1.2), sharey=True)
    for i, c in enumerate(conds):
        a1.barh(i, t.loc[c, "cost_per_1k_usd"], color=COND_COLOR[c], height=0.6)
        a2.barh(i, t.loc[c, "latency_median"], color=COND_COLOR[c], height=0.6)
        a2.plot([t.loc[c, "latency_p90"]], [i], marker="|", color=INK2, ms=10)
    a1.set_yticks(range(len(conds)))
    a1.set_yticklabels([COND_LABEL[c] for c in conds])
    a1.invert_yaxis()
    a1.set_xlabel("Model cost per 1,000 questions (USD)")
    a2.set_xlabel("Latency per answer, s (bar: median, tick: 90th percentile)")
    for a in (a1, a2):
        a.grid(axis="y", visible=False)
    return save(fig, out, name)
