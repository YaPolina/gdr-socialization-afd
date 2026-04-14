#!/usr/bin/env python3
"""
step8_thesis_figures.py

Publication-quality figures for the thesis:
"Does growing up in the former GDR increase AfD support?"

Produces 10 figures in outputs/figures/.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Style ────────────────────────────────────────────────────────────────────

PALETTE = {
    "east":    "#c0392b",   # red — treated
    "west":    "#2980b9",   # blue — control
    "aipw":    "#2c3e50",   # dark navy
    "ow":      "#e67e22",   # orange
    "accent":  "#27ae60",   # green
    "grey":    "#7f8c8d",
    "light":   "#ecf0f1",
}


def set_thesis_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


# ── Paths ────────────────────────────────────────────────────────────────────

def find_repo_root() -> Path:
    start = Path(__file__).resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start.parents[1]


# ── Helpers ──────────────────────────────────────────────────────────────────

OUTCOME_SHORT = {
    "y_afd_vote":           "AfD vote\n(party choosers)",
    "y_afd_vote_all_valid": "AfD vote\n(all valid resp.)",
    "would_not_vote":       "Would not vote",
}

SPEC_SHORT = {
    "spec_a":      "Base (A)",
    "spec_a_plus": "Extended (A+)",
}

ESTIMAND_SHORT = {
    "aipw_ate": "AIPW",
    "ow_ato":   "Overlap wt.",
}


def weighted_mean(y: np.ndarray, w: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(w) & (w > 0)
    return float(np.average(y[m], weights=w[m])) if m.sum() > 0 else np.nan


def weighted_se_proportion(y: np.ndarray, w: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(w) & (w > 0)
    y, w = y[m], w[m]
    p = np.average(y, weights=w)
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    return float(np.sqrt(p * (1 - p) / n_eff))


# ── Figure 1: Raw AfD gap ────────────────────────────────────────────────────

def fig01_raw_afd_gap(df: pd.DataFrame, outpath: Path) -> None:
    """Bar chart: AfD vote share among East youth vs West youth (survey-weighted)."""
    sub = df[df["y_afd_vote"].notna() & df["east_youth"].notna()].copy()
    sub["y"] = pd.to_numeric(sub["y_afd_vote"], errors="coerce").to_numpy()
    sub["w"] = pd.to_numeric(sub["wghtpew"], errors="coerce").to_numpy()
    sub["t"] = pd.to_numeric(sub["east_youth"], errors="coerce").astype(int).to_numpy()

    groups = {"West youth\n(control)": sub[sub["t"] == 0], "East youth\n(treated)": sub[sub["t"] == 1]}
    means, ses, ns = [], [], []
    for g in groups.values():
        y, w = g["y"].to_numpy(), g["w"].to_numpy()
        means.append(weighted_mean(y, w))
        ses.append(weighted_se_proportion(y, w))
        ns.append(len(g))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    x = np.arange(2)
    bars = ax.bar(x, means, yerr=[1.96 * s for s in ses], capsize=5,
                  color=[PALETTE["west"], PALETTE["east"]], width=0.55,
                  edgecolor="white", linewidth=1.2)

    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text(i, m + 1.96 * ses[i] + 0.008, f"{m:.1%}", ha="center", fontweight="bold", fontsize=12)
        ax.text(i, 0.003, f"n = {n:,}", ha="center", fontsize=9, color="white", fontweight="bold")


    ax.set_xticks(x)
    ax.set_xticklabels(list(groups.keys()))
    ax.set_ylabel("AfD vote share (survey-weighted)")
    ax.set_title("AfD vote intention: East vs West youth")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(0, max(means) * 1.4)

    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [ok] {outpath.name}")


# ── Figure 2: Birth cohort distribution ─────────────────────────────────────

def fig02_birth_cohorts(df: pd.DataFrame, outpath: Path) -> None:
    """Overlapping histograms of birth year by treatment group."""
    sub = df[df["east_youth"].notna() & df["yborn"].notna()].copy()
    sub["t"] = pd.to_numeric(sub["east_youth"], errors="coerce").astype(int)
    sub["yborn"] = pd.to_numeric(sub["yborn"], errors="coerce")

    east = sub.loc[sub["t"] == 1, "yborn"].to_numpy()
    west = sub.loc[sub["t"] == 0, "yborn"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(sub["yborn"].min() - 0.5, sub["yborn"].max() + 1.5, 2)
    ax.hist(west, bins=bins, alpha=0.55, color=PALETTE["west"], label=f"West youth (n={len(west):,})", density=True)
    ax.hist(east, bins=bins, alpha=0.55, color=PALETTE["east"], label=f"East youth (n={len(east):,})", density=True)

    ax.axvline(1990, ls="--", color=PALETTE["grey"], lw=1.2)
    ax.text(1990.5, ax.get_ylim()[1] * 0.92, "Reunification\n(1990)", fontsize=9, color=PALETTE["grey"])

    ax.set_xlabel("Year of birth")
    ax.set_ylabel("Density")
    ax.set_title("Birth year distribution by treatment group")
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="lightgrey",
              bbox_to_anchor=(0.0, 1.0))

    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [ok] {outpath.name}")


# ── Figure 3: PS overlap + ESS ──────────────────────────────────────────────

def fig03_ps_overlap(individual: pd.DataFrame, step3_res: pd.DataFrame, outpath: Path) -> None:
    """Two-panel: PS overlap histogram + effective sample size comparison."""
    sub = individual[
        (individual["treatment"] == "east_youth") &
        (individual["spec"] == "spec_a_plus") &
        (individual["outcome"] == "y_afd_vote")
    ].copy()
    e = pd.to_numeric(sub["e_hat"], errors="coerce")
    t = pd.to_numeric(sub["treatment_value"], errors="coerce")
    m = e.notna() & t.isin([0, 1])
    e, t = e[m].to_numpy(), t[m].astype(int).to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]})

    # Left: PS histogram
    bins = np.linspace(0, 1, 50)
    ax1.hist(e[t == 0], bins=bins, alpha=0.5, density=True, color=PALETTE["west"], label="West youth")
    ax1.hist(e[t == 1], bins=bins, alpha=0.5, density=True, color=PALETTE["east"], label="East youth")
    ax1.axvline(0.05, ls=":", color=PALETTE["grey"], lw=1)
    ax1.axvline(0.95, ls=":", color=PALETTE["grey"], lw=1)
    ax1.fill_betweenx([0, ax1.get_ylim()[1] * 1.1], 0, 0.05, alpha=0.06, color="black")
    ax1.fill_betweenx([0, ax1.get_ylim()[1] * 1.1], 0.95, 1.0, alpha=0.06, color="black")
    ax1.set_xlabel("Estimated propensity score ê(X)")
    ax1.set_ylabel("Density")
    ax1.set_title("Propensity score overlap")
    ax1.legend()

    # Right: Effective sample sizes
    r = step3_res[
        (step3_res["treatment"] == "east_youth") &
        (step3_res["spec"] == "spec_a_plus") &
        (step3_res["outcome"] == "y_afd_vote")
    ].iloc[0]
    labels = ["Nominal\n(survey wt.)", "IPW × survey", "OW × survey"]
    ess_vals = [float(r["ess_survey_only"]), float(r["ess_ipw_total"]), float(r["ess_ow_total"])]
    colors = [PALETTE["grey"], PALETTE["aipw"], PALETTE["ow"]]
    bars = ax2.barh(range(3), ess_vals, color=colors, height=0.5, edgecolor="white")
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Effective sample size")
    ax2.set_title("Weighting efficiency")
    for i, v in enumerate(ess_vals):
        ax2.text(v + 30, i, f"{v:.0f}", va="center", fontsize=10)
    ax2.set_xlim(0, max(ess_vals) * 1.2)

    fig.suptitle("Overlap diagnostics (spec A+, AfD vote)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [ok] {outpath.name}")


# ── Figure 4: Love plot ─────────────────────────────────────────────────────

def fig04_love_plot(balance: pd.DataFrame, outpath: Path) -> None:
    """Cleaner love plot: raw vs overlap-weighted SMD."""
    sub = balance[
        (balance["spec"] == "spec_a_plus") &
        (balance["outcome"] == "y_afd_vote") &
        (balance["weighting"].isin(["raw_unweighted", "ow_total"]))
    ].copy()

    raw = sub[sub["weighting"] == "raw_unweighted"].copy()
    ow = sub[sub["weighting"] == "ow_total"].copy()

    # Top 25 by raw imbalance
    top_terms = raw.sort_values("abs_smd", ascending=False).head(25)["variable_display"].values
    raw = raw[raw["variable_display"].isin(top_terms)]
    ow = ow[ow["variable_display"].isin(top_terms)]

    order = (
        raw.groupby("variable_display")["abs_smd"].max()
        .sort_values(ascending=True).index.tolist()
    )
    y_map = {t: i for i, t in enumerate(order)}

    fig, ax = plt.subplots(figsize=(8, 0.32 * len(order) + 1.5))

    # Draw connecting lines
    for term in order:
        r_val = raw.loc[raw["variable_display"] == term, "abs_smd"]
        o_val = ow.loc[ow["variable_display"] == term, "abs_smd"]
        if len(r_val) > 0 and len(o_val) > 0:
            ax.plot([r_val.values[0], o_val.values[0]], [y_map[term]] * 2,
                    color=PALETTE["light"], lw=1.5, zorder=1)

    ax.scatter(raw["abs_smd"].values, [y_map[t] for t in raw["variable_display"]],
               color=PALETTE["east"], s=35, label="Unweighted", zorder=2, marker="D")
    ax.scatter(ow["abs_smd"].values, [y_map[t] for t in ow["variable_display"]],
               color=PALETTE["accent"], s=35, label="Overlap weighted", zorder=2)

    ax.axvline(0.1, ls="--", color=PALETTE["grey"], lw=1, label="SMD = 0.10 threshold")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("|Standardized mean difference|")
    ax.set_title("Covariate balance: before vs after overlap weighting")
    ax.legend(loc="lower right", fontsize=9)

    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [ok] {outpath.name}")


# ── Figure 5: Main results forest plot ──────────────────────────────────────

def fig05_main_forest(step4: pd.DataFrame, outpath: Path) -> None:
    """Comprehensive forest plot: both specs, both estimators, cluster-bootstrap CI (land)."""
    outcome = "y_afd_vote"
    cluster = "land"
    specs = ["spec_a", "spec_a_plus"]
    estimands = ["aipw_ate", "ow_ato"]

    rows = []
    for spec in specs:
        for est in estimands:
            r = step4[
                (step4["outcome"] == outcome) &
                (step4["spec"] == spec) &
                (step4["estimand"] == est) &
                (step4["cluster_var"] == cluster)
            ]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({
                "label": f"{SPEC_SHORT[spec]}  ×  {ESTIMAND_SHORT[est]}",
                "est": float(r["estimate"]),
                "lo": float(r["ci_cluster_bootstrap_low"]),
                "hi": float(r["ci_cluster_bootstrap_high"]),
                "spec": spec,
                "estimand": est,
            })

    fig, ax = plt.subplots(figsize=(8, 3.5))
    y = np.arange(len(rows))[::-1]

    for i, row in enumerate(rows):
        color = PALETTE["aipw"] if row["estimand"] == "aipw_ate" else PALETTE["ow"]
        marker = "o" if row["estimand"] == "aipw_ate" else "s"
        ax.errorbar(row["est"], y[i], xerr=[[row["est"] - row["lo"]], [row["hi"] - row["est"]]],
                     fmt=marker, color=color, capsize=4, markersize=7, lw=1.8, markeredgecolor="white")
        ax.text(row["hi"] + 0.004, y[i], f"{row['est']:.3f}", va="center", fontsize=9, color=color)

    ax.axvline(0, ls="--", color=PALETTE["grey"], lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("Effect on AfD vote probability (pp)")
    ax.set_title("Main causal estimates: East youth → AfD vote\n(95% cluster-bootstrap CI, clustered by federal state)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

    # Add right margin for labels
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] + 0.025)

    # Manual legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color=PALETTE["aipw"], lw=0, markersize=7, label="AIPW (ATE)"),
        Line2D([0], [0], marker="s", color=PALETTE["ow"], lw=0, markersize=7, label="Overlap wt. (ATO)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.9)

    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [ok] {outpath.name}")


# ── Figure 6: Robustness across outcomes ─────────────────────────────────────

def fig06_robustness_outcomes(step4: pd.DataFrame, outpath: Path) -> None:
    """Forest plot comparing all 3 outcomes, spec_a_plus, AIPW, land clustering."""
    spec = "spec_a_plus"
    est = "aipw_ate"
    cluster = "land"
    outcomes = ["y_afd_vote", "y_afd_vote_all_valid", "would_not_vote"]

    rows = []
    for out in outcomes:
        r = step4[
            (step4["outcome"] == out) &
            (step4["spec"] == spec) &
            (step4["estimand"] == est) &
            (step4["cluster_var"] == cluster)
        ]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append({
            "label": OUTCOME_SHORT[out],
            "est": float(r["estimate"]),
            "lo": float(r["ci_cluster_bootstrap_low"]),
            "hi": float(r["ci_cluster_bootstrap_high"]),
        })

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    y = np.arange(len(rows))[::-1]
    colors = [PALETTE["east"], PALETTE["aipw"], PALETTE["grey"]]

    for i, (row, c) in enumerate(zip(rows, colors)):
        ax.errorbar(row["est"], y[i], xerr=[[row["est"] - row["lo"]], [row["hi"] - row["est"]]],
                     fmt="o", color=c, capsize=4, markersize=8, lw=1.8, markeredgecolor="white")
        side = "right" if row["est"] > 0.03 else "left"
        offset = 0.004 if side == "left" else -0.004
        ax.text(row["hi"] + 0.004, y[i], f"{row['est']:.3f} [{row['lo']:.3f}, {row['hi']:.3f}]",
                va="center", fontsize=9, color=c)

    ax.axvline(0, ls="--", color=PALETTE["grey"], lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("AIPW effect estimate (pp)")
    ax.set_title("Robustness: effect across outcome definitions\n(spec A+, cluster-bootstrap CI by federal state)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [ok] {outpath.name}")


# ── Figure 7: CI comparison ─────────────────────────────────────────────────

def fig07_ci_comparison(step4: pd.DataFrame, outpath: Path) -> None:
    """How CIs change with different clustering and SE methods."""
    outcome = "y_afd_vote"
    spec = "spec_a_plus"
    est = "aipw_ate"

    sub = step4[
        (step4["outcome"] == outcome) &
        (step4["spec"] == spec) &
        (step4["estimand"] == est) &
        (step4["cluster_var"].isin(["xs11", "land"]))
    ].copy()

    cluster_labels = {
        "xs11": "Sampling point\n(191 clusters)",
        "land": "Federal state\n(17 clusters)",
    }

    fig, ax = plt.subplots(figsize=(8, 3.5))
    methods = [
        ("ci_naive_low", "ci_naive_high", "Naive (IID)"),  # ASCII only
        ("ci_cluster_low", "ci_cluster_high", "Cluster-robust sandwich"),
        ("ci_cluster_bootstrap_low", "ci_cluster_bootstrap_high", "Cluster bootstrap"),
    ]
    colors_m = [PALETTE["accent"], PALETTE["aipw"], PALETTE["east"]]

    y_base = {"xs11": 2, "land": 0}
    for cv in ["xs11", "land"]:
        r = sub[sub["cluster_var"] == cv].iloc[0]
        point = float(r["estimate"])
        for j, (lo_col, hi_col, label) in enumerate(methods):
            yy = y_base[cv] + (1 - j) * 0.28
            lo, hi = float(r[lo_col]), float(r[hi_col])
            lab = label if cv == "xs11" else None
            ax.plot([lo, hi], [yy, yy], color=colors_m[j], lw=2.5, solid_capstyle="round", label=lab)
            ax.plot(point, yy, "o", color=colors_m[j], markersize=5, markeredgecolor="white", zorder=5)

    ax.axvline(0, ls="--", color=PALETTE["grey"], lw=0.8)
    ax.set_yticks([0.28, 2.28])
    ax.set_yticklabels([cluster_labels["land"], cluster_labels["xs11"]])
    ax.set_xlabel("AIPW effect estimate")
    ax.set_title("Inference sensitivity to clustering level\n(AfD vote, spec A+)")
    ax.legend(loc="upper left", fontsize=9, bbox_to_anchor=(0.0, 1.0))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [ok] {outpath.name}")


# ── Figure 8: Spatial placebo ────────────────────────────────────────────────

def fig08_spatial_placebo(placebo_dir: Path, outpath: Path) -> None:
    """Annotated placebo histogram."""
    # Use xs11 for y_afd_vote spec_a_plus
    res_path = placebo_dir / "step6_spatial_placebo__east_youth__y_afd_vote__spec_a_plus__xs11_results.csv"
    draws_path = placebo_dir / "step6_spatial_placebo__east_youth__y_afd_vote__spec_a_plus__xs11_draws.csv"

    if not res_path.exists() or not draws_path.exists():
        print(f"  [skip] Placebo files not found")
        return

    res = pd.read_csv(res_path).iloc[0]
    draws = pd.read_csv(draws_path)["placebo_tau"].to_numpy()
    tau_obs = float(res["tau_observed"])
    pval = float(res["permutation_p_value"])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(draws, bins=45, color=PALETTE["west"], alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(tau_obs, color=PALETTE["east"], ls="--", lw=2.2, label=f"Observed τ = {tau_obs:.3f}")
    ax.axvline(0, ls=":", color=PALETTE["grey"], lw=0.8)

    # Annotation — avoid "p = 0.000"; use "p < 0.001" instead
    n_perms = int(res["B"])
    if pval < 0.001:
        p_str = "p < 0.001"
    else:
        p_str = f"p = {pval:.3f}"
    ax.text(tau_obs + 0.002, ax.get_ylim()[1] * 0.85,
            f"{p_str}\n({n_perms:,} permutations)",
            fontsize=10, color=PALETTE["east"], fontweight="bold")

    ax.set_xlabel("Placebo treatment effect (τ)")
    ax.set_ylabel("Frequency")
    ax.set_title("Spatial placebo test\n(permuting treatment across sampling points)")
    # No legend box — the dashed line label is shown via the annotation text
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [ok] {outpath.name}")


# ── Figure 9: Rosenbaum sensitivity (combined) ──────────────────────────────

def fig09_rosenbaum_combined(tables_dir: Path, outpath: Path) -> None:
    """Sensitivity curves for all 8 mediators on one plot."""
    summary_path = tables_dir / "step5_rosenbaum_delta_curve_summary_all.csv"
    if not summary_path.exists():
        print(f"  [skip] Rosenbaum summary not found")
        return

    summary = pd.read_csv(summary_path)
    summary = summary[
        (summary["outcome"] == "y_afd_vote") &
        (summary["spec"] == "spec_a_plus") &
        (summary["treatment"] == "east_youth")
    ]

    mediator_labels = {
        "df44": "Lived with both parents at 15 (df44)",
        "dn05": "German citizen from birth (dn05)",
        "dn07": "Born in territory of FRG (dn07)",
        "no_denomination": "No religious denomination",
        "rp01": "Interest in politics (rp01)",
        "gs01": "Subjective health (gs01)",
        "ru01": "Urban/rural residence (ru01)",
        "rb07": "Religiosity (rb07)",
    }
    _cycle = [PALETTE["east"], PALETTE["aipw"], PALETTE["accent"],
              PALETTE["ow"], "#8e44ad", "#16a085", "#d35400", "#2c3e50"]
    colors_med = {med: _cycle[i % len(_cycle)]
                  for i, med in enumerate(mediator_labels)}

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for _, row in summary.iterrows():
        med = row["mediator_col"]
        curve_path = tables_dir / f"step5_rosenbaum_delta_curve__y_afd_vote__spec_a_plus__ate_aipw__xs11__S_{med}_curve.csv"
        if not curve_path.exists():
            continue

        curve = pd.read_csv(curve_path)
        x = curve["delta_sd_units"].to_numpy()
        y_adj = curve["tau_adjusted"].to_numpy()
        delta_star = float(row["delta_star"])

        label = mediator_labels.get(med, med)
        c = colors_med.get(med, PALETTE["grey"])

        ax.plot(x, y_adj, color=c, lw=2, label=f"{label} (δ*={delta_star:.1f})")

        if "tau_gamma_lo" in curve.columns:
            ax.fill_between(x, curve["tau_gamma_lo"].to_numpy(), curve["tau_gamma_hi"].to_numpy(),
                            color=c, alpha=0.08)

    ax.axhline(0, ls="--", color=PALETTE["grey"], lw=1)

    tau_hat = float(summary.iloc[0]["tau_hat"])
    ax.axhline(tau_hat, ls="-", color=PALETTE["grey"], lw=0.8, alpha=0.4)
    ax.text(0.5, tau_hat + 0.003, f"τ̂ = {tau_hat:.3f}", fontsize=9, color=PALETTE["grey"])

    ax.set_xlabel("δ: hypothetical shift in mediator due to treatment (SD units)")
    ax.set_ylabel("Adjusted treatment effect τ(δ)")
    ax.set_title("Sensitivity analysis: how large must unobserved confounding be\nto explain away the effect?")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [ok] {outpath.name}")


# ── Figure 10: Geographic map ────────────────────────────────────────────────

DG10_TO_STATE = {
    1: "Baden-Württemberg", 2: "Bayern", 3: "Berlin", 4: "Bremen",
    5: "Hamburg", 6: "Hessen", 7: "Niedersachsen", 8: "Nordrhein-Westfalen",
    9: "Rheinland-Pfalz", 10: "Saarland", 11: "Schleswig-Holstein",
    12: "Berlin", 13: "Brandenburg", 14: "Mecklenburg-Vorpommern",
    15: "Sachsen", 16: "Sachsen-Anhalt", 17: "Thüringen",
}

STATE_NAME_MAP = {
    "Baden-Württemberg": "Baden-Württemberg", "Baden-Wuerttemberg": "Baden-Württemberg",
    "Baden Württemberg": "Baden-Württemberg", "Baden-Wurttemberg": "Baden-Württemberg",
    "Bayern": "Bayern", "Bavaria": "Bayern",
    "Berlin": "Berlin", "Bremen": "Bremen", "Hamburg": "Hamburg",
    "Hessen": "Hessen", "Hesse": "Hessen",
    "Niedersachsen": "Niedersachsen", "Lower Saxony": "Niedersachsen",
    "Nordrhein-Westfalen": "Nordrhein-Westfalen", "North Rhine-Westphalia": "Nordrhein-Westfalen",
    "Rheinland-Pfalz": "Rheinland-Pfalz", "Rhineland-Palatinate": "Rheinland-Pfalz",
    "Saarland": "Saarland", "Schleswig-Holstein": "Schleswig-Holstein",
    "Brandenburg": "Brandenburg",
    "Mecklenburg-Vorpommern": "Mecklenburg-Vorpommern",
    "Mecklenburg Western Pomerania": "Mecklenburg-Vorpommern",
    "Mecklenburg-Western Pomerania": "Mecklenburg-Vorpommern",
    "Sachsen": "Sachsen", "Saxony": "Sachsen",
    "Sachsen-Anhalt": "Sachsen-Anhalt", "Saxony-Anhalt": "Sachsen-Anhalt",
    "Thüringen": "Thüringen", "Thueringen": "Thüringen", "Thuringia": "Thüringen",
}

HIGH_SES_GROUPS = {"managerial", "professional", "associate_professional"}
LOW_SES_GROUPS = {"skilled_industrial", "lower_status_employee"}


def _load_gdf(root: Path):
    import geopandas as gpd
    geo_path = root / "data" / "external" / "germany_states.geojson"
    if not geo_path.exists():
        return None
    gdf = gpd.read_file(geo_path)
    for col_name in ["shapeName", "name", "NAME", "NAME_1", "GEN"]:
        if col_name in gdf.columns:
            gdf["state_name"] = gdf[col_name].astype(str).map(lambda x: STATE_NAME_MAP.get(x.strip()))
            break
    return gdf[gdf["state_name"].notna()].copy()


def fig10_map_afd_overall(df: pd.DataFrame, outpath: Path, root: Path) -> None:
    """Single-panel introductory map: overall AfD vote share by federal state."""
    try:
        import geopandas as gpd  # noqa: F401
    except ImportError:
        print("  [skip] geopandas not installed, skipping map")
        return

    gdf = _load_gdf(root)
    if gdf is None:
        print("  [skip] GeoJSON not found")
        return

    sub = df[df["y_afd_vote"].notna() & df["dg10"].notna()].copy()
    sub["dg10"] = pd.to_numeric(sub["dg10"], errors="coerce")
    sub["y"] = pd.to_numeric(sub["y_afd_vote"], errors="coerce")
    sub["w"] = pd.to_numeric(sub["wghtpew"], errors="coerce")
    sub["state"] = sub["dg10"].map(DG10_TO_STATE)
    sub = sub[sub["state"].notna() & sub["w"].notna() & (sub["w"] > 0)]

    state_data = []
    for state, g in sub.groupby("state"):
        state_data.append({
            "state_name": state,
            "afd_share": weighted_mean(g["y"].to_numpy(), g["w"].to_numpy()),
        })
    gdf = gdf.merge(pd.DataFrame(state_data), on="state_name", how="left")

    fig, ax = plt.subplots(figsize=(6, 8))
    gdf.plot(column="afd_share", cmap="YlOrRd", linewidth=0.8, edgecolor="black",
             legend=True, ax=ax, missing_kwds={"color": "lightgrey"},
             legend_kwds={"label": "AfD vote share", "shrink": 0.55, "aspect": 20})
    ax.set_title("AfD vote intention by federal state\n(ALLBUS 2023, survey-weighted)", fontsize=13)
    ax.axis("off")

    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [ok] {outpath.name}")


def fig11_map_ses(df: pd.DataFrame, outpath: Path, root: Path) -> None:
    """Three-panel map: AfD share by family SES (high / low / gap)."""
    try:
        import geopandas as gpd  # noqa: F401
    except ImportError:
        print("  [skip] geopandas not installed, skipping map")
        return

    gdf = _load_gdf(root)
    if gdf is None:
        print("  [skip] GeoJSON not found")
        return

    sub = df[df["y_afd_vote"].notna() & df["dg10"].notna()].copy()
    sub["dg10"] = pd.to_numeric(sub["dg10"], errors="coerce")
    sub["y"] = pd.to_numeric(sub["y_afd_vote"], errors="coerce")
    sub["w"] = pd.to_numeric(sub["wghtpew"], errors="coerce")
    sub["state"] = sub["dg10"].map(DG10_TO_STATE)
    sub = sub[sub["state"].notna() & sub["w"].notna() & (sub["w"] > 0)]

    # Classify family SES
    def classify(row):
        f = str(row.get("feseg_grp", "")) if pd.notna(row.get("feseg_grp")) else ""
        m = str(row.get("meseg_grp", "")) if pd.notna(row.get("meseg_grp")) else ""
        has_high = f in HIGH_SES_GROUPS or m in HIGH_SES_GROUPS
        has_low = f in LOW_SES_GROUPS or m in LOW_SES_GROUPS
        if has_high and not has_low:
            return "high"
        if has_low and not has_high:
            return "low"
        return "other"

    sub["ses"] = sub.apply(classify, axis=1)

    rows = []
    for state, g in sub.groupby("state"):
        hi = g[g["ses"] == "high"]
        lo = g[g["ses"] == "low"]
        hi_share = weighted_mean(hi["y"].to_numpy(), hi["w"].to_numpy()) if len(hi) > 5 else np.nan
        lo_share = weighted_mean(lo["y"].to_numpy(), lo["w"].to_numpy()) if len(lo) > 5 else np.nan
        rows.append({"state_name": state, "high_ses": hi_share, "low_ses": lo_share})

    sdf = pd.DataFrame(rows)
    sdf["gap"] = sdf["low_ses"] - sdf["high_ses"]
    gdf = gdf.merge(sdf, on="state_name", how="left")

    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    panels = [
        ("high_ses", "AfD share: high-SES family", "Blues"),
        ("low_ses", "AfD share: low-SES family", "Blues"),
        ("gap", "Gap: low minus high SES", "coolwarm"),
    ]
    for ax, (col, title, cmap) in zip(axes, panels):
        vmin = None
        vcenter = None
        if col == "gap":
            from matplotlib.colors import TwoSlopeNorm
            vals = gdf[col].dropna()
            if len(vals) > 0:
                norm = TwoSlopeNorm(vmin=vals.min(), vcenter=0, vmax=vals.max())
                gdf.plot(column=col, cmap=cmap, linewidth=0.8, edgecolor="black",
                         legend=True, ax=ax, norm=norm,
                         missing_kwds={"color": "lightgrey"})
            else:
                gdf.plot(column=col, cmap=cmap, linewidth=0.8, edgecolor="black",
                         legend=True, ax=ax, missing_kwds={"color": "lightgrey"})
        else:
            gdf.plot(column=col, cmap=cmap, linewidth=0.8, edgecolor="black",
                     legend=True, ax=ax, missing_kwds={"color": "lightgrey"})
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.suptitle("AfD vote intention by family socioeconomic background", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [ok] {outpath.name}")


# ── Figure 12: Treatment prevalence + AfD by state ──────────────────────────

def fig12_map_treatment_outcome(df: pd.DataFrame, outpath: Path, root: Path) -> None:
    """Two-panel map: East youth prevalence + AfD share by state."""
    try:
        import geopandas as gpd  # noqa: F401
    except ImportError:
        print("  [skip] geopandas not installed, skipping map")
        return

    gdf = _load_gdf(root)
    if gdf is None:
        print("  [skip] GeoJSON not found")
        return

    sub = df[df["y_afd_vote"].notna() & df["east_youth"].notna() & df["dg10"].notna()].copy()
    sub["dg10"] = pd.to_numeric(sub["dg10"], errors="coerce")
    sub["y"] = pd.to_numeric(sub["y_afd_vote"], errors="coerce")
    sub["w"] = pd.to_numeric(sub["wghtpew"], errors="coerce")
    sub["t"] = pd.to_numeric(sub["east_youth"], errors="coerce").astype(int)
    sub["state"] = sub["dg10"].map(DG10_TO_STATE)
    sub = sub[sub["state"].notna() & sub["w"].notna() & (sub["w"] > 0)]

    state_data = []
    for state, g in sub.groupby("state"):
        y, w, t = g["y"].to_numpy(), g["w"].to_numpy(), g["t"].to_numpy()
        state_data.append({
            "state_name": state,
            "afd_share": weighted_mean(y, w),
            "east_youth_share": weighted_mean(t.astype(float), w),
        })

    gdf = gdf.merge(pd.DataFrame(state_data), on="state_name", how="left")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))

    gdf.plot(column="east_youth_share", cmap="Reds", linewidth=0.8, edgecolor="black",
             legend=True, ax=ax1, missing_kwds={"color": "lightgrey"},
             legend_kwds={"label": "Share", "shrink": 0.55, "aspect": 20})
    ax1.set_title("Treatment prevalence\n(share of East youth)", fontsize=12)
    ax1.axis("off")

    gdf.plot(column="afd_share", cmap="Blues", linewidth=0.8, edgecolor="black",
             legend=True, ax=ax2, missing_kwds={"color": "lightgrey"},
             legend_kwds={"label": "Share", "shrink": 0.55, "aspect": 20})
    ax2.set_title("AfD vote share\n(survey-weighted)", fontsize=12)
    ax2.axis("off")

    fig.suptitle("Geographic co-variation: treatment and outcome by federal state", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  [ok] {outpath.name}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    set_thesis_style()
    root = find_repo_root()

    data_path = root / "data" / "derived" / "step1_model_afd.csv"
    indiv_path = root / "data" / "derived" / "step2_ate_overlap_eastyouth_individual.csv"
    tables_dir = root / "outputs" / "tables"
    placebo_dir = root / "outputs" / "tables" / "placebo"

    out_dir = root / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(data_path)
    individual = pd.read_csv(indiv_path)
    step2_res = pd.read_csv(tables_dir / "step2_ate_overlap_eastyouth_results.csv")
    step4 = pd.read_csv(tables_dir / "step4_inference_multicluster_eastyouth_results.csv")
    balance = pd.read_csv(tables_dir / "step3_balance_diagnostics_detail.csv")

    print("\nGenerating thesis figures...")

    fig01_raw_afd_gap(df, out_dir / "fig01_raw_afd_gap.png")
    fig02_birth_cohorts(df, out_dir / "fig02_birth_cohorts.png")
    fig03_ps_overlap(individual, step2_res, out_dir / "fig03_ps_overlap_ess.png")
    fig04_love_plot(balance, out_dir / "fig04_love_plot.png")
    fig05_main_forest(step4, out_dir / "fig05_main_results_forest.png")
    fig06_robustness_outcomes(step4, out_dir / "fig06_robustness_outcomes.png")
    fig07_ci_comparison(step4, out_dir / "fig07_ci_clustering.png")
    fig08_spatial_placebo(placebo_dir, out_dir / "fig08_spatial_placebo.png")
    fig09_rosenbaum_combined(tables_dir, out_dir / "fig09_rosenbaum_sensitivity.png")
    fig10_map_afd_overall(df, out_dir / "fig10_map_afd_overall.png", root)
    fig11_map_ses(df, out_dir / "fig11_map_ses_gap.png", root)
    fig12_map_treatment_outcome(df, out_dir / "fig12_map_treatment_outcome.png", root)

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
