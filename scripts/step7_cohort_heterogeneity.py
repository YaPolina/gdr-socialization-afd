#!/usr/bin/env python3
"""
Estimate treatment effect heterogeneity across age cohorts.

Uses AIPW pseudo-outcomes (psi) from step2 individual-level data.
For each cohort bin, computes weighted mean of psi = cohort-specific ATE,
plus bootstrap CI clustered by xs11.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MAIN_OUTCOME = "y_afd_vote"
MAIN_SPEC = "spec_a_plus"
WEIGHT_VAR = "wghtpew"
CLUSTER_VAR = "xs11"
N_BOOT = 499
SEED = 42

# Age in 1990 bins → 5-year cohorts from 1 to max observed
AGE_1990_BINS = [-100, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 100]
AGE_1990_LABELS = [
    "born after 1990", "1-5", "6-10", "11-15", "16-20",
    "21-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56+",
]

# Birth year bins
YBORN_BINS = [1900, 1945, 1955, 1965, 1975, 1985, 1990, 2010]
YBORN_LABELS = ["≤1945", "1946-55", "1956-65", "1966-75", "1976-85", "1986-90", "1991+"]


def find_repo_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path(__file__).resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def wavg(x: np.ndarray, w: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if m.sum() == 0:
        return float("nan")
    return float(np.average(x[m], weights=w[m]))


def cluster_bootstrap_ci(
    psi: np.ndarray,
    w: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 499,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Cluster bootstrap: resample clusters, compute weighted mean of psi."""
    rng = np.random.RandomState(seed)
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    boot_ests = []

    for _ in range(n_boot):
        sampled = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx = np.concatenate([np.where(clusters == c)[0] for c in sampled])
        est = wavg(psi[idx], w[idx])
        if np.isfinite(est):
            boot_ests.append(est)

    if len(boot_ests) < 50:
        return float("nan"), float("nan"), float("nan")

    boot_ests = np.array(boot_ests)
    se = float(np.std(boot_ests, ddof=1))
    ci_lo = float(np.percentile(boot_ests, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_ests, 100 * (1 - alpha / 2)))
    return se, ci_lo, ci_hi


def compute_cohort_effects(
    df: pd.DataFrame,
    bin_col: str,
    bins: list,
    labels: list,
    outcome: str,
    spec: str,
) -> pd.DataFrame:
    """Compute ATE by cohort bin with cluster bootstrap CI."""
    sub = df[(df["outcome"] == outcome) & (df["spec"] == spec)].copy()
    sub["yborn"] = pd.to_numeric(sub["yborn"], errors="coerce")
    sub["age_1990"] = 1990 - sub["yborn"]

    values = sub[bin_col].to_numpy(dtype=float)
    sub["cohort_bin"] = pd.cut(values, bins=bins, labels=labels, include_lowest=True)

    psi = sub["psi_aipw"].to_numpy(dtype=float)
    w = sub[WEIGHT_VAR].to_numpy(dtype=float)
    t = sub["treatment_value"].to_numpy(dtype=float)
    clusters = sub[CLUSTER_VAR].astype(str).to_numpy()

    rows = []
    for label in labels:
        mask = (sub["cohort_bin"] == label).to_numpy()
        if mask.sum() < 20:
            continue

        ate = wavg(psi[mask], w[mask])
        n = int(mask.sum())
        n_treated = int((t[mask] == 1).sum())
        treated_share = float(np.mean(t[mask]))

        se, ci_lo, ci_hi = cluster_bootstrap_ci(
            psi[mask], w[mask], clusters[mask],
            n_boot=N_BOOT, seed=SEED,
        )

        rows.append({
            "outcome": outcome,
            "spec": spec,
            "cohort_bin": label,
            "n": n,
            "n_treated": n_treated,
            "treated_share": round(treated_share, 3),
            "ate": round(ate, 4),
            "se_cluster_boot": round(se, 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
        })

    return pd.DataFrame(rows)


def plot_cohort_age1990(age_table: pd.DataFrame, outpath: Path) -> None:
    """Plot treatment effect heterogeneity by age at reunification."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

    sub = age_table[
        (age_table["outcome"] == "y_afd_vote") &
        (age_table["spec"] == "spec_a_plus")
    ].copy()

    labels = sub["cohort_bin"].tolist()
    ate = sub["ate"].to_numpy(dtype=float)
    ci_lo = sub["ci_lo"].to_numpy(dtype=float)
    ci_hi = sub["ci_hi"].to_numpy(dtype=float)
    n_obs = sub["n"].to_numpy(dtype=int)

    x = np.arange(len(labels))
    err_lo = ate - ci_lo
    err_hi = ci_hi - ate

    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Color: grey for born after 1990 (placebo), red gradient for GDR-socialized
    colors = []
    for lab in labels:
        if lab == "born after 1990":
            colors.append("#95a5a6")
        else:
            colors.append("#c0392b")

    ax.bar(x, ate, yerr=[err_lo, err_hi], capsize=4, color=colors,
           width=0.65, edgecolor="white", linewidth=1.2, alpha=0.85)

    # Zero line
    ax.axhline(0, ls="--", color="#7f8c8d", lw=1)

    # Overall ATE reference line
    overall_ate = float(age_table[
        (age_table["outcome"] == "y_afd_vote") &
        (age_table["spec"] == "spec_a_plus") &
        (age_table["cohort_bin"] != "born after 1990")
    ]["ate"].mean())

    # Sample size labels on top
    for i, (a, n) in enumerate(zip(ate, n_obs)):
        y_pos = max(ci_hi[i], a) + 0.015
        ax.text(i, y_pos, f"n={n}", ha="center", fontsize=8, color="#7f8c8d")

    # Reunification annotation
    ax.axvline(0.5, ls=":", color="#7f8c8d", lw=1, alpha=0.5)
    ax.text(0.15, ax.get_ylim()[1] if ax.get_ylim()[1] > 0.1 else 0.2,
            "Born\nafter\n1990", ha="center", fontsize=8, color="#95a5a6", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_xlabel("Age at reunification (1990)")
    ax.set_ylabel("CATE: effect on AfD vote probability")
    ax.set_title(
        "Treatment effect heterogeneity by age at reunification\n"
        "(AIPW, spec A+, 95% cluster-bootstrap CI)",
        fontsize=13,
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[ok] Figure: {outpath.name}")


def main() -> None:
    root = find_repo_root()
    in_path = root / "data" / "derived" / "step2_ate_overlap_eastyouth_individual.csv"
    out_dir = root / "outputs" / "tables"
    fig_dir = root / "outputs" / "figures"
    safe_mkdir(out_dir)
    safe_mkdir(fig_dir)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}")

    df = pd.read_csv(in_path)
    df["yborn"] = pd.to_numeric(df["yborn"], errors="coerce")
    df["age_1990"] = 1990 - df["yborn"]

    outcomes = ["y_afd_vote", "y_afd_vote_all_valid"]
    specs = ["spec_a_plus"]

    # By age in 1990
    all_age = []
    for outcome in outcomes:
        for spec in specs:
            result = compute_cohort_effects(df, "age_1990", AGE_1990_BINS, AGE_1990_LABELS, outcome, spec)
            all_age.append(result)

    age_table = pd.concat(all_age, ignore_index=True)
    age_path = out_dir / "table5_cohort_by_age1990.csv"
    age_table.to_csv(age_path, index=False)
    print(f"[ok] Table 5: Cohort effects by age in 1990 ({len(age_table)} rows)")
    print(age_table.to_string(index=False))
    print()

    # Plot
    plot_cohort_age1990(age_table, fig_dir / "fig_cohort_age1990.png")

    # By birth year
    all_yborn = []
    for outcome in outcomes:
        for spec in specs:
            result = compute_cohort_effects(df, "yborn", YBORN_BINS, YBORN_LABELS, outcome, spec)
            all_yborn.append(result)

    yborn_table = pd.concat(all_yborn, ignore_index=True)
    yborn_path = out_dir / "table6_cohort_by_yborn.csv"
    yborn_table.to_csv(yborn_path, index=False)
    print(f"[ok] Table 6: Cohort effects by birth year ({len(yborn_table)} rows)")
    print(yborn_table.to_string(index=False))


if __name__ == "__main__":
    main()
