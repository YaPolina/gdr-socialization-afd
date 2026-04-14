#!/usr/bin/env python3
"""
step6_spatial_placebo.py

Spatial placebo test for ALLBUS 2023 thesis project.

What this version does:
1) Loads the project CSV produced by step1_build_model_afd.py.
2) Selects one outcome, one treatment, and one X specification (spec_a or spec_a_plus).
3) Estimates the observed effect with a weighted linear probability model (WLS)
   and cluster-robust standard errors.
4) Builds a spatial placebo distribution by shuffling cluster-level treatment
   intensities across clusters and then randomly re-assigning binary treatment
   status within each cluster to match the permuted intensity.
5) Computes a two-sided placebo p-value.
6) Saves summary CSV, placebo draws CSV, and histogram PNG.

This rewrite is compatible with the user's current project structure:
- treatment: east_youth
- outcomes: y_afd_vote, y_afd_vote_all_valid, would_not_vote
- covariate specs: spec_a, spec_a_plus
- weights: wghtpew
- clusters: land or xs11

Dependencies: pandas, numpy, statsmodels, matplotlib, argparse
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


DEFAULT_INPUT = "data/derived/step1_model_afd.csv"
DEFAULT_OUTPUT_DIR = "outputs/tables/placebo"
DEFAULT_TREATMENT = "east_youth"
DEFAULT_OUTCOME = "y_afd_vote"
DEFAULT_SPEC = "spec_a_plus"
DEFAULT_CLUSTER = "land"
DEFAULT_B = 1000
DEFAULT_SEED = 12345
DEFAULT_WEIGHT = "wghtpew"

SPECS: dict[str, list[str]] = {
    "spec_a": ["yborn", "sex", "feseg_grp", "meseg_grp"],
    "spec_a_plus": ["yborn", "sex", "feseg_grp", "meseg_grp", "feduc", "meduc", "fde01", "mde01"],
}

NUMERIC_COVARIATES = {"yborn"}
ALLOWED_OUTCOMES = {"y_afd_vote", "y_afd_vote_all_valid", "would_not_vote"}
ALLOWED_TREATMENTS = {"east_youth"}
ALLOWED_CLUSTERS = {"land", "xs11"}


def find_repo_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path(__file__).resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spatial placebo test for east_youth in ALLBUS 2023")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save outputs")
    parser.add_argument("--outcome", type=str, default=DEFAULT_OUTCOME, choices=sorted(ALLOWED_OUTCOMES))
    parser.add_argument("--treatment", type=str, default=DEFAULT_TREATMENT, choices=sorted(ALLOWED_TREATMENTS))
    parser.add_argument("--spec", type=str, default=DEFAULT_SPEC, choices=sorted(SPECS.keys()))
    parser.add_argument("--cluster-var", type=str, default=DEFAULT_CLUSTER, choices=sorted(ALLOWED_CLUSTERS))
    parser.add_argument("--weight-var", type=str, default=DEFAULT_WEIGHT)
    parser.add_argument("--B", type=int, default=DEFAULT_B, help="Number of placebo draws")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    return parser.parse_args()


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def check_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sum(x[mask] * w[mask]) / np.sum(w[mask]))


def build_analysis_sample(
    df: pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    spec: str,
    cluster_var: str,
    weight_var: str,
) -> pd.DataFrame:
    covariates = SPECS[spec]
    required = [outcome, treatment, cluster_var, weight_var] + covariates
    check_required_columns(df, required)

    out = df.copy()

    out[outcome] = pd.to_numeric(out[outcome], errors="coerce")
    out[treatment] = pd.to_numeric(out[treatment], errors="coerce")
    out[weight_var] = pd.to_numeric(out[weight_var], errors="coerce")

    for col in covariates:
        if col in NUMERIC_COVARIATES:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=required).copy()
    out = out.loc[out[weight_var] > 0].copy()
    out = out.loc[out[treatment].isin([0, 1])].copy()
    out = out.loc[out[outcome].isin([0, 1])].copy()

    if out.empty:
        raise ValueError("Analysis sample is empty after dropping missing values.")

    return out


def build_covariate_matrix(df: pd.DataFrame, spec: str) -> pd.DataFrame:
    covariates = SPECS[spec]
    x = df[covariates].copy()

    numeric_cols = [c for c in covariates if c in NUMERIC_COVARIATES]
    categorical_cols = [c for c in covariates if c not in NUMERIC_COVARIATES]

    for col in categorical_cols:
        x[col] = x[col].astype(object)
        x[col] = x[col].where(~x[col].isna(), "__MISSING__")
        x[col] = x[col].astype(str)

    x = pd.get_dummies(x, columns=categorical_cols, drop_first=True, dtype=float)
    x = sm.add_constant(x, has_constant="add")
    x = x.astype(float)
    return x


def build_design_matrix(x_cov: pd.DataFrame, treatment_values: np.ndarray, treatment_name: str) -> pd.DataFrame:
    X = x_cov.copy()
    X.insert(1, treatment_name, np.asarray(treatment_values, dtype=float))
    return X


def estimate_lpm_cluster(
    *,
    y: np.ndarray,
    treatment_values: np.ndarray,
    x_cov: pd.DataFrame,
    w: np.ndarray,
    clusters: np.ndarray,
    treatment_name: str,
) -> tuple[float, float, sm.regression.linear_model.RegressionResultsWrapper]:
    X = build_design_matrix(x_cov=x_cov, treatment_values=treatment_values, treatment_name=treatment_name)

    model = sm.WLS(y, X, weights=w).fit(
        cov_type="cluster",
        cov_kwds={"groups": clusters},
    )

    tau_hat = float(model.params[treatment_name])
    cluster_se = float(model.bse[treatment_name])
    return tau_hat, cluster_se, model


def build_cluster_index(cluster_series: pd.Series) -> dict[object, np.ndarray]:
    groups: dict[object, list[int]] = {}
    for i, g in enumerate(cluster_series.tolist()):
        groups.setdefault(g, []).append(i)
    return {g: np.asarray(idx, dtype=int) for g, idx in groups.items()}


def weighted_cluster_treated_share(
    treatment: np.ndarray,
    weights: np.ndarray,
    cluster_index: dict[object, np.ndarray],
) -> pd.Series:
    shares = {}
    for g, idx in cluster_index.items():
        shares[g] = weighted_mean(treatment[idx], weights[idx])
    return pd.Series(shares, dtype=float)


def assign_placebo_treatment(
    *,
    n: int,
    cluster_index: dict[object, np.ndarray],
    permuted_cluster_shares: pd.Series,
    rng: np.random.Generator,
) -> np.ndarray:
    t_placebo = np.zeros(n, dtype=float)

    for g, idx in cluster_index.items():
        p = float(permuted_cluster_shares.loc[g])
        p = min(max(p, 0.0), 1.0)
        n_g = len(idx)

        # Stochastic rounding so the expected treated count equals p * n_g.
        raw_k = p * n_g
        k = int(np.floor(raw_k))
        if rng.random() < (raw_k - k):
            k += 1
        k = max(0, min(k, n_g))

        if k > 0:
            treated_idx = rng.choice(idx, size=k, replace=False)
            t_placebo[treated_idx] = 1.0

    return t_placebo


def spatial_placebo(
    *,
    y: np.ndarray,
    treatment: np.ndarray,
    x_cov: pd.DataFrame,
    weights: np.ndarray,
    clusters: pd.Series,
    treatment_name: str,
    B: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    cluster_index = build_cluster_index(clusters)
    cluster_shares = weighted_cluster_treated_share(treatment, weights, cluster_index)
    cluster_labels = cluster_shares.index.to_numpy()
    share_values = cluster_shares.to_numpy(dtype=float)

    placebo_taus = np.empty(B, dtype=float)

    for b in range(B):
        shuffled_shares = rng.permutation(share_values)
        permuted_cluster_shares = pd.Series(shuffled_shares, index=cluster_labels, dtype=float)

        t_placebo = assign_placebo_treatment(
            n=len(y),
            cluster_index=cluster_index,
            permuted_cluster_shares=permuted_cluster_shares,
            rng=rng,
        )

        tau_b, _, _ = estimate_lpm_cluster(
            y=y,
            treatment_values=t_placebo,
            x_cov=x_cov,
            w=weights,
            clusters=clusters.to_numpy(),
            treatment_name=treatment_name,
        )
        placebo_taus[b] = tau_b

    return placebo_taus


def save_results(
    output_path: Path,
    *,
    outcome: str,
    treatment: str,
    spec: str,
    cluster_var: str,
    n_obs: int,
    n_clusters: int,
    tau_obs: float,
    se_cluster: float,
    p_value: float,
    placebo_mean: float,
    placebo_sd: float,
    B: int,
    seed: int,
) -> None:
    results = pd.DataFrame(
        {
            "outcome": [outcome],
            "treatment": [treatment],
            "spec": [spec],
            "cluster_var": [cluster_var],
            "n_obs": [n_obs],
            "n_clusters": [n_clusters],
            "tau_observed": [tau_obs],
            "cluster_robust_se": [se_cluster],
            "permutation_p_value": [p_value],
            "placebo_mean": [placebo_mean],
            "placebo_sd": [placebo_sd],
            "B": [B],
            "seed": [seed],
        }
    )
    results.to_csv(output_path, index=False)


def save_draws(output_path: Path, placebo_taus: np.ndarray) -> None:
    pd.DataFrame({"draw_id": np.arange(1, len(placebo_taus) + 1), "placebo_tau": placebo_taus}).to_csv(
        output_path, index=False
    )


def plot_histogram(
    *,
    placebo_taus: np.ndarray,
    tau_obs: float,
    outcome: str,
    treatment: str,
    spec: str,
    cluster_var: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    plt.hist(placebo_taus, bins=40, alpha=0.75)
    plt.axvline(tau_obs, color="red", linestyle="--", linewidth=2)
    plt.title(f"Spatial placebo: {treatment} → {outcome} | {spec} | cluster={cluster_var}")
    plt.xlabel("Placebo tau")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()

    root = find_repo_root()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = root / input_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    safe_mkdir(output_dir)

    df = pd.read_csv(input_path)

    sample = build_analysis_sample(
        df,
        outcome=args.outcome,
        treatment=args.treatment,
        spec=args.spec,
        cluster_var=args.cluster_var,
        weight_var=args.weight_var,
    )

    y = sample[args.outcome].to_numpy(dtype=float)
    t = sample[args.treatment].to_numpy(dtype=float)
    w = sample[args.weight_var].to_numpy(dtype=float)
    clusters = sample[args.cluster_var].astype(str)
    x_cov = build_covariate_matrix(sample, spec=args.spec)

    tau_obs, se_cluster, _ = estimate_lpm_cluster(
        y=y,
        treatment_values=t,
        x_cov=x_cov,
        w=w,
        clusters=clusters.to_numpy(),
        treatment_name=args.treatment,
    )

    placebo_taus = spatial_placebo(
        y=y,
        treatment=t,
        x_cov=x_cov,
        weights=w,
        clusters=clusters,
        treatment_name=args.treatment,
        B=args.B,
        seed=args.seed,
    )

    p_value = float(np.mean(np.abs(placebo_taus) >= np.abs(tau_obs)))
    placebo_mean = float(np.mean(placebo_taus))
    placebo_sd = float(np.std(placebo_taus, ddof=1))

    stem = f"step6_spatial_placebo__{args.treatment}__{args.outcome}__{args.spec}__{args.cluster_var}"
    results_path = output_dir / f"{stem}_results.csv"
    draws_path = output_dir / f"{stem}_draws.csv"
    plot_path = output_dir / f"{stem}_hist.png"

    save_results(
        results_path,
        outcome=args.outcome,
        treatment=args.treatment,
        spec=args.spec,
        cluster_var=args.cluster_var,
        n_obs=len(sample),
        n_clusters=int(clusters.nunique()),
        tau_obs=tau_obs,
        se_cluster=se_cluster,
        p_value=p_value,
        placebo_mean=placebo_mean,
        placebo_sd=placebo_sd,
        B=args.B,
        seed=args.seed,
    )
    save_draws(draws_path, placebo_taus)
    plot_histogram(
        placebo_taus=placebo_taus,
        tau_obs=tau_obs,
        outcome=args.outcome,
        treatment=args.treatment,
        spec=args.spec,
        cluster_var=args.cluster_var,
        output_path=plot_path,
    )

    print(f"Observed tau: {tau_obs:.6f}")
    print(f"Cluster-robust SE: {se_cluster:.6f}")
    print(f"Placebo p-value: {p_value:.6f}")
    print(f"Placebo mean: {placebo_mean:.6f}")
    print(f"Placebo sd: {placebo_sd:.6f}")
    print(f"Results saved to: {results_path}")
    print(f"Draws saved to: {draws_path}")
    print(f"Histogram saved to: {plot_path}")


def run_all() -> None:
    """Run spatial placebos for all outcome × cluster combinations used in the thesis."""
    import sys

    root = find_repo_root()
    outcomes = ["y_afd_vote", "y_afd_vote_all_valid"]
    specs = ["spec_a_plus"]
    clusters = ["land", "xs11"]

    jobs = [
        {"outcome": o, "spec": s, "cluster_var": c}
        for o in outcomes for s in specs for c in clusters
    ]

    print(f"Found {len(jobs)} placebo jobs.\n")

    for i, job in enumerate(jobs, start=1):
        print(f"[{i}/{len(jobs)}] outcome={job['outcome']}, spec={job['spec']}, cluster={job['cluster_var']}")
        sys.argv = [
            __file__,
            "--input", DEFAULT_INPUT,
            "--output-dir", DEFAULT_OUTPUT_DIR,
            "--outcome", job["outcome"],
            "--treatment", "east_youth",
            "--spec", job["spec"],
            "--cluster-var", job["cluster_var"],
            "--B", "1000",
            "--seed", "12345",
        ]
        main()
        print("-" * 80 + "\n")

    print(f"All placebo runs completed. Outputs: {root / DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    import sys
    if "--all" in sys.argv:
        sys.argv.remove("--all")
        run_all()
    else:
        main()
