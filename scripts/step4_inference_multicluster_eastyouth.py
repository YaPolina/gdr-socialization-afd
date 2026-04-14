from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ID_VAR = "respid"
WEIGHT_VAR = "wghtpew"
CLUSTER_VARS = ["xs11", "land"]

AIPW_SCORE_VAR = "psi_aipw"
OW_WEIGHT_VAR = "ow_total_w"

N_BOOT = 499
RANDOM_STATE = 42
Z_975 = 1.959963984540054


def find_repo_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path(__file__).resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path(__file__).resolve().parents[1]


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sum(w[mask] * x[mask]) / np.sum(w[mask]))


def cluster_sandwich_se(phi: np.ndarray, cluster: np.ndarray) -> float:
    mask = np.isfinite(phi) & pd.notna(cluster)
    phi = phi[mask]
    cluster = cluster[mask]

    if len(phi) == 0:
        return float("nan")

    cluster_df = pd.DataFrame({"cluster": cluster, "phi": phi})
    cluster_sums = cluster_df.groupby("cluster", dropna=False)["phi"].sum().to_numpy(dtype=float)

    g = len(cluster_sums)
    if g <= 1:
        return float("nan")

    se2 = np.sum(cluster_sums**2)
    correction = g / (g - 1)
    return float(np.sqrt(correction * se2))


def naive_se(phi: np.ndarray) -> float:
    mask = np.isfinite(phi)
    phi = phi[mask]
    if len(phi) == 0:
        return float("nan")
    return float(np.sqrt(np.sum(phi**2)))


def normal_ci(estimate: float, se: float) -> tuple[float, float]:
    if not np.isfinite(estimate) or not np.isfinite(se):
        return (float("nan"), float("nan"))
    return (float(estimate - Z_975 * se), float(estimate + Z_975 * se))


def estimate_aipw(df: pd.DataFrame) -> tuple[float, np.ndarray]:
    psi = pd.to_numeric(df[AIPW_SCORE_VAR], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(df[WEIGHT_VAR], errors="coerce").to_numpy(dtype=float)

    theta = weighted_mean(psi, w)
    denom = np.nansum(w[np.isfinite(w) & (w > 0)])

    phi = np.full(len(df), np.nan, dtype=float)
    mask = np.isfinite(psi) & np.isfinite(w) & (w > 0) & np.isfinite(theta) & (denom > 0)
    phi[mask] = w[mask] * (psi[mask] - theta) / denom

    return theta, phi


def estimate_ow(df: pd.DataFrame) -> tuple[float, np.ndarray]:
    y = pd.to_numeric(df["outcome_value"], errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(df["treatment_value"], errors="coerce").to_numpy(dtype=float)
    ow = pd.to_numeric(df[OW_WEIGHT_VAR], errors="coerce").to_numpy(dtype=float)

    treated_mask = np.isfinite(y) & np.isfinite(t) & np.isfinite(ow) & (ow > 0) & (t == 1)
    control_mask = np.isfinite(y) & np.isfinite(t) & np.isfinite(ow) & (ow > 0) & (t == 0)

    s1 = np.sum(ow[treated_mask])
    s0 = np.sum(ow[control_mask])

    if s1 <= 0 or s0 <= 0:
        return float("nan"), np.full(len(df), np.nan, dtype=float)

    mu1 = np.sum(ow[treated_mask] * y[treated_mask]) / s1
    mu0 = np.sum(ow[control_mask] * y[control_mask]) / s0
    theta = float(mu1 - mu0)

    phi = np.full(len(df), np.nan, dtype=float)
    phi[treated_mask] = ow[treated_mask] * (y[treated_mask] - mu1) / s1
    phi[control_mask] = -ow[control_mask] * (y[control_mask] - mu0) / s0

    return theta, phi


def bootstrap_cluster_estimates(
    df: pd.DataFrame,
    estimate_fn,
    *,
    cluster_var: str,
    n_boot: int = N_BOOT,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    work = df.copy()
    work[cluster_var] = work[cluster_var].astype(str)

    groups = {cid: g.copy() for cid, g in work.groupby(cluster_var, dropna=False)}
    cluster_ids = np.array(list(groups.keys()), dtype=object)
    g = len(cluster_ids)
    if g == 0:
        return np.array([], dtype=float)

    draws = np.full(n_boot, np.nan, dtype=float)
    for b in range(n_boot):
        sampled = rng.choice(cluster_ids, size=g, replace=True)
        boot_df = pd.concat([groups[cid] for cid in sampled], axis=0, ignore_index=True)
        theta, _ = estimate_fn(boot_df)
        draws[b] = theta

    return draws


def bootstrap_ci(draws: np.ndarray) -> tuple[float, float, float]:
    draws = draws[np.isfinite(draws)]
    if len(draws) == 0:
        return (float("nan"), float("nan"), float("nan"))
    se = float(np.std(draws, ddof=1)) if len(draws) > 1 else float("nan")
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return (se, float(lo), float(hi))


def prepare_df(df: pd.DataFrame, *, cluster_var: str, treatment: str, outcome: str, spec: str) -> pd.DataFrame:
    needed = [
        "treatment",
        "outcome",
        "spec",
        ID_VAR,
        cluster_var,
        "outcome_value",
        "treatment_value",
        WEIGHT_VAR,
        AIPW_SCORE_VAR,
        OW_WEIGHT_VAR,
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in step3 file: {missing}")

    out = df.loc[
        (df["treatment"] == treatment) & (df["outcome"] == outcome) & (df["spec"] == spec),
        needed,
    ].copy()

    out["outcome_value"] = pd.to_numeric(out["outcome_value"], errors="coerce")
    out["treatment_value"] = pd.to_numeric(out["treatment_value"], errors="coerce")
    out[WEIGHT_VAR] = pd.to_numeric(out[WEIGHT_VAR], errors="coerce")
    out[AIPW_SCORE_VAR] = pd.to_numeric(out[AIPW_SCORE_VAR], errors="coerce")
    out[OW_WEIGHT_VAR] = pd.to_numeric(out[OW_WEIGHT_VAR], errors="coerce")

    mask = (
        out["outcome_value"].notna()
        & out["treatment_value"].notna()
        & out[WEIGHT_VAR].notna()
        & (out[WEIGHT_VAR] > 0)
        & out[cluster_var].notna()
    )
    out = out.loc[mask].copy()
    out[cluster_var] = out[cluster_var].astype(str)
    return out


def summarize_estimand(
    df: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    spec: str,
    estimand: str,
    cluster_var: str,
    estimate_fn,
    n_boot: int,
    random_state: int,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    estimate, phi = estimate_fn(df)
    clusters = df[cluster_var].astype(str).to_numpy()
    n_clusters = df[cluster_var].nunique(dropna=True)

    se_n = naive_se(phi)
    ci_n_lo, ci_n_hi = normal_ci(estimate, se_n)

    se_c = cluster_sandwich_se(phi, clusters)
    ci_c_lo, ci_c_hi = normal_ci(estimate, se_c)

    boot_draws = bootstrap_cluster_estimates(
        df=df,
        estimate_fn=estimate_fn,
        cluster_var=cluster_var,
        n_boot=n_boot,
        random_state=random_state,
    )
    se_b, ci_b_lo, ci_b_hi = bootstrap_ci(boot_draws)

    result = {
        "treatment": treatment,
        "outcome": outcome,
        "spec": spec,
        "estimand": estimand,
        "cluster_var": cluster_var,
        "estimate": float(estimate),
        "n": int(len(df)),
        "n_clusters": int(n_clusters),
        "se_naive": float(se_n),
        "ci_naive_low": float(ci_n_lo),
        "ci_naive_high": float(ci_n_hi),
        "se_cluster": float(se_c),
        "ci_cluster_low": float(ci_c_lo),
        "ci_cluster_high": float(ci_c_hi),
        "se_cluster_bootstrap": float(se_b),
        "ci_cluster_bootstrap_low": float(ci_b_lo),
        "ci_cluster_bootstrap_high": float(ci_b_hi),
        "n_boot": int(n_boot),
    }

    draws_df = pd.DataFrame(
        {
            "treatment": treatment,
            "outcome": outcome,
            "spec": spec,
            "estimand": estimand,
            "cluster_var": cluster_var,
            "bootstrap_draw_id": np.arange(1, len(boot_draws) + 1),
            "estimate_bootstrap": boot_draws,
        }
    )
    return result, draws_df


def main() -> None:
    root = find_repo_root()

    in_path = root / "data" / "derived" / "step2_ate_overlap_eastyouth_individual.csv"
    out_tables = root / "outputs" / "tables"
    out_derived = root / "data" / "derived"

    safe_mkdir(out_tables)
    safe_mkdir(out_derived)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path}")

    df = pd.read_csv(in_path)
    combo_cols = ["treatment", "outcome", "spec"]
    missing_combo = [c for c in combo_cols if c not in df.columns]
    if missing_combo:
        raise KeyError(f"Missing combo columns in input file: {missing_combo}")

    combos = (
        df[combo_cols]
        .dropna()
        .drop_duplicates()
        .sort_values(combo_cols)
        .to_dict(orient="records")
    )

    results_rows: list[dict[str, float | int | str]] = []
    draws_list: list[pd.DataFrame] = []

    for combo in combos:
        treatment = str(combo["treatment"])
        outcome = str(combo["outcome"])
        spec = str(combo["spec"])

        for cluster_var in CLUSTER_VARS:
            spec_df = prepare_df(df, cluster_var=cluster_var, treatment=treatment, outcome=outcome, spec=spec)

            aipw_result, aipw_draws = summarize_estimand(
                spec_df,
                treatment=treatment,
                outcome=outcome,
                spec=spec,
                estimand="aipw_ate",
                cluster_var=cluster_var,
                estimate_fn=estimate_aipw,
                n_boot=N_BOOT,
                random_state=RANDOM_STATE,
            )
            ow_result, ow_draws = summarize_estimand(
                spec_df,
                treatment=treatment,
                outcome=outcome,
                spec=spec,
                estimand="ow_ato",
                cluster_var=cluster_var,
                estimate_fn=estimate_ow,
                n_boot=N_BOOT,
                random_state=RANDOM_STATE + 1000,
            )

            results_rows.extend([aipw_result, ow_result])
            draws_list.extend([aipw_draws, ow_draws])

    results = pd.DataFrame(results_rows)
    draws = pd.concat(draws_list, ignore_index=True)

    results_path = out_tables / "step4_inference_multicluster_eastyouth_results.csv"
    draws_path = out_derived / "step4_inference_multicluster_eastyouth_bootstrap_draws.csv"

    results.to_csv(results_path, index=False)
    draws.to_csv(draws_path, index=False)

    print(f"[ok] wrote: {results_path}")
    print(f"[ok] wrote: {draws_path}")
    print("\nInference results:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
