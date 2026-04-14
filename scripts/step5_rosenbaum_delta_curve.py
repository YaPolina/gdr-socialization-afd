#!/usr/bin/env python3
"""
Rosenbaum-style sensitivity analysis: for each candidate unobserved
confounder, compute the tau(delta) curve and tipping-point delta*.
"""

from __future__ import annotations

from pathlib import Path
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


# -----------------------------
# Project defaults
# -----------------------------
DEFAULT_OUTCOME = "y_afd_vote"
DEFAULT_TREATMENT = "east_youth"
DEFAULT_SPEC = "spec_a_plus"
DEFAULT_EFFECT_COL = "ate_aipw"
DEFAULT_CLUSTER_VAR = "xs11"
DEFAULT_WEIGHT_VAR = "wghtpew"
DEFAULT_DELTA_MIN = -2.0
DEFAULT_DELTA_MAX = 2.0
DEFAULT_N_GRID = 401
DEFAULT_SIGN = "minus"
DEFAULT_MEDIATOR_CANDIDATES = ["df44", "dn05", "dn07", "no_denomination", "ru01", "rb07", "rp01", "gs01"]

SPECS: dict[str, list[str]] = {
    "spec_a": ["yborn", "sex", "feseg_grp", "meseg_grp"],
    "spec_a_plus": ["yborn", "sex", "feseg_grp", "meseg_grp", "feduc", "meduc", "fde01", "mde01"],
}
NUMERIC_X = {"yborn"}


# -----------------------------
# Helpers
# -----------------------------
def find_repo_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path(__file__).resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_first_existing(root: Path, candidates: list[str]) -> Path | None:
    for rel in candidates:
        p = root / rel
        if p.exists():
            return p
    return None


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if m.sum() == 0:
        return float("nan")
    return float(np.sum(x[m] * w[m]) / np.sum(w[m]))


def weighted_sd(x: np.ndarray, w: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if m.sum() <= 1:
        return float("nan")
    mu = np.sum(w[m] * x[m]) / np.sum(w[m])
    var = np.sum(w[m] * (x[m] - mu) ** 2) / np.sum(w[m])
    return float(np.sqrt(var))


def pretty_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(s))


# -----------------------------
# Data prep
# -----------------------------
def load_step1(root: Path) -> pd.DataFrame:
    path = find_first_existing(root, [
        "data/derived/step1_model_afd.csv",
        "outputs/derived/step1_model_afd.csv",
    ])
    if path is None:
        raise FileNotFoundError("Could not find data/derived/step1_model_afd.csv")
    return pd.read_csv(path)


def build_sample(df: pd.DataFrame, outcome: str, treatment: str, spec: str, mediator_col: str, weight_var: str, cluster_var: str) -> pd.DataFrame:
    needed = [outcome, treatment, mediator_col, weight_var, cluster_var] + SPECS[spec]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = df[needed].copy()
    out[outcome] = pd.to_numeric(out[outcome], errors="coerce")
    out[treatment] = pd.to_numeric(out[treatment], errors="coerce")
    out[weight_var] = pd.to_numeric(out[weight_var], errors="coerce")
    out[mediator_col] = pd.to_numeric(out[mediator_col], errors="coerce")
    if "yborn" in out.columns:
        out["yborn"] = pd.to_numeric(out["yborn"], errors="coerce")

    mask = (
        out[outcome].isin([0, 1])
        & out[treatment].isin([0, 1])
        & out[weight_var].notna()
        & (out[weight_var] > 0)
        & out[cluster_var].notna()
        & out[mediator_col].notna()
    )
    for x in SPECS[spec]:
        mask &= out[x].notna()

    out = out.loc[mask].copy()
    if out.empty:
        raise ValueError(f"Empty sample after filtering for mediator {mediator_col}")

    # Standardize mediator to weighted SD units to make delta interpretable.
    s = out[mediator_col].to_numpy(dtype=float)
    w = out[weight_var].to_numpy(dtype=float)
    mu_s = weighted_mean(s, w)
    sd_s = weighted_sd(s, w)
    if not np.isfinite(sd_s) or sd_s <= 0:
        raise ValueError(f"Mediator {mediator_col} has zero or undefined weighted SD")
    out["mediator_std"] = (out[mediator_col] - mu_s) / sd_s
    out[cluster_var] = out[cluster_var].astype(str)
    return out


def build_x_matrix(df: pd.DataFrame, spec: str) -> pd.DataFrame:
    x = df[SPECS[spec]].copy()
    cat_cols = [c for c in SPECS[spec] if c not in NUMERIC_X]
    for c in cat_cols:
        x[c] = x[c].astype(object).where(~x[c].isna(), "__MISSING__").astype(str)
    x = pd.get_dummies(x, columns=cat_cols, drop_first=True, dtype=float)
    x = sm.add_constant(x, has_constant="add")
    return x.astype(float)


# -----------------------------
# Tau extraction
# -----------------------------
def try_read_tau_from_step2(root: Path, outcome: str, spec: str, effect_col: str) -> float | None:
    path = find_first_existing(root, [
        "outputs/tables/step2_ate_overlap_eastyouth_results.csv",
    ])
    if path is None:
        return None

    try:
        res = pd.read_csv(path)
    except Exception:
        return None

    if effect_col not in res.columns or "spec" not in res.columns:
        return None

    cand = res.loc[res["spec"].astype(str) == spec].copy()
    if "outcome" in cand.columns:
        cand = cand.loc[cand["outcome"].astype(str) == outcome].copy()
    if cand.empty:
        return None
    return float(pd.to_numeric(cand.iloc[0][effect_col], errors="coerce"))


def estimate_tau_fallback(sample: pd.DataFrame, outcome: str, treatment: str, spec: str, weight_var: str, cluster_var: str) -> tuple[float, float]:
    y = sample[outcome].to_numpy(dtype=float)
    t = sample[treatment].to_numpy(dtype=float)
    w = sample[weight_var].to_numpy(dtype=float)
    X = build_x_matrix(sample, spec)
    X.insert(1, treatment, t)
    model = sm.WLS(y, X, weights=w).fit(cov_type="cluster", cov_kwds={"groups": sample[cluster_var]})
    return float(model.params[treatment]), float(model.bse[treatment])


def try_read_tau_se_from_step4(root: Path, spec: str, effect_col: str, cluster_var: str) -> float | None:
    path = find_first_existing(root, [
        "outputs/tables/step4_inference_multicluster_eastyouth_results.csv",
        "outputs/tables/step4_inference_xs11_results.csv",
    ])
    if path is None:
        return None
    try:
        res = pd.read_csv(path)
    except Exception:
        return None

    if "spec" not in res.columns:
        return None

    estimand = "aipw_ate" if effect_col == "ate_aipw" else "ow_ato"
    cand = res.loc[res["spec"].astype(str) == spec].copy()
    if "estimand" in cand.columns:
        cand = cand.loc[cand["estimand"].astype(str) == estimand].copy()
    if cand.empty:
        return None

    preferred_se_cols = [
        f"se_cluster_{cluster_var}",
        "se_cluster_bootstrap",
        "se_cluster_xs11",
        "cluster_robust_se",
    ]
    for col in preferred_se_cols:
        if col in cand.columns:
            val = float(pd.to_numeric(cand.iloc[0][col], errors="coerce"))
            if np.isfinite(val):
                return val
    return None


# -----------------------------
# Gamma estimation and curve
# -----------------------------
def estimate_gamma(sample: pd.DataFrame, outcome: str, treatment: str, spec: str, weight_var: str, cluster_var: str) -> tuple[float, float, object]:
    y = sample[outcome].to_numpy(dtype=float)
    t = sample[treatment].to_numpy(dtype=float)
    s = sample["mediator_std"].to_numpy(dtype=float)
    w = sample[weight_var].to_numpy(dtype=float)
    X = build_x_matrix(sample, spec)
    X.insert(1, treatment, t)
    X.insert(2, "mediator_std", s)
    model = sm.WLS(y, X, weights=w).fit(cov_type="cluster", cov_kwds={"groups": sample[cluster_var]})
    gamma_hat = float(model.params["mediator_std"])
    gamma_se = float(model.bse["mediator_std"])
    return gamma_hat, gamma_se, model


def build_curve(tau_hat: float, tau_se: float | None, gamma_hat: float, gamma_se: float, delta_min: float, delta_max: float, n_grid: int, sign_convention: str) -> tuple[pd.DataFrame, float | None]:
    delta = np.linspace(delta_min, delta_max, n_grid)
    if sign_convention == "minus":
        tau_adj = tau_hat - gamma_hat * delta
        gamma_lo = gamma_hat - 1.959963984540054 * gamma_se
        gamma_hi = gamma_hat + 1.959963984540054 * gamma_se
        tau_gamma_lo = tau_hat - gamma_hi * delta
        tau_gamma_hi = tau_hat - gamma_lo * delta
        delta_star = float(tau_hat / gamma_hat) if np.isfinite(gamma_hat) and abs(gamma_hat) > 1e-12 else None
    else:
        tau_adj = tau_hat + gamma_hat * delta
        gamma_lo = gamma_hat - 1.959963984540054 * gamma_se
        gamma_hi = gamma_hat + 1.959963984540054 * gamma_se
        tau_gamma_lo = tau_hat + gamma_lo * delta
        tau_gamma_hi = tau_hat + gamma_hi * delta
        delta_star = float(-tau_hat / gamma_hat) if np.isfinite(gamma_hat) and abs(gamma_hat) > 1e-12 else None

    out = pd.DataFrame({
        "delta_sd_units": delta,
        "tau_adjusted": tau_adj,
        "tau_gamma_lo": tau_gamma_lo,
        "tau_gamma_hi": tau_gamma_hi,
    })

    if tau_se is not None and np.isfinite(tau_se):
        z = 1.959963984540054
        out["tau_ci_low_fixed"] = out["tau_adjusted"] - z * tau_se
        out["tau_ci_high_fixed"] = out["tau_adjusted"] + z * tau_se

    # Keep delta* only if it lies on or near a meaningful scale.
    if delta_star is not None and not np.isfinite(delta_star):
        delta_star = None
    return out, delta_star


def plot_curve(curve: pd.DataFrame, tau_hat: float, delta_star: float | None, mediator_col: str, outcome: str, spec: str, effect_col: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(curve["delta_sd_units"], curve["tau_adjusted"], linewidth=2, label="tau(delta)")
    if {"tau_gamma_lo", "tau_gamma_hi"}.issubset(curve.columns):
        plt.fill_between(
            curve["delta_sd_units"],
            curve["tau_gamma_lo"],
            curve["tau_gamma_hi"],
            alpha=0.2,
            label="gamma uncertainty band",
        )
    if {"tau_ci_low_fixed", "tau_ci_high_fixed"}.issubset(curve.columns):
        plt.fill_between(
            curve["delta_sd_units"],
            curve["tau_ci_low_fixed"],
            curve["tau_ci_high_fixed"],
            alpha=0.15,
            label="fixed tau CI band",
        )
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.axhline(tau_hat, linestyle=":", linewidth=1)
    if delta_star is not None:
        plt.axvline(delta_star, linestyle="--", linewidth=1)
    plt.xlabel("delta (change in mediator, SD units)")
    plt.ylabel("tau(delta)")
    plt.title(f"Rosenbaum delta curve: {outcome} | {spec} | {effect_col} | S={mediator_col}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    root = find_repo_root()
    out_dir = root / "outputs" / "tables"
    fig_dir = root / "outputs" / "figures"
    safe_mkdir(out_dir)
    safe_mkdir(fig_dir)

    df = load_step1(root)

    # Preferred defaults chosen to match the current main analysis.
    outcome = DEFAULT_OUTCOME
    treatment = DEFAULT_TREATMENT
    spec = DEFAULT_SPEC
    effect_col = DEFAULT_EFFECT_COL
    cluster_var = DEFAULT_CLUSTER_VAR
    weight_var = DEFAULT_WEIGHT_VAR
    sign_convention = DEFAULT_SIGN

    available_mediators = [m for m in DEFAULT_MEDIATOR_CANDIDATES if m in df.columns]
    if not available_mediators:
        raise KeyError(
            f"None of the default mediator candidates were found: {DEFAULT_MEDIATOR_CANDIDATES}"
        )

    combined_rows: list[dict[str, float | int | str | None]] = []

    for mediator_col in available_mediators:
        sample = build_sample(
            df,
            outcome=outcome,
            treatment=treatment,
            spec=spec,
            mediator_col=mediator_col,
            weight_var=weight_var,
            cluster_var=cluster_var,
        )

        tau_hat = try_read_tau_from_step2(root, outcome=outcome, spec=spec, effect_col=effect_col)
        tau_se = try_read_tau_se_from_step4(root, spec=spec, effect_col=effect_col, cluster_var=cluster_var)

        if tau_hat is None or not np.isfinite(tau_hat):
            tau_hat_fallback, tau_se_fallback = estimate_tau_fallback(
                sample,
                outcome=outcome,
                treatment=treatment,
                spec=spec,
                weight_var=weight_var,
                cluster_var=cluster_var,
            )
            tau_hat = tau_hat_fallback
            if tau_se is None or not np.isfinite(tau_se):
                tau_se = tau_se_fallback
            tau_source = "fallback_wls"
        else:
            tau_source = "step3_results"
            if tau_se is None or not np.isfinite(tau_se):
                tau_source += "_no_step4_se"

        gamma_hat, gamma_se, gamma_model = estimate_gamma(
            sample,
            outcome=outcome,
            treatment=treatment,
            spec=spec,
            weight_var=weight_var,
            cluster_var=cluster_var,
        )

        # Save full regression table for this mediator (advisor requires transparency)
        reg_table = pd.DataFrame({
            "variable": gamma_model.params.index.tolist(),
            "coefficient": gamma_model.params.values,
            "std_error": gamma_model.bse.values,
            "t_stat": gamma_model.tvalues.values,
            "p_value": gamma_model.pvalues.values,
            "ci_low": gamma_model.conf_int()[0].values,
            "ci_high": gamma_model.conf_int()[1].values,
        })
        reg_table.insert(0, "mediator_col", mediator_col)
        reg_table.insert(0, "spec", spec)
        reg_table.insert(0, "outcome", outcome)
        reg_table_path = out_dir / f"step5_sensitivity_regression__{pretty_slug(mediator_col)}.csv"
        reg_table.to_csv(reg_table_path, index=False)

        # Also save model diagnostics
        model_diag = pd.DataFrame([{
            "outcome": outcome,
            "spec": spec,
            "mediator_col": mediator_col,
            "cluster_var": cluster_var,
            "n_obs": int(gamma_model.nobs),
            "n_clusters": int(getattr(gamma_model, "ngroups", 0)) if hasattr(gamma_model, "ngroups") else np.nan,
            "r_squared": float(gamma_model.rsquared),
            "r_squared_adj": float(gamma_model.rsquared_adj),
            "f_statistic": float(gamma_model.fvalue) if np.isfinite(gamma_model.fvalue) else np.nan,
            "f_pvalue": float(gamma_model.f_pvalue) if np.isfinite(gamma_model.f_pvalue) else np.nan,
            "cov_type": str(gamma_model.cov_type),
        }])
        model_diag_path = out_dir / f"step5_sensitivity_model_diag__{pretty_slug(mediator_col)}.csv"
        model_diag.to_csv(model_diag_path, index=False)

        curve, delta_star = build_curve(
            tau_hat=tau_hat,
            tau_se=tau_se,
            gamma_hat=gamma_hat,
            gamma_se=gamma_se,
            delta_min=DEFAULT_DELTA_MIN,
            delta_max=DEFAULT_DELTA_MAX,
            n_grid=DEFAULT_N_GRID,
            sign_convention=sign_convention,
        )

        z = 1.959963984540054
        summary_row = {
            "outcome": outcome,
            "treatment": treatment,
            "spec": spec,
            "effect_col": effect_col,
            "cluster_var": cluster_var,
            "mediator_col": mediator_col,
            "n": int(len(sample)),
            "tau_hat": float(tau_hat),
            "tau_se": float(tau_se) if tau_se is not None and np.isfinite(tau_se) else np.nan,
            "tau_ci_low": float(tau_hat - z * tau_se) if tau_se is not None and np.isfinite(tau_se) else np.nan,
            "tau_ci_high": float(tau_hat + z * tau_se) if tau_se is not None and np.isfinite(tau_se) else np.nan,
            "gamma_hat": float(gamma_hat),
            "gamma_se": float(gamma_se),
            "gamma_ci_low": float(gamma_hat - z * gamma_se),
            "gamma_ci_high": float(gamma_hat + z * gamma_se),
            "delta_star": float(delta_star) if delta_star is not None and np.isfinite(delta_star) else np.nan,
            "delta_units": "mediator SD units",
            "tau_source": tau_source,
            "sign_convention": sign_convention,
        }
        combined_rows.append(summary_row)

        stem = (
            f"step5_rosenbaum_delta_curve__{pretty_slug(outcome)}__{pretty_slug(spec)}__"
            f"{pretty_slug(effect_col)}__{pretty_slug(cluster_var)}__S_{pretty_slug(mediator_col)}"
        )
        curve_path = out_dir / f"{stem}_curve.csv"
        summary_path = out_dir / f"{stem}_summary.csv"
        plot_path = fig_dir / f"{stem}.png"

        curve.to_csv(curve_path, index=False)
        pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
        plot_curve(
            curve=curve,
            tau_hat=float(tau_hat),
            delta_star=float(delta_star) if delta_star is not None and np.isfinite(delta_star) else None,
            mediator_col=mediator_col,
            outcome=outcome,
            spec=spec,
            effect_col=effect_col,
            output_path=plot_path,
        )

        print(f"[ok] mediator={mediator_col}")
        print(f"     tau_hat   = {tau_hat:.6f}")
        print(f"     gamma_hat = {gamma_hat:.6f}")
        print(f"     delta*    = {summary_row['delta_star']}")
        print(f"     wrote     = {summary_path}")
        print(f"     wrote     = {curve_path}")
        print(f"     wrote     = {plot_path}")
        print(f"     wrote     = {reg_table_path}")
        print(f"     wrote     = {model_diag_path}")
        print()

    combined = pd.DataFrame(combined_rows)
    combined_path = out_dir / "step5_rosenbaum_delta_curve_summary_all.csv"
    combined.to_csv(combined_path, index=False)
    print(f"[ok] wrote combined summary: {combined_path}")
    print("\nCombined summary:")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
