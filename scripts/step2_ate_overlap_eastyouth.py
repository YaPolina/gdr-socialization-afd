from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TREATMENTS = ["east_youth"]
DEFAULT_OUTCOMES = ["y_afd_vote", "y_afd_vote_all_valid", "would_not_vote"]
WEIGHT_VAR = "wghtpew"
CLUSTER_VARS = ["xs11", "land"]
ID_VAR = "respid"
BASE_X_VARS = ["yborn", "sex", "feseg_grp", "meseg_grp", "feduc", "meduc", "fde01", "mde01", "educ", "inc", "feseg", "meseg"]

SPECS: dict[str, list[str]] = {
    "spec_a": ["yborn", "sex", "feseg_grp", "meseg_grp"],
    "spec_a_plus": ["yborn", "sex", "feseg_grp", "meseg_grp", "feduc", "meduc", "fde01", "mde01"],
    "spec_b_ses": ["yborn", "sex", "feseg_grp", "meseg_grp", "feduc", "meduc", "fde01", "mde01", "educ", "inc"],
    "spec_c_granular": ["yborn", "sex", "feseg_grp", "meseg_grp", "feseg", "meseg", "feduc", "meduc", "fde01", "mde01"],
}


def find_repo_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path(__file__).resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path(__file__).resolve().parents[1]


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--treatments", nargs="*", default=DEFAULT_TREATMENTS)
    parser.add_argument("--outcomes", nargs="*", default=DEFAULT_OUTCOMES)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def wavg(x: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.average(x[mask], weights=w[mask]))


def ess(weights: np.ndarray) -> float:
    w = weights[np.isfinite(weights) & (weights > 0)]
    if len(w) == 0:
        return float("nan")
    return float((w.sum() ** 2) / np.sum(w**2))


def weighted_group_mean(y: np.ndarray, group_mask: np.ndarray, weights: np.ndarray) -> float:
    m = group_mask & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    if m.sum() == 0:
        return float("nan")
    return float(np.average(y[m], weights=weights[m]))


NUMERIC_X_COLS = {"yborn", "inc"}


def prepare_x(df: pd.DataFrame, x_cols: list[str]) -> pd.DataFrame:
    x = df[x_cols].copy()
    for col in x_cols:
        if col in NUMERIC_X_COLS:
            x[col] = pd.to_numeric(x[col], errors="coerce")
            x[col] = x[col].fillna(x[col].median())
        else:
            x[col] = x[col].map(lambda v: "__MISSING__" if pd.isna(v) else str(v)).astype(object)
    return x


def make_preprocessor(x_cols: list[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_cols = [c for c in x_cols if c in NUMERIC_X_COLS]
    categorical_cols = [c for c in x_cols if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline(steps=[("scale", StandardScaler())]), numeric_cols))
    if categorical_cols:
        transformers.append(
            ("cat", Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols)
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_logit_pipeline(x_cols: list[str]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pre = make_preprocessor(x_cols)
    return Pipeline(steps=[("pre", pre), ("clf", LogisticRegression(max_iter=4000, solver="lbfgs"))])


def fit_predict_binary_probability(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_test: pd.DataFrame,
    x_cols: list[str],
) -> np.ndarray:
    y_unique = np.unique(y_train[np.isfinite(y_train)])
    if len(y_unique) < 2:
        constant_prob = wavg(y_train.astype(float), w_train.astype(float))
        return np.full(len(x_test), constant_prob, dtype=float)

    model = make_logit_pipeline(x_cols)
    model.fit(x_train, y_train, clf__sample_weight=w_train)
    return model.predict_proba(x_test)[:, 1]


def run_cross_fitting(
    df: pd.DataFrame,
    *,
    outcome_var: str,
    treatment_var: str,
    x_cols: list[str],
    n_splits: int = 5,
    random_state: int = 42,
    cluster_var: str = "xs11",
) -> pd.DataFrame:
    from sklearn.model_selection import GroupKFold

    out = df.copy()
    X = prepare_x(out, x_cols=x_cols)
    Y = pd.to_numeric(out[outcome_var], errors="coerce").astype(int).to_numpy()
    T = pd.to_numeric(out[treatment_var], errors="coerce").astype(int).to_numpy()
    W = pd.to_numeric(out[WEIGHT_VAR], errors="coerce").astype(float).to_numpy()

    # Use cluster-aware splitting to prevent data leakage within sampling units
    groups = out[cluster_var].astype(str).to_numpy()

    n = len(out)
    e_hat = np.full(n, np.nan, dtype=float)
    m1_hat = np.full(n, np.nan, dtype=float)
    m0_hat = np.full(n, np.nan, dtype=float)

    gkf = GroupKFold(n_splits=n_splits)

    for train_idx, test_idx in gkf.split(X, T, groups=groups):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()

        Y_train = Y[train_idx]
        T_train = T[train_idx]
        W_train = W[train_idx]

        e_hat[test_idx] = fit_predict_binary_probability(
            x_train=X_train,
            y_train=T_train,
            w_train=W_train,
            x_test=X_test,
            x_cols=x_cols,
        )

        treated_mask = T_train == 1
        control_mask = T_train == 0

        m1_hat[test_idx] = fit_predict_binary_probability(
            x_train=X_train.loc[treated_mask],
            y_train=Y_train[treated_mask],
            w_train=W_train[treated_mask],
            x_test=X_test,
            x_cols=x_cols,
        )
        m0_hat[test_idx] = fit_predict_binary_probability(
            x_train=X_train.loc[control_mask],
            y_train=Y_train[control_mask],
            w_train=W_train[control_mask],
            x_test=X_test,
            x_cols=x_cols,
        )

    # Store unclipped propensity scores for diagnostics
    out["e_hat_unclipped"] = e_hat.copy()

    eps = 1e-3
    e_hat = np.clip(e_hat, eps, 1 - eps)
    m1_hat = np.clip(m1_hat, eps, 1 - eps)
    m0_hat = np.clip(m0_hat, eps, 1 - eps)

    psi = (m1_hat - m0_hat) + (T * (Y - m1_hat) / e_hat) - ((1 - T) * (Y - m0_hat) / (1 - e_hat))

    out["e_hat"] = e_hat
    out["m1_hat"] = m1_hat
    out["m0_hat"] = m0_hat
    out["psi_aipw"] = psi

    return out


def compute_overlap_weights(df: pd.DataFrame, treatment_var: str) -> pd.DataFrame:
    out = df.copy()

    t = pd.to_numeric(out[treatment_var], errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(out["e_hat"], errors="coerce").to_numpy(dtype=float)
    survey_w = pd.to_numeric(out[WEIGHT_VAR], errors="coerce").to_numpy(dtype=float)

    ow = np.where(t == 1, 1 - e, e)
    ipw = np.where(t == 1, 1 / e, 1 / (1 - e))

    out["ow_raw"] = ow
    out["ipw_raw"] = ipw
    out["ow_total_w"] = survey_w * ow
    out["ipw_total_w"] = survey_w * ipw

    return out


def compute_results(
    df: pd.DataFrame,
    *,
    treatment_var: str,
    outcome_var: str,
    spec_name: str,
    x_cols: list[str],
) -> dict[str, float | int | str]:
    y = pd.to_numeric(df[outcome_var], errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(df[treatment_var], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(df[WEIGHT_VAR], errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(df["e_hat"], errors="coerce").to_numpy(dtype=float)
    psi = pd.to_numeric(df["psi_aipw"], errors="coerce").to_numpy(dtype=float)

    ow_total_w = pd.to_numeric(df["ow_total_w"], errors="coerce").to_numpy(dtype=float)
    ipw_total_w = pd.to_numeric(df["ipw_total_w"], errors="coerce").to_numpy(dtype=float)

    ate_aipw = wavg(psi, w)
    ow_treated_mean = weighted_group_mean(y, t == 1, ow_total_w)
    ow_control_mean = weighted_group_mean(y, t == 0, ow_total_w)
    ate_ow = ow_treated_mean - ow_control_mean

    # Analytical SE from influence function: SE = sqrt( sum(w^2 * (psi - ate)^2) / (sum(w))^2 )
    w_sum = np.nansum(w[np.isfinite(w) & (w > 0)])
    mask_se = np.isfinite(psi) & np.isfinite(w) & (w > 0)
    se_aipw = float(np.sqrt(np.sum(w[mask_se] ** 2 * (psi[mask_se] - ate_aipw) ** 2) / w_sum ** 2))
    z_975 = 1.959963984540054
    ci_aipw_lo = ate_aipw - z_975 * se_aipw
    ci_aipw_hi = ate_aipw + z_975 * se_aipw

    # OW influence function SE
    s1 = np.sum(ow_total_w[t == 1])
    s0 = np.sum(ow_total_w[t == 0])
    phi_ow = np.full(len(y), np.nan, dtype=float)
    m1 = (t == 1) & np.isfinite(y) & np.isfinite(ow_total_w) & (ow_total_w > 0)
    m0 = (t == 0) & np.isfinite(y) & np.isfinite(ow_total_w) & (ow_total_w > 0)
    if s1 > 0 and s0 > 0:
        phi_ow[m1] = ow_total_w[m1] * (y[m1] - ow_treated_mean) / s1
        phi_ow[m0] = -ow_total_w[m0] * (y[m0] - ow_control_mean) / s0
    mask_ow = np.isfinite(phi_ow)
    se_ow = float(np.sqrt(np.sum(phi_ow[mask_ow] ** 2))) if mask_ow.sum() > 0 else float("nan")
    ci_ow_lo = ate_ow - z_975 * se_ow
    ci_ow_hi = ate_ow + z_975 * se_ow

    # Unclipped propensity score diagnostics
    e_unclipped = pd.to_numeric(df["e_hat_unclipped"], errors="coerce").to_numpy(dtype=float)

    return {
        "treatment": treatment_var,
        "outcome": outcome_var,
        "spec": spec_name,
        "x_vars": ", ".join(x_cols),
        "n": int(len(df)),
        "n_treated": int(np.sum(t == 1)),
        "n_control": int(np.sum(t == 0)),
        "treated_share_unw": float(np.mean(t)),
        "treated_share_w": float(np.sum(w * t) / np.sum(w)),
        "y_mean_w": wavg(y, w),
        "ate_aipw": ate_aipw,
        "se_aipw": se_aipw,
        "ci_aipw_lo": ci_aipw_lo,
        "ci_aipw_hi": ci_aipw_hi,
        "ate_ow": ate_ow,
        "se_ow": se_ow,
        "ci_ow_lo": ci_ow_lo,
        "ci_ow_hi": ci_ow_hi,
        "e_hat_q01": float(np.quantile(e, 0.01)),
        "e_hat_q05": float(np.quantile(e, 0.05)),
        "e_hat_q50": float(np.quantile(e, 0.50)),
        "e_hat_q95": float(np.quantile(e, 0.95)),
        "e_hat_q99": float(np.quantile(e, 0.99)),
        "e_unclipped_q01": float(np.quantile(e_unclipped, 0.01)),
        "e_unclipped_q05": float(np.quantile(e_unclipped, 0.05)),
        "e_unclipped_q50": float(np.quantile(e_unclipped, 0.50)),
        "e_unclipped_q95": float(np.quantile(e_unclipped, 0.95)),
        "e_unclipped_q99": float(np.quantile(e_unclipped, 0.99)),
        "e_unclipped_min": float(np.min(e_unclipped)),
        "e_unclipped_max": float(np.max(e_unclipped)),
        "share_e_lt_001": float(np.mean(e < 0.01)),
        "share_e_lt_005": float(np.mean(e < 0.05)),
        "share_e_gt_095": float(np.mean(e > 0.95)),
        "share_e_gt_099": float(np.mean(e > 0.99)),
        "ess_survey_only": ess(w),
        "ess_ipw_total": ess(ipw_total_w),
        "ess_ow_total": ess(ow_total_w),
    }


def make_individual_output(
    spec_df: pd.DataFrame,
    *,
    treatment_var: str,
    outcome_var: str,
    spec_name: str,
) -> pd.DataFrame:
    keep_cols = [
        ID_VAR,
        WEIGHT_VAR,
        "e_hat",
        "m1_hat",
        "m0_hat",
        "psi_aipw",
        "ow_raw",
        "ipw_raw",
        "ow_total_w",
        "ipw_total_w",
    ]
    for col in CLUSTER_VARS + [treatment_var, outcome_var] + BASE_X_VARS:
        if col in spec_df.columns and col not in keep_cols:
            keep_cols.append(col)

    out = spec_df[keep_cols].copy()
    out.insert(0, "spec", spec_name)
    out.insert(0, "outcome", outcome_var)
    out.insert(0, "treatment", treatment_var)
    out = out.rename(columns={treatment_var: "treatment_value", outcome_var: "outcome_value"})
    return out


def main() -> None:
    args = parse_args()
    root = find_repo_root()

    in_path = Path(args.input) if args.input else root / "data" / "derived" / "step1_model_afd.csv"
    out_dir_tables = root / "outputs" / "tables"
    out_dir_derived = root / "data" / "derived"

    safe_mkdir(out_dir_tables)
    safe_mkdir(out_dir_derived)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path}")

    df = pd.read_csv(in_path)

    missing_base = [c for c in [ID_VAR, WEIGHT_VAR] + CLUSTER_VARS if c not in df.columns]
    if missing_base:
        raise KeyError(f"Missing required base columns in input data: {missing_base}")

    results_rows: list[dict[str, float | int | str]] = []
    individual_rows: list[pd.DataFrame] = []
    sample_rows: list[dict[str, int | str]] = []

    for treatment_var in args.treatments:
        if treatment_var not in df.columns:
            raise KeyError(f"Missing treatment column: {treatment_var}")

        for outcome_var in args.outcomes:
            if outcome_var not in df.columns:
                raise KeyError(f"Missing outcome column: {outcome_var}")

            base_sample_mask = (
                pd.to_numeric(df[outcome_var], errors="coerce").notna()
                & pd.to_numeric(df[treatment_var], errors="coerce").notna()
                & pd.to_numeric(df[WEIGHT_VAR], errors="coerce").notna()
                & (pd.to_numeric(df[WEIGHT_VAR], errors="coerce") > 0)
            )
            df_base = df.loc[base_sample_mask].copy()

            for spec_name, x_cols in SPECS.items():
                missing_x = [col for col in x_cols if col not in df_base.columns]
                if missing_x:
                    raise KeyError(f"Missing X columns for {spec_name}: {missing_x}")

                keep_cols = list(dict.fromkeys([ID_VAR, WEIGHT_VAR] + CLUSTER_VARS + [treatment_var, outcome_var] + BASE_X_VARS))
                spec_df = df_base[keep_cols].copy()

                spec_df = run_cross_fitting(
                    spec_df,
                    outcome_var=outcome_var,
                    treatment_var=treatment_var,
                    x_cols=x_cols,
                    n_splits=args.n_splits,
                    random_state=args.random_state,
                    cluster_var="xs11",
                )
                spec_df = compute_overlap_weights(spec_df, treatment_var=treatment_var)

                results_rows.append(
                    compute_results(
                        spec_df,
                        treatment_var=treatment_var,
                        outcome_var=outcome_var,
                        spec_name=spec_name,
                        x_cols=x_cols,
                    )
                )

                sample_rows.append(
                    {
                        "treatment": treatment_var,
                        "outcome": outcome_var,
                        "spec": spec_name,
                        "n_estimation_sample": int(len(spec_df)),
                        "n_complete_x": int(spec_df[x_cols].notna().all(axis=1).sum()),
                    }
                )

                individual_rows.append(
                    make_individual_output(
                        spec_df,
                        treatment_var=treatment_var,
                        outcome_var=outcome_var,
                        spec_name=spec_name,
                    )
                )

    results = pd.DataFrame(results_rows)
    samples = pd.DataFrame(sample_rows)
    individual = pd.concat(individual_rows, axis=0, ignore_index=True)

    results_path = out_dir_tables / "step2_ate_overlap_eastyouth_results.csv"
    samples_path = out_dir_tables / "step2_ate_overlap_eastyouth_sample_sizes.csv"
    individual_path = out_dir_derived / "step2_ate_overlap_eastyouth_individual.csv"

    results.to_csv(results_path, index=False)
    samples.to_csv(samples_path, index=False)
    individual.to_csv(individual_path, index=False)

    print(f"[ok] wrote: {results_path}")
    print(f"[ok] wrote: {samples_path}")
    print(f"[ok] wrote: {individual_path}")
    print("\nATE and overlap diagnostics:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
