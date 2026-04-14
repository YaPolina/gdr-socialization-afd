from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ID_VAR = "respid"
TREATMENT_VALUE_VAR = "treatment_value"
WEIGHTINGS = {
    "raw_unweighted": None,
    "survey_only": "wghtpew",
    "ipw_total": "ipw_total_w",
    "ow_total": "ow_total_w",
}
NUMERIC_VARS = ["yborn", "inc"]
CATEGORICAL_VARS = ["sex", "feseg_grp", "meseg_grp", "feduc", "meduc", "fde01", "mde01", "educ"]


def find_repo_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path(__file__).resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path(__file__).resolve().parents[1]


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _clean_weight(series: pd.Series | None, n: int) -> np.ndarray:
    if series is None:
        return np.ones(n, dtype=float)
    w = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(w) & (w > 0), w, 0.0)


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sum(w[mask] * x[mask]) / np.sum(w[mask]))


def weighted_var(x: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return float("nan")
    x_m = x[mask]
    w_m = w[mask]
    mu = np.sum(w_m * x_m) / np.sum(w_m)
    return float(np.sum(w_m * (x_m - mu) ** 2) / np.sum(w_m))


def smd_continuous(x: np.ndarray, t: np.ndarray, w: np.ndarray) -> float:
    treated = t == 1
    control = t == 0

    mu1 = weighted_mean(x[treated], w[treated])
    mu0 = weighted_mean(x[control], w[control])
    v1 = weighted_var(x[treated], w[treated])
    v0 = weighted_var(x[control], w[control])

    denom = np.sqrt((v1 + v0) / 2.0)
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float((mu1 - mu0) / denom)


def smd_binary(x: np.ndarray, t: np.ndarray, w: np.ndarray) -> float:
    treated = t == 1
    control = t == 0

    p1 = weighted_mean(x[treated], w[treated])
    p0 = weighted_mean(x[control], w[control])
    v1 = p1 * (1 - p1) if np.isfinite(p1) else np.nan
    v0 = p0 * (1 - p0) if np.isfinite(p0) else np.nan

    denom = np.sqrt((v1 + v0) / 2.0)
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float((p1 - p0) / denom)


def level_share(x: np.ndarray, t: np.ndarray, w: np.ndarray, level: str) -> tuple[float, float]:
    indicator = (x == level).astype(float)
    return weighted_mean(indicator[t == 1], w[t == 1]), weighted_mean(indicator[t == 0], w[t == 0])


def combo_key(df: pd.DataFrame) -> dict[str, str]:
    return {
        "treatment": str(df["treatment"].iloc[0]),
        "outcome": str(df["outcome"].iloc[0]),
        "spec": str(df["spec"].iloc[0]),
    }


def compute_diagnostics_for_combo(df_combo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | str]] = []
    summary_rows: list[dict[str, float | str]] = []

    key = combo_key(df_combo)
    t = pd.to_numeric(df_combo[TREATMENT_VALUE_VAR], errors="coerce").to_numpy(dtype=float)

    for weighting_name, weight_col in WEIGHTINGS.items():
        w = _clean_weight(df_combo[weight_col] if weight_col is not None else None, len(df_combo))

        for var in NUMERIC_VARS:
            if var not in df_combo.columns:
                continue
            x = pd.to_numeric(df_combo[var], errors="coerce").to_numpy(dtype=float)
            smd = smd_continuous(x, t, w)
            treated_mean = weighted_mean(x[t == 1], w[t == 1])
            control_mean = weighted_mean(x[t == 0], w[t == 0])
            rows.append(
                {
                    **key,
                    "weighting": weighting_name,
                    "variable": var,
                    "variable_level": "__numeric__",
                    "variable_display": var,
                    "treated_mean": treated_mean,
                    "control_mean": control_mean,
                    "smd": smd,
                    "abs_smd": abs(smd) if np.isfinite(smd) else np.nan,
                }
            )

        for var in CATEGORICAL_VARS:
            if var not in df_combo.columns:
                continue
            x = df_combo[var].map(lambda v: "__MISSING__" if pd.isna(v) else str(v)).to_numpy(dtype=object)
            levels = sorted(pd.Series(x).dropna().unique().tolist())
            for level in levels:
                indicator = (x == level).astype(float)
                smd = smd_binary(indicator, t, w)
                treated_mean, control_mean = level_share(x, t, w, level)
                rows.append(
                    {
                        **key,
                        "weighting": weighting_name,
                        "variable": var,
                        "variable_level": level,
                        "variable_display": f"{var}={level}",
                        "treated_mean": treated_mean,
                        "control_mean": control_mean,
                        "smd": smd,
                        "abs_smd": abs(smd) if np.isfinite(smd) else np.nan,
                    }
                )

        current = pd.DataFrame([r for r in rows if all(r[k] == key[k] for k in key) and r["weighting"] == weighting_name])
        summary_rows.append(
            {
                **key,
                "weighting": weighting_name,
                "n_rows": int(len(df_combo)),
                "max_abs_smd": float(current["abs_smd"].max()) if len(current) else np.nan,
                "mean_abs_smd": float(current["abs_smd"].mean()) if len(current) else np.nan,
                "n_terms": int(len(current)),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def main() -> None:
    root = find_repo_root()

    in_path = root / "data" / "derived" / "step2_ate_overlap_eastyouth_individual.csv"
    out_tables = root / "outputs" / "tables"

    safe_mkdir(out_tables)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path}")

    df = pd.read_csv(in_path)
    combo_cols = ["treatment", "outcome", "spec"]
    missing = [c for c in combo_cols + [TREATMENT_VALUE_VAR] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    detail_list: list[pd.DataFrame] = []
    summary_list: list[pd.DataFrame] = []

    combos = (
        df[combo_cols]
        .dropna()
        .drop_duplicates()
        .sort_values(combo_cols)
        .to_dict(orient="records")
    )

    for combo in combos:
        mask = np.ones(len(df), dtype=bool)
        for col in combo_cols:
            mask &= df[col].astype(str) == str(combo[col])
        df_combo = df.loc[mask].copy()
        detail_df, summary_df = compute_diagnostics_for_combo(df_combo)
        detail_list.append(detail_df)
        summary_list.append(summary_df)

    detail = pd.concat(detail_list, ignore_index=True)
    summary = pd.concat(summary_list, ignore_index=True)

    detail_path = out_tables / "step3_balance_diagnostics_detail.csv"
    summary_path = out_tables / "step3_balance_diagnostics_summary.csv"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"[ok] wrote: {detail_path}")
    print(f"[ok] wrote: {summary_path}")
    print("\nBalance summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
