#!/usr/bin/env python3
"""
step9_thesis_tables.py

Produce consolidated thesis-ready tables from existing pipeline outputs.

Tables produced:
1. Main results table (Table 1): ATE and ATO by outcome × spec, with clustered CIs
2. Balance summary table (Table 2): SMD before/after weighting
3. Sensitivity summary table (Table 3): Rosenbaum delta-star for each mediator
4. Spatial placebo summary (Table 4): observed vs. placebo distribution
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def find_repo_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path(__file__).resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def round_or_na(val, digits=4):
    if pd.isna(val) or not np.isfinite(val):
        return ""
    return round(val, digits)


def build_main_results_table(tables_dir: Path) -> pd.DataFrame | None:
    """
    Table 1: Main results.

    Columns: outcome, spec, n, ATE (AIPW), SE, CI,
             ATO (overlap-weighted), SE, CI,
             clustered CI (xs11), clustered CI (land)
    """
    ate_path = tables_dir / "step2_ate_overlap_eastyouth_results.csv"
    cluster_path = tables_dir / "step4_inference_multicluster_eastyouth_results.csv"

    if not ate_path.exists():
        print(f"[skip] Missing {ate_path}")
        return None

    ate = pd.read_csv(ate_path)
    cluster = pd.read_csv(cluster_path) if cluster_path.exists() else None

    rows = []
    for _, r in ate.iterrows():
        outcome = r["outcome"]
        spec = r["spec"]

        row = {
            "outcome": outcome,
            "spec": spec,
            "n": int(r["n"]),
            "ate_aipw": round_or_na(r["ate_aipw"]),
            "se_aipw": round_or_na(r["se_aipw"]),
            "ci_aipw": f"[{round_or_na(r['ci_aipw_lo'])}, {round_or_na(r['ci_aipw_hi'])}]",
            "ato_ow": round_or_na(r["ate_ow"]),
            "se_ow": round_or_na(r["se_ow"]),
            "ci_ow": f"[{round_or_na(r['ci_ow_lo'])}, {round_or_na(r['ci_ow_hi'])}]",
        }

        # Add clustered CIs from step4
        if cluster is not None:
            for cluster_var in ["xs11", "land"]:
                for estimand_key, prefix in [("aipw_ate", "ate"), ("ow_ato", "ato")]:
                    mask = (
                        (cluster["outcome"] == outcome)
                        & (cluster["spec"] == spec)
                        & (cluster["estimand"] == estimand_key)
                        & (cluster["cluster_var"] == cluster_var)
                    )
                    match = cluster.loc[mask]
                    if len(match) > 0:
                        m = match.iloc[0]
                        row[f"{prefix}_ci_cluster_{cluster_var}"] = (
                            f"[{round_or_na(m['ci_cluster_low'])}, {round_or_na(m['ci_cluster_high'])}]"
                        )

        rows.append(row)

    return pd.DataFrame(rows)


def build_balance_summary_table(tables_dir: Path) -> pd.DataFrame | None:
    """
    Table 2: Balance summary showing max/mean SMD before and after weighting.
    """
    path = tables_dir / "step3_balance_diagnostics_summary.csv"
    if not path.exists():
        print(f"[skip] Missing {path}")
        return None

    bal = pd.read_csv(path)

    rows = []
    for (outcome, spec), grp in bal.groupby(["outcome", "spec"]):
        row = {"outcome": outcome, "spec": spec}
        for _, r in grp.iterrows():
            w = r["weighting"]
            row[f"max_smd_{w}"] = round_or_na(r["max_abs_smd"])
            row[f"mean_smd_{w}"] = round_or_na(r["mean_abs_smd"])
        rows.append(row)

    return pd.DataFrame(rows)


def build_sensitivity_summary_table(tables_dir: Path) -> pd.DataFrame | None:
    """
    Table 3: Rosenbaum sensitivity delta-star summary.
    """
    path = tables_dir / "step5_rosenbaum_delta_curve_summary_all.csv"
    if not path.exists():
        print(f"[skip] Missing {path}")
        return None

    sens = pd.read_csv(path)
    cols = [
        "mediator_col", "n", "tau_hat", "tau_se", "tau_ci_low", "tau_ci_high",
        "gamma_hat", "gamma_se", "gamma_ci_low", "gamma_ci_high",
        "delta_star",
    ]
    available = [c for c in cols if c in sens.columns]
    return sens[available].copy()


def build_placebo_summary_table(placebo_dir: Path) -> pd.DataFrame | None:
    """
    Table 4: Spatial placebo summary.
    """
    import glob
    result_files = sorted(placebo_dir.glob("*_results.csv"))
    if not result_files:
        print(f"[skip] No placebo result files in {placebo_dir}")
        return None

    dfs = [pd.read_csv(f) for f in result_files]
    combined = pd.concat(dfs, ignore_index=True)

    cols = [
        "outcome", "cluster_var", "n_obs", "n_clusters",
        "tau_observed", "cluster_robust_se",
        "permutation_p_value", "placebo_mean", "placebo_sd", "B",
    ]
    available = [c for c in cols if c in combined.columns]
    return combined[available].copy()


def main() -> None:
    root = find_repo_root()
    tables_dir = root / "outputs" / "tables"
    placebo_dir = root / "outputs" / "tables" / "placebo"
    thesis_dir = root / "outputs" / "tables"

    # Table 1: Main results
    t1 = build_main_results_table(tables_dir)
    if t1 is not None:
        t1.to_csv(thesis_dir / "table1_main_results.csv", index=False)
        print(f"[ok] Table 1: Main results ({len(t1)} rows)")
        print(t1.to_string(index=False))
        print()

    # Table 2: Balance
    t2 = build_balance_summary_table(tables_dir)
    if t2 is not None:
        t2.to_csv(thesis_dir / "table2_balance_summary.csv", index=False)
        print(f"[ok] Table 2: Balance summary ({len(t2)} rows)")
        print(t2.to_string(index=False))
        print()

    # Table 3: Sensitivity
    t3 = build_sensitivity_summary_table(tables_dir)
    if t3 is not None:
        t3.to_csv(thesis_dir / "table3_sensitivity_summary.csv", index=False)
        print(f"[ok] Table 3: Sensitivity ({len(t3)} rows)")
        print(t3.to_string(index=False))
        print()

    # Table 4: Spatial placebo
    t4 = build_placebo_summary_table(placebo_dir)
    if t4 is not None:
        t4.to_csv(thesis_dir / "table4_placebo_summary.csv", index=False)
        print(f"[ok] Table 4: Placebo ({len(t4)} rows)")
        print(t4.to_string(index=False))
        print()

    print(f"\n[done] All thesis tables written to: {thesis_dir}")


if __name__ == "__main__":
    main()
