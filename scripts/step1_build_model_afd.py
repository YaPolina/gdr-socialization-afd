from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyreadstat


RAW_REQUIRED_VARS = [
    "respid",
    "pv01",
    "dg03",
    "dg10",
    "eastwest",
    "land",
    "bik",
    "xs11",
    "wghtpew",
    "sex",
    "yborn",
    "age",
    "agec",
    "dn05",
    "dn07",
    "fdm01",
    "mdm01",
    "df44",
    "feduc",
    "meduc",
    "fde01",
    "mde01",
    "feseg",
    "meseg",
    "educ",
    "inc",
    "rd01",
    "ru01",
    "rb07",
    "rp01",
    "gs01",
]

TREATMENT_VAR = "east_youth"

SPEC_A_VARS = [
    "yborn",
    "sex",
    "feseg_grp",
    "meseg_grp",
]

SPEC_A_PLUS_VARS = [
    "yborn",
    "sex",
    "feseg_grp",
    "meseg_grp",
    "feduc",
    "meduc",
    "fde01",
    "mde01",
]

DESCRIPTIVE_VARS = [
    "yborn",
    "sex",
    "dn05",
    "dn07",
    "fdm01",
    "mdm01",
    "df44",
]

AUX_VARS = [
    "respid",
    "wghtpew",
    "xs11",
    "eastwest",
    "land",
    "bik",
    "age",
    "agec",
]

GENERAL_MISSING_CODES = {
    -9,
    -8,
    -7,
    -10,
    -11,
    -12,
    -13,
    -14,
    -15,
    -19,
    -20,
    -21,
    -22,
    -23,
    -25,
    -26,
    -27,
    -28,
    -29,
    -30,
    -31,
    -32,
    -33,
    -34,
    -35,
    -36,
    -37,
    -38,
    -39,
    -40,
    -41,
    -42,
    -46,
    -47,
    -48,
    -49,
    -50,
    -51,
    -52,
    -53,
    -54,
    -55,
    -56,
    -57,
    -58,
    -59,
}

DERIVED_LABELS = {
    "party_chooser": "Party chooser indicator",
    "y_afd_vote": "AfD vote intention among party choosers",
    "y_afd_vote_all_valid": "AfD vote intention among all valid vote-intention responses",
    "would_not_vote": "Would not vote indicator",
    "east_youth": "Youth in East Germany indicator",
    "east_interview": "Interview in East Germany indicator",
    "moved_east_west": "Youth region differs from interview region indicator",
    "age_1990": "Age in 1990",
    "feseg_grp": "Father ESeG broad group",
    "meseg_grp": "Mother ESeG broad group",
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


def load_data(path: str | Path) -> tuple[pd.DataFrame, Any]:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".sav", ".zsav"}:
        df, meta = pyreadstat.read_sav(path)
    elif suffix == ".dta":
        df, meta = pyreadstat.read_dta(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return df, meta


def ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in raw data: {missing}")


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_allbus_missing_codes(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Recode ALLBUS special negative codes to missing values for selected variables.
    """
    out = df.copy()

    for col in cols:
        if col not in out.columns:
            continue
        s = coerce_numeric(out[col])
        s = s.mask(s.isin(GENERAL_MISSING_CODES), np.nan)
        out[col] = s

    return out


def build_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pv = out["pv01"]

    party_chooser_codes = {1, 2, 3, 4, 6, 42, 90}
    all_valid_vote_intention_codes = party_chooser_codes | {91}

    out["party_chooser"] = np.where(pv.isin(party_chooser_codes), 1.0, 0.0)
    out.loc[pv.isna(), "party_chooser"] = np.nan

    out["y_afd_vote"] = np.where(
        pv.isin(party_chooser_codes),
        np.where(pv == 42, 1.0, 0.0),
        np.nan,
    )

    out["y_afd_vote_all_valid"] = np.where(
        pv.isin(all_valid_vote_intention_codes),
        np.where(pv == 42, 1.0, 0.0),
        np.nan,
    )

    out["would_not_vote"] = np.where(pv == 91, 1.0, 0.0)
    out.loc[pv.isna(), "would_not_vote"] = np.nan

    return out


def build_treatments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    dg03 = out["dg03"]
    yborn = out["yborn"]

    out["east_youth"] = np.where(
        dg03.isin({1, 2}),
        1.0,
        np.where(dg03.isin({3, 4}), 0.0, np.nan),
    )

    out["east_interview"] = np.where(
        dg03.isin({1, 3}),
        1.0,
        np.where(dg03.isin({2, 4}), 0.0, np.nan),
    )

    out["moved_east_west"] = np.where(
        dg03.isin({2, 3}),
        1.0,
        np.where(dg03.isin({1, 4}), 0.0, np.nan),
    )

    out["age_1990"] = 1990 - yborn

    return out


def collapse_eseg(series: pd.Series) -> pd.Series:
    """
    Collapse detailed ESeG codes into broad socioeconomic groups.
    """
    s = coerce_numeric(series)

    out = pd.Series(pd.NA, index=s.index, dtype="object")

    out.loc[s == 1] = "employed_unknown"
    out.loc[s == 2] = "unemployed_or_never_worked"

    out.loc[s.between(10, 14, inclusive="both")] = "managerial"
    out.loc[s.between(20, 25, inclusive="both")] = "professional"
    out.loc[s.between(30, 35, inclusive="both")] = "associate_professional"
    out.loc[s.between(40, 43, inclusive="both")] = "small_entrepreneur"
    out.loc[s.between(50, 54, inclusive="both")] = "clerical_skilled_service"
    out.loc[s.between(60, 65, inclusive="both")] = "skilled_industrial"
    out.loc[s.between(70, 74, inclusive="both")] = "lower_status_employee"

    out.loc[s == 80] = "retired"
    out.loc[s.isin({90, 99})] = "outside_labor_force"

    return out


def build_parental_eseg_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["feseg_grp"] = collapse_eseg(out["feseg"])
    out["meseg_grp"] = collapse_eseg(out["meseg"])
    return out


def make_variable_audit(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    rows = []

    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "variable": col,
                "dtype": str(s.dtype),
                "missing_share": float(s.isna().mean()),
                "nunique": int(s.nunique(dropna=True)),
                "top_values": str(s.value_counts(dropna=True).head(top_n).to_dict()),
            }
        )

    return pd.DataFrame(rows).sort_values(["missing_share", "variable"]).reset_index(drop=True)


def complete_case_mask(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].notna().all(axis=1)


def main() -> None:
    root = find_repo_root()

    in_path = root / "data" / "raw" / "ALLBUS_2023.dta"
    out_path = root / "data" / "derived" / "step1_model_afd.csv"
    out_tables = root / "outputs" / "tables"

    safe_mkdir(out_path.parent)
    safe_mkdir(out_tables)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path}")

    df_raw, meta = load_data(in_path)
    ensure_columns(df_raw, RAW_REQUIRED_VARS)

    keep_cols = list(dict.fromkeys(RAW_REQUIRED_VARS))
    df = df_raw[keep_cols].copy()

    df = clean_allbus_missing_codes(df, cols=keep_cols)
    df = build_outcomes(df)
    df = build_treatments(df)
    df = build_parental_eseg_groups(df)

    # Derived: no religious denomination (rd01 == 6 in ALLBUS coding = keine Konfession)
    rd01 = pd.to_numeric(df["rd01"], errors="coerce")
    df["no_denomination"] = np.where(rd01 == 6, 1.0, np.where(rd01.notna() & (rd01 > 0), 0.0, np.nan))

    final_cols = [
        "respid",
        "pv01",
        "party_chooser",
        "y_afd_vote",
        "y_afd_vote_all_valid",
        "would_not_vote",
        "dg03",
        "east_youth",
        "east_interview",
        "moved_east_west",
        "dg10",
        "feseg",
        "meseg",
        "feseg_grp",
        "meseg_grp",
        "feduc",
        "meduc",
        "fde01",
        "mde01",
        "yborn",
        "age",
        "agec",
        "age_1990",
        "sex",
        "dn05",
        "dn07",
        "fdm01",
        "mdm01",
        "df44",
        "eastwest",
        "land",
        "bik",
        "xs11",
        "wghtpew",
        "educ",
        "inc",
        "rd01",
        "no_denomination",
        "ru01",
        "rb07",
        "rp01",
        "gs01",
    ]

    df = df[final_cols].copy()
    df.to_csv(out_path, index=False)

    audit = make_variable_audit(df)
    audit_path = out_tables / "step1_model_afd_audit.csv"
    audit.to_csv(audit_path, index=False)

    main_sample_mask = (
        df["y_afd_vote"].notna()
        & df[TREATMENT_VAR].notna()
        & df["wghtpew"].notna()
        & (df["wghtpew"] > 0)
    )

    spec_a_mask = (
        main_sample_mask
        & complete_case_mask(df, SPEC_A_VARS)
    )

    spec_a_plus_mask = (
        main_sample_mask
        & complete_case_mask(df, SPEC_A_PLUS_VARS)
    )

    summary = pd.DataFrame(
        [
            {
                "n_raw": int(len(df)),
                "n_party_choosers": int(df["party_chooser"].fillna(0).sum()),
                "n_valid_treatment": int(df[TREATMENT_VAR].notna().sum()),
                "n_main_sample": int(main_sample_mask.sum()),
                "n_spec_a_complete": int(spec_a_mask.sum()),
                "n_spec_a_plus_complete": int(spec_a_plus_mask.sum()),
                "treated_share_unw_main": float(df.loc[main_sample_mask, TREATMENT_VAR].mean()),
                "afd_share_unw_main": float(df.loc[main_sample_mask, "y_afd_vote"].mean()),
                "weight_min": float(df["wghtpew"].min(skipna=True)),
                "weight_max": float(df["wghtpew"].max(skipna=True)),
            }
        ]
    )
    summary_path = out_tables / "step1_model_afd_summary.csv"
    summary.to_csv(summary_path, index=False)

    labels = getattr(meta, "column_names_to_labels", {})
    label_rows = []

    for col in df.columns:
        if col in labels:
            label_rows.append({"variable": col, "label": labels[col]})
        elif col in DERIVED_LABELS:
            label_rows.append({"variable": col, "label": DERIVED_LABELS[col]})
        else:
            label_rows.append({"variable": col, "label": ""})

    labels_df = pd.DataFrame(label_rows)
    labels_path = out_tables / "step1_model_afd_labels.csv"
    labels_df.to_csv(labels_path, index=False)

    print(f"[ok] wrote: {out_path}")
    print(f"[ok] wrote: {audit_path}")
    print(f"[ok] wrote: {summary_path}")
    print(f"[ok] wrote: {labels_path}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()