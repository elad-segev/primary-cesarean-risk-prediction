# -*- coding: utf-8 -*-
"""
Data preparation pipeline for the primary-cesarean predictive model.

Responsibilities
----------------
- Apply clinical sanity checks to flag implausible values (no imputation).
- Execute complete-case analysis and report the effective N for the
  Riley / pmsampsize sample-size calculation.
- Split and export cohorts for the primary model and sensitivity analyses.

All functions operate on copies of the input DataFrame and leave the
original object unchanged.
"""

from __future__ import annotations

import os

import pandas as pd


# ---------------------------------------------------------------------------
# Clinical logic
# ---------------------------------------------------------------------------

def apply_clinical_logic(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clinical sanity checks and flag implausible values.

    This function is the designated place for domain-driven data quality rules
    derived from obstetric clinical knowledge.  It does **not** drop or impute
    any rows; it only flags or replaces values that violate hard physiological
    constraints, leaving the decision to exclude to the analyst.

    Example checks to implement here
    ---------------------------------
    - Pre-pregnancy weight must be strictly lower than admission weight
      (weight gain is expected during pregnancy).
    - Gestational age at delivery must fall within a clinically plausible
      range (e.g. 22–44 weeks).
    - Maternal age should be within a plausible reproductive range
      (e.g. 15–55 years).
    - Gravidity must be ≥ 1 for any delivery record.

    Implementation guide
    --------------------
    For each rule, add a boolean mask column (e.g. ``flag_weight_inconsistent``)
    that is True where the constraint is violated.  This preserves the raw
    data and keeps every downstream decision auditable.  Example::

        out = df.copy()
        out["flag_weight_inconsistent"] = (
            out["pre_pregnancy_weight"] >= out["admission_weight"]
        )
        return out

    Parameters
    ----------
    df:
        Raw or minimally-cleaned DataFrame as loaded from the source CSV.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with any flag columns added.  No rows are removed.
    """
    out = df.copy()

    # ── TODO: add clinical sanity-check rules here ──────────────────────────
    # Each rule should add a boolean flag column to `out`.
    # Example (uncomment and adapt):
    #
    # out["flag_weight_inconsistent"] = (
    #     out["pre_pregnancy_weight"] >= out["admission_weight"]
    # )

    return out


# ---------------------------------------------------------------------------
# Complete-case analysis
# ---------------------------------------------------------------------------

def apply_complete_case_analysis(
    df: pd.DataFrame,
    required_cols: list[str],
) -> pd.DataFrame:
    """Drop rows with any missing value in *required_cols* (complete-case only).

    No imputation is performed at any stage.  The function prints the original
    N and the effective N so the analyst can feed these numbers directly into
    the Riley / ``pmsampsize`` sample-size calculation for the target model.

    Parameters
    ----------
    df:
        DataFrame before missingness filtering.
    required_cols:
        Column names that must all be non-missing for a row to be retained.
        Typically the union of the target variable, candidate predictors, and
        any adjustment variables included in the model.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only complete cases on *required_cols*.
        The index is reset.

    Raises
    ------
    ValueError
        If any column in *required_cols* is not present in *df*.
    """
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following required columns are not in the DataFrame: {missing_cols}"
        )

    n_original = len(df)
    complete = df.dropna(subset=required_cols).reset_index(drop=True)
    n_effective = len(complete)
    n_dropped = n_original - n_effective

    print("Complete-case analysis summary")
    print("─" * 40)
    print(f"  Original N          : {n_original:>6,}")
    print(f"  Rows dropped        : {n_dropped:>6,}  ({100 * n_dropped / n_original:.1f}%)")
    print(f"  Effective N         : {n_effective:>6,}")
    print("─" * 40)
    print("  → Use 'Effective N' as the sample size input for pmsampsize (Riley).")

    return complete


# ---------------------------------------------------------------------------
# Cohort splitting and export
# ---------------------------------------------------------------------------

def split_and_save_cohorts(
    df: pd.DataFrame,
    flag_col: str = "was_planned_cs",
    export_dir: str = "data/processed/",
) -> dict[str, pd.DataFrame]:
    """Save the full cohort and a planned-cesarean-excluded subset to CSV.

    Two files are written:

    1. **full_cohort.csv** — all records (N ≈ 3,690), used for descriptive
       statistics and the primary model if the sensitivity analysis is run
       post-hoc.
    2. **spontaneous_labor_cohort.csv** — records where *flag_col* == 0,
       i.e. planned cesareans excluded (N ≈ 3,559).  This is the primary
       analysis cohort per the Chapter 4 methodology.

    Parameters
    ----------
    df:
        DataFrame to split.  Should be the post-cleaning, pre-modelling
        dataset (complete-case filtering applied separately).
    flag_col:
        Binary column where 1 = planned cesarean, 0 = not planned.
        Rows with *flag_col* == 1 are excluded from the spontaneous cohort.
    export_dir:
        Directory path where the CSV files will be written.  Created
        automatically if it does not already exist.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{"full": <full DataFrame>, "spontaneous": <filtered DataFrame>}``
        so the caller can inspect or continue processing in-memory.

    Raises
    ------
    ValueError
        If *flag_col* is not present in *df*.
    """
    if flag_col not in df.columns:
        raise ValueError(
            f"Flag column '{flag_col}' not found in DataFrame.  "
            f"Available columns: {df.columns.tolist()}"
        )

    os.makedirs(export_dir, exist_ok=True)

    # ── 1. Full cohort ───────────────────────────────────────────────────────
    full_path = os.path.join(export_dir, "full_cohort.csv")
    df.to_csv(full_path, index=False)
    print(f"Full cohort saved      : {full_path}  (N={len(df):,})")

    # ── 2. Spontaneous-labour cohort (planned CS excluded) ───────────────────
    spontaneous = df[df[flag_col] == 0].reset_index(drop=True)
    spont_path = os.path.join(export_dir, "spontaneous_labor_cohort.csv")
    spontaneous.to_csv(spont_path, index=False)

    n_excluded = len(df) - len(spontaneous)
    print(f"Spontaneous cohort saved: {spont_path}  (N={len(spontaneous):,})")
    print(f"  → {n_excluded:,} planned cesarean record(s) excluded ('{flag_col}' == 1).")

    return {"full": df, "spontaneous": spontaneous}
