# -*- coding: utf-8 -*-
"""
EDA utility functions for clinical predictive modelling.

Design constraints
------------------
- Complete-case analysis only: no imputation of any kind.
- Outlier detection is for inspection only; rows are never dropped here.
- All missing-value percentages are returned as raw floats for downstream
  arithmetic (not formatted strings).
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# ---------------------------------------------------------------------------
# Descriptive summary tables
# ---------------------------------------------------------------------------

def describe_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """Return a descriptive summary for every numeric column in *df*.

    Parameters
    ----------
    df:
        DataFrame whose numeric columns are to be summarised.  Mixed-type
        DataFrames are accepted; non-numeric columns are ignored.

    Returns
    -------
    pd.DataFrame
        One row per numeric variable with columns:
        count, range, min, max, mean, 25%, 50%, 75%, std,
        missing_count, missing_pct (raw float, 0–100), data_type.
    """
    summary = round(df.describe(), 4).T
    summary["missing_count"] = df.isnull().sum()
    summary["missing_pct"] = (df.isnull().sum() / len(df)) * 100
    summary["range"] = summary["max"] - summary["min"]
    summary["data_type"] = df.dtypes

    col_order = [
        "count", "range", "min", "max", "mean",
        "25%", "50%", "75%", "std",
        "missing_count", "missing_pct", "data_type",
    ]
    return summary[col_order]


def describe_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Return a descriptive summary for every non-numeric column in *df*.

    Parameters
    ----------
    df:
        DataFrame whose categorical / object columns are summarised.

    Returns
    -------
    pd.DataFrame
        One row per categorical variable with the standard `describe` columns
        (count, unique, top, freq) plus missing_count, missing_pct (raw float,
        0–100), and data_type.
    """
    cat_df = df.select_dtypes(exclude=["number"])
    summary = cat_df.describe().T
    summary["missing_count"] = df.isnull().sum()
    summary["missing_pct"] = (df.isnull().sum() / len(df)) * 100
    summary["data_type"] = df.dtypes
    return summary


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------

def plot_distributions(
    df: pd.DataFrame,
    target_col: str,
    shapiro: bool = True,
) -> None:
    """Plot histograms or bar charts for every column in *df*, coloured by
    *target_col* class membership.

    Numeric columns receive a KDE histogram with `hue=target_col`.
    Categorical columns receive a count-plot with `hue=target_col`.
    When *shapiro* is True, a Shapiro-Wilk normality test annotation is added
    to each numeric plot (sample capped at 5 000 to keep the test valid).

    Parameters
    ----------
    df:
        DataFrame containing both feature columns and *target_col*.
    target_col:
        Column name used as the hue / grouping variable.
    shapiro:
        Whether to annotate numeric plots with the Shapiro-Wilk p-value.
    """
    plot_cols = [c for c in df.columns if c != target_col]
    n_plots = len(plot_cols)
    n_cols = 3
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 6))
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.7, wspace=0.4)

    for idx, column in enumerate(plot_cols):
        ax = axes[idx]
        miss_pct = round(df[column].isnull().mean() * 100, 3)

        if pd.api.types.is_numeric_dtype(df[column]):
            plot_data = df[[column, target_col]].dropna(subset=[column])

            sns.histplot(
                data=plot_data,
                x=column,
                hue=target_col,
                ax=ax,
                bins=20,
                kde=True,
                fill=True,
                alpha=0.5,
            )

            if shapiro:
                sample = plot_data[column]
                if len(sample) > 5000:
                    sample = sample.sample(n=5000, random_state=42)
                _, p_val = stats.shapiro(sample)

                if p_val < 0.05:
                    ax.annotate(
                        f"Not normal\np={p_val:.1e}",
                        xy=(0.98, 0.96),
                        xycoords="axes fraction",
                        ha="right", va="top",
                        color="red", fontsize=9,
                    )
                else:
                    ax.annotate(
                        f"Normal\np={round(p_val, 4)}",
                        xy=(0.98, 0.96),
                        xycoords="axes fraction",
                        ha="right", va="top",
                        color="green", fontsize=9,
                    )

            ax.set_title(f"{idx + 1}) Histogram of {column}", fontsize=11)
            ax.set_xlabel(f"{column}   [miss: {miss_pct}%]")
            ax.set_ylabel("Frequency")

        else:
            sns.countplot(
                data=df,
                x=column,
                hue=target_col,
                ax=ax,
            )
            ax.set_title(f"{idx + 1}) Bar Chart of {column}", fontsize=11)
            ax.set_xlabel(f"{column}   [miss: {miss_pct}%]")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)

        ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Hide unused axes
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Variable Distributions by Target Class", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def _spearman_heatmaps(df: pd.DataFrame, vars_: list[str]) -> None:
    """Compute and plot Spearman correlation and p-value heatmaps.

    Intended for numeric and ordinal variables.  Handles pairwise complete
    observations (drops rows missing in either variable of each pair).

    Parameters
    ----------
    df:
        Source DataFrame.
    vars_:
        Column names to include in the analysis.
    """
    corr_matrix = pd.DataFrame(index=vars_, columns=vars_, dtype=float)
    pval_matrix = pd.DataFrame(index=vars_, columns=vars_, dtype=float)

    for col1 in vars_:
        for col2 in vars_:
            if col1 == col2:
                corr_matrix.loc[col1, col2] = 1.0
                pval_matrix.loc[col1, col2] = 0.0
            else:
                pair = df[[col1, col2]].dropna()
                corr, p = stats.spearmanr(pair[col1], pair[col2])
                corr_matrix.loc[col1, col2] = corr
                pval_matrix.loc[col1, col2] = p

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    sns.heatmap(
        corr_matrix.astype(float),
        mask=mask,
        annot=True, fmt=".2f",
        cmap="PiYG", cbar=True, square=True,
        vmin=-1, vmax=1,
        annot_kws={"size": 9},
        ax=axes[0],
    )
    axes[0].set_title("Spearman Correlation (lower triangle)", fontsize=14)
    axes[0].tick_params(axis="x", rotation=45)

    sns.heatmap(
        (pval_matrix.astype(float) * 100),
        mask=mask,
        annot=True, fmt=".2f",
        cmap="PiYG", cbar=True, square=True,
        vmin=0, vmax=5,
        annot_kws={"size": 9},
        ax=axes[1],
    )
    axes[1].set_title("Spearman p-value × 100 (lower triangle)", fontsize=14)
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def _cramers_v(contingency_table: pd.DataFrame) -> float:
    """Compute Cramér's V for a contingency table."""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.values.sum()
    min_dim = min(contingency_table.shape) - 1
    if min_dim == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def _chi2_heatmap(df: pd.DataFrame, vars_: list[str]) -> None:
    """Compute Chi-Square p-values and Cramér's V for categorical pairs and
    display a p-value heatmap.

    Parameters
    ----------
    df:
        Source DataFrame.
    vars_:
        Categorical column names to include.
    """
    pval_matrix = pd.DataFrame(index=vars_, columns=vars_, dtype=float)

    for var1 in vars_:
        for var2 in vars_:
            if var1 == var2:
                pval_matrix.loc[var1, var2] = np.nan
            else:
                pair = df[[var1, var2]].dropna()
                ct = pd.crosstab(pair[var1], pair[var2])
                _, p, _, _ = stats.chi2_contingency(ct)
                pval_matrix.loc[var1, var2] = p

    mask = np.triu(np.ones_like(pval_matrix, dtype=bool))

    plt.figure(figsize=(max(10, len(vars_)), max(8, len(vars_) - 2)))
    sns.heatmap(
        pval_matrix.astype(float) * 100,
        mask=mask,
        annot=True, fmt=".2f",
        cmap="PiYG", cbar=True, square=True,
        vmin=0, vmax=5,
        annot_kws={"size": 9},
    )
    plt.title("Chi-Square p-value × 100 — Categorical Variables (lower triangle)", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.show()

    # Print pairs above the significance threshold ordered by Cramér's V
    significant: list[tuple[str, str, float, float]] = []
    for i, var1 in enumerate(vars_):
        for j, var2 in enumerate(vars_):
            if i >= j:
                continue
            pair = df[[var1, var2]].dropna()
            ct = pd.crosstab(pair[var1], pair[var2])
            chi2, p, _, _ = stats.chi2_contingency(ct)
            if p <= 0.05:
                cv = _cramers_v(ct)
                significant.append((var1, var2, p, cv))

    significant.sort(key=lambda x: x[3], reverse=True)
    print(f"\nSignificant pairs (p ≤ 0.05), sorted by Cramér's V [{len(significant)}]:")
    for i, (v1, v2, p, cv) in enumerate(significant, start=1):
        print(f"  {i:>3}) {v1} × {v2}  |  Cramér's V = {cv:.3f}  |  p = {p:.3e}")


def plot_correlations(
    df: pd.DataFrame,
    num_vars: list[str],
    cat_vars: list[str],
) -> None:
    """Generate correlation heatmaps for numeric/ordinal and categorical
    variable sets.

    Numeric/ordinal variables are assessed with Spearman rank correlation
    (two heatmaps: coefficients and p-values).
    Categorical variables are assessed with Chi-Square / Cramér's V
    (one p-value heatmap plus a printed ranking of significant pairs).

    Parameters
    ----------
    df:
        Full DataFrame (or the subset you want to analyse).
    num_vars:
        Names of numeric and ordinal columns.
    cat_vars:
        Names of categorical columns.
    """
    if num_vars:
        print("── Spearman correlation (numeric / ordinal) ──")
        _spearman_heatmaps(df, num_vars)

    if cat_vars:
        print("── Chi-Square / Cramér's V (categorical) ──")
        _chi2_heatmap(df, cat_vars)


# ---------------------------------------------------------------------------
# Missingness analysis  (MCAR vs MAR)
# ---------------------------------------------------------------------------

def analyze_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Formally test whether missing data in each variable are MCAR or MAR.

    For each variable with at least one missing value the function tests its
    missing-value indicator against every other column:

    - **Numeric other column** → Kolmogorov-Smirnov test comparing the full
      distribution to the subset observed when the focal column is present.
    - **Categorical other column** → Chi-Square test between the binary
      missing indicator and the categorical column.

    A p-value ≤ 0.05 is treated as evidence that the other column is
    *associated* with the missingness of the focal variable (MAR signal).

    No imputation is performed; the function is diagnostic only.

    Parameters
    ----------
    df:
        DataFrame to inspect.  All column types are supported.

    Returns
    -------
    pd.DataFrame
        Boolean matrix (rows = variables with missing data, columns = all
        other variables).  True means the association test was significant
        (p ≤ 0.05).  Two extra columns are appended:

        - ``n_associated``: count of columns significantly associated with
          the missingness of the focal variable.
        - ``is_MCAR``: True when no association is detected (n_associated == 0).
    """
    missing_cols = [c for c in df.columns if df[c].isnull().any()]

    if not missing_cols:
        print("No missing values detected — nothing to analyse.")
        return pd.DataFrame()

    p_values: dict[str, dict[str, float]] = {}

    for focal in missing_cols:
        p_values[focal] = {}
        for other in df.columns:
            if other == focal:
                continue

            if pd.api.types.is_numeric_dtype(df[other]):
                full_series = df[other].dropna()
                observed_series = df.loc[df[focal].notnull(), other].dropna()
                if len(observed_series) < 2 or len(full_series) < 2:
                    p_values[focal][other] = np.nan
                    continue
                _, p = stats.ks_2samp(full_series, observed_series)
            else:
                missing_indicator = df[focal].isnull().astype(int)
                ct = pd.crosstab(missing_indicator, df[other].dropna())
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    p_values[focal][other] = np.nan
                    continue
                _, p, _, _ = stats.chi2_contingency(ct)

            p_values[focal][other] = p

    pval_df = pd.DataFrame(p_values).T          # rows = focal vars
    sig_df = pval_df <= 0.05                     # True where association is significant

    # ── Heatmap of significance flags ──────────────────────────────────────
    plt.figure(figsize=(max(12, pval_df.shape[1] // 2), max(6, len(missing_cols))))
    sns.heatmap(
        sig_df.astype(float),
        annot=True, fmt=".0f",
        cmap="PiYG", cbar=True,
        annot_kws={"size": 8},
        vmin=0, vmax=1,
    )
    plt.title(
        "Missingness association heatmap\n"
        "1 = significant (p ≤ 0.05)  |  0 = not significant\n"
        "(rows: variables with missing data; columns: potential predictors of missingness)",
        fontsize=12,
    )
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.show()

    # ── Summary columns ────────────────────────────────────────────────────
    sig_df["n_associated"] = sig_df.sum(axis=1)
    sig_df["is_MCAR"] = sig_df["n_associated"] == 0

    print("\nMissingness mechanism summary:")
    print(sig_df[["n_associated", "is_MCAR"]].to_string())

    return sig_df
