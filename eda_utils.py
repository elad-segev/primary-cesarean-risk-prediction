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
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import config as c


# ---------------------------------------------------------------------------
# Descriptive summary tables
# ---------------------------------------------------------------------------

def describe_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates an extended statistical summary table for numerical columns in a DataFrame.

    The function analyzes numerical features and calculates descriptive statistics
    together with additional distribution and data quality metrics.

    The summary includes:
    - Basic statistics: count, minimum, maximum, mean, median, quartiles, and standard deviation.
    - Value range (max - min).
    - Missing value count and percentage.
    - Column data types.
    - Mode values.
    - Mean, median, and mode similarity check (within 10% proximity).
    - Skewness and kurtosis measurements.
    - Percentage of values within three standard deviations from the mean.

    :param df: Input pandas DataFrame containing numerical columns.
    :type df: pandas.DataFrame
    :return: Extended summary DataFrame describing each numerical column.
    :rtype: pandas.DataFrame
    """

    num_df = df.select_dtypes(include=[np.number])
    summary = round(num_df.describe(), 4).T
    summary["missing_count"] = num_df.isnull().sum()
    summary["missing_pct"] = round((num_df.isnull().sum() / len(df)) * 100, 3)
    summary["range"] = summary["max"] - summary["min"]
    summary["data_type"] = num_df.dtypes

    summary["mode"] = round(num_df.mode().iloc[0], 4)


    # Auxiliary function for checking 10% proximity
    def check_mmm_similarity(row):
        mean_val, median_val, mode_val = row["mean"], row["50%"], row["mode"]
        
        if pd.isna(mean_val) or pd.isna(median_val) or pd.isna(mode_val):
            return False

        def is_close(x, y):
            m = (x + y) / 2
            return x == y if m == 0 else abs(x - y) / abs(m) <= 0.10

        return (is_close(mean_val, median_val) and 
                is_close(mean_val, mode_val) and 
                is_close(median_val, mode_val))

    # Run the test on each row (original column) in the summary table
    summary["MMM_Similar"] = summary.apply(check_mmm_similarity, axis=1)

    # Skewness, Kurtosis 
    summary["skewness"] = round(num_df.skew(), 3)
    summary["kurtosis"] = round(num_df.kurt(), 3)

    # Calculate the percentage of data within 3 standard deviations
    def get_pct_within_3sd(col_name):
        s = num_df[col_name].dropna()
        mean_val = summary.loc[col_name, "mean"]
        std_val = summary.loc[col_name, "std"]
        
        # If there is no standard deviation - return NaN
        if pd.isna(std_val) or std_val == 0:
            return np.nan
            
        lower_bound = mean_val - 3 * std_val
        upper_bound = mean_val + 3 * std_val
        return round(((s >= lower_bound) & (s <= upper_bound)).mean() * 100, 2)

    summary["pct_within_3sd"] = [get_pct_within_3sd(col) for col in summary.index]

    col_order = [
        "count", "min", "max", "range",
        "mean", "50%", "mode","std", 
        "MMM_Similar", "skewness", "kurtosis", "pct_within_3sd",
        "25%", "75%", "missing_count", "missing_pct", "data_type"
    ]
    
    return summary[col_order]


def describe_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a detailed statistical summary for categorical columns in a DataFrame.

    The function iterates through all non-numerical columns and computes
    key categorical statistics for each feature.

    For each column, it calculates:
    - Number of non-missing values.
    - Number of unique values.
    - Most frequent value (mode).
    - Frequency of the most common value.
    - Missing value count and percentage.
    - Original data type.

    The function is robust to empty columns and handles missing values safely.

    :param df: Input pandas DataFrame containing categorical columns.
    :type df: pandas.DataFrame
    :return: Summary DataFrame indexed by variable name.
    :rtype: pandas.DataFrame
    """
    cat_df = df.select_dtypes(exclude=["number"])
    
    summary_list = []
    
    for col in cat_df.columns:
        col_data = cat_df[col].dropna()
        
        if col_data.empty:
            unique_count = 0
            top_val = np.nan
            freq_val = 0

        else:
            unique_count = col_data.nunique()
            top_val = col_data.mode().iloc[0]
            freq_val = col_data.value_counts().iloc[0]

        missing_count = df[col].isnull().sum()
        missing_pct = round((missing_count / len(df)) * 100, 3)

        summary_list.append({
            "Variable": col,
            "count": len(col_data),
            "unique": unique_count,
            "top": top_val,
            "freq": freq_val,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "data_type": df[col].dtype
        })
        
    summary = pd.DataFrame(summary_list).set_index("Variable")
    
    return summary


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------


def visualize_feature_distributions(df, show=False, save_plt=True, output_path:str=c.OUTPUT_DIR):
    """
    Visualizes feature distributions for numerical, categorical, and datetime columns.

    The function automatically detects column types and generates appropriate plots:
    - Histograms for numerical variables.
    - Bar charts for categorical variables.
    - Line plots (monthly aggregated counts) for datetime variables.

    The plots are arranged in a grid layout and can be optionally displayed
    and/or saved to disk.

    :param df: Input pandas DataFrame containing mixed feature types.
    :param show: Whether to display the generated plots.
    :param save_plt: Whether to save the generated figure as an image file.
    :param output_path: Directory where the plot image will be saved.
    :type df: pandas.DataFrame
    :type show: bool
    :type save_plt: bool
    :type output_path: str
    :return: None
    :rtype: None
"""
    summary_num_table = describe_numerical(df)
    summary_cat_table = describe_categorical(df)
    cols_to_plot = df.select_dtypes(exclude=["string"]).columns
    print(cols_to_plot)
    total_cols = len(cols_to_plot)

    num_rows = int(np.ceil(total_cols / 3))
    # Create a grid of graphs
    fig, axes = plt.subplots(num_rows, 3, figsize=(18, num_rows * 6))
    axes = axes.flatten()  # Converting the 2D array to a list
    plt.subplots_adjust(hspace=0.6, wspace=1.0)  # Spacing between graphs

    for idx, column in enumerate(cols_to_plot):
        ax = axes[idx]
        if pd.api.types.is_numeric_dtype(df[column]):
          # If the variable is continuous, print a histogram
            sns.histplot(data=df[column].dropna(), ax=ax, bins=20, color='skyblue')

            ax.set_title(f"{idx+1}) Histogram of {column}")
            ax.set_xlabel(column + f"     miss: {summary_num_table.loc[column, 'missing_pct']}")
            ax.set_ylabel("Frequency")
        
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            monthly_counts = df[column].dropna().dt.to_period('M').value_counts().sort_index()
            
            monthly_counts.plot(kind='line', ax=ax, color='mediumpurple', marker='o', linewidth=2)
            
            ax.set_title(f"{idx+1}) Trend over Time: {column}")
            
            ax.set_xlabel(column + f"     miss: {summary_cat_table.loc[column, 'missing_pct']}%")
            ax.set_ylabel("Count per Month")
        

        else:
            # If the variable is categorical, print a bar graph
            df[column].value_counts().plot(kind='bar', ax=ax, color='orange', edgecolor='black')
            ax.set_title(f"{idx+1})Bar Chart of {column}")
            ax.set_xlabel(str(column)  + f"     miss: {summary_cat_table.loc[column, 'missing_pct']}")
            ax.set_ylabel("Count")

        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
    for blank_idx in range(total_cols, len(axes)):
            fig.delaxes(axes[blank_idx])

    if save_plt:
        plt.savefig(output_path + "/plot_hist_or_bar.png", bbox_inches='tight', dpi=300)

    if show:
        plt.show()


# ---------------------------------------------------------------------------
# Schema Enforcement Layer
# ---------------------------------------------------------------------------


# add docstring --------
def apply_data_schema(df: pd.DataFrame, schema_dict: dict) -> pd.DataFrame:
    """
    Applies a predefined data schema to a pandas DataFrame by casting column types.

    The function iterates over a schema dictionary and converts each column
    to the specified data type. It supports multiple semantic data types,
    including continuous, binary, categorical, datetime, and identifier fields.

    Supported schema types:
    - continuous: Converts to numeric (float/int), invalid parsing becomes NaN.
    - binary: Converts to numeric and then categorical type.
    - nominal: Converts to categorical type.
    - datetime: Converts to pandas datetime with day-first parsing.
    - identifier: Converts to string type.

    Behavior:
    - Columns not present in the DataFrame are skipped.
    - Unknown types generate a warning message.
    - Conversion errors are caught and printed without stopping execution.

    :param df: Input pandas DataFrame to be transformed.
    :param schema_dict: Dictionary mapping column names to type configurations.
    :type df: pandas.DataFrame
    :type schema_dict: dict
    :return: DataFrame with applied type conversions.
    :rtype: pandas.DataFrame
    """
    df_clean = df.copy()
    
    for col, config in schema_dict.items():
        if col not in df_clean.columns:
            continue
            
        col_type = config.get("type")
        
        try:
            if col_type == "continuous":
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
            elif col_type == "binary":
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype('category')
                
            elif col_type == "nominal":
                df_clean[col] = df_clean[col].astype('category')
                
            elif col_type == "datetime":
                df_clean[col] = pd.to_datetime(df_clean[col], dayfirst=True, errors='coerce')
                
            elif col_type == "identifier":
                df_clean[col] = df_clean[col].astype('string')
                
            else:
                print(f"Warning: Unknown type '{col_type}' for column '{col}'.\tSkipping.")
                
        except Exception as e:
            print(f"Error casting column '{col}' to '{col_type}': {e}")
            
    return df_clean

# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

# I STOPED HERE
# add docstring --------
def _spearman_heatmaps(df: pd.DataFrame, vars_: list[str]) -> None:

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


# ---------------------------------------------------------------------------
# Quick missing-value overview
# ---------------------------------------------------------------------------

def plot_simple_missing_heatmap(df: pd.DataFrame) -> None:
    """Render a high-level heatmap of missing values across the full dataset.

    Each cell is coloured by whether the value is missing (bright) or present
    (dark), giving an instant visual impression of missingness patterns without
    any statistical testing.  Use this as a first-pass sanity check before
    calling :func:`analyze_missingness`.

    Parameters
    ----------
    df:
        DataFrame to inspect.  All column types are supported.
    """
    n_missing = df.isnull().sum().sum()
    pct_missing = round(100 * n_missing / df.size, 2)

    plt.figure(figsize=(max(12, len(df.columns) // 2), 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
    plt.title(
        f"Missing-value map  |  {n_missing:,} missing cells ({pct_missing}% of total)",
        fontsize=13,
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Rare-category detection
# ---------------------------------------------------------------------------

def detect_rare_categories(
    df: pd.DataFrame,
    cat_vars: list[str],
    threshold: float = 0.05,
) -> dict[str, list[str]]:
    """Identify category levels whose relative frequency falls below *threshold*.

    Rare categories can inflate model complexity and introduce instability,
    especially in small clinical cohorts.  This function prints a structured
    report and returns the findings for downstream consolidation decisions.

    Frequencies are computed on non-missing values only so that missingness
    does not artificially suppress category counts.

    Parameters
    ----------
    df:
        Source DataFrame.
    cat_vars:
        Names of categorical columns to scan.
    threshold:
        Minimum acceptable relative frequency (default 0.05 = 5 %).
        Categories below this threshold are flagged.

    Returns
    -------
    dict[str, list[str]]
        Mapping of column name → list of rare category labels.
        Columns with no rare categories are omitted from the dict.
    """
    rare: dict[str, list[str]] = {}

    print(f"Rare-category scan  (threshold < {threshold * 100:.1f}%)\n{'─' * 55}")

    for col in cat_vars:
        if col not in df.columns:
            print(f"  [WARN] '{col}' not found in DataFrame — skipped.")
            continue

        non_null = df[col].dropna()
        if non_null.empty:
            continue

        freq = non_null.value_counts(normalize=True)
        rare_levels = freq[freq < threshold]

        if rare_levels.empty:
            continue

        rare[col] = [str(lv) for lv in rare_levels.index]

        print(f"\n  {col}  ({len(rare_levels)} rare / {freq.shape[0]} total categories)")
        for level, rel_freq in rare_levels.items():
            abs_count = non_null.value_counts()[level]
            print(
                f"    • {str(level):<20}  {rel_freq * 100:5.2f}%  "
                f"(n={abs_count})  → consider consolidating"
            )

    if not rare:
        print("  No rare categories detected at the current threshold.")
    else:
        print(f"\n{'─' * 55}")
        print(
            f"  {len(rare)} column(s) contain rare categories.  "
            "Review the report above before modelling."
        )

    return rare


def main():
    import config
    df = pd.read_csv
    plot_distributions()
