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
from typing import List, Tuple, Optional

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


def visualize_feature_distributions(df: pd.DataFrame, show:bool=False, save_plt:bool=True, output_path:str=c.OUTPUT_DIR):
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
        plt.savefig(output_path + "/visualize_feature_distributions.png", bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)



def visualize_outliers_and_proportions(df: pd.DataFrame, show: bool = False, save_plt: bool = True, output_path:str=c.OUTPUT_DIR) -> Tuple[pd.DataFrame, dict, list]:
    """
    Visualizes outliers and category proportions for variables in a DataFrame.

    The function creates a combined exploratory visualization:
    - Pie charts for categorical variables showing category proportions.
    - Box plots for numerical variables highlighting potential outliers.

    Outlier detection:
    - Numerical columns:
    Uses the Interquartile Range (IQR) method:
    values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are marked as outliers.
    - Categorical columns:
    Categories representing less than 5% of valid observations are considered
    rare categories.

    The function also tracks:
    - A boolean matrix indicating which rows contain outliers/rare categories.
    - Number of detected numerical outliers per variable.
    - List of rare categorical values.

    The generated figure can be displayed and/or saved.

    :param df: Input pandas DataFrame.
    :param show: Whether to display the generated plots.
    :param save_plt: Whether to save the generated figure.
    :param output_path: Directory path for saving the plot image.
    :type df: pandas.DataFrame
    :type show: bool
    :type save_plt: bool
    :type output_path: str | Path

    :return:
        Tuple containing:
        - Boolean DataFrame marking outlier/rare values.
        - Dictionary with numerical outlier counts by column.
        - List describing rare categories.
    :rtype: Tuple[pandas.DataFrame, dict, list]
    """
    summary_num_table = describe_numerical(df)
    summary_cat_table = describe_categorical(df)

    # Exclude identifiers and dates
    cols_to_plot = df.select_dtypes(exclude=["string", "datetime", "datetime64"]).columns
    total_cols = len(cols_to_plot)
    
    num_rows = int(np.ceil(total_cols / 3))
    fig, axes = plt.subplots(num_rows, 3, figsize=(18, num_rows * 6))
    axes = axes.flatten()
        
    plt.subplots_adjust(hspace=0.6, wspace=1.0)

    # Initialize tracking variables
    outliers_matrix = pd.DataFrame(False, index=df.index, columns=df.columns)
    rare_categories_cat_list = []
    count_of_outliers = {}

    for plot_index, col in enumerate(cols_to_plot):
        ax = axes[plot_index]
        
        # Categorical Logic (Pie Chart)
        if isinstance(df[col].dtype, pd.CategoricalDtype) or df[col].dtype == 'object':
            value_counts = df[col].value_counts()
            
            # Draw Pie Chart
            if not value_counts.empty:
                value_counts.plot.pie(autopct='%1.1f%%', startangle=90, ax=ax)
            
            # Add Legend
            labels = [f"{label} -> {count}" for label, count in value_counts.items()]
            ax.legend(labels, title="Categories Frequency", loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
            
            # Identify Rare Categories (< 5%)
            valid_count = len(df[col].dropna())
            threshold = 0.05 * valid_count
            rare_categories = value_counts[value_counts < threshold].index
            
            if len(rare_categories) > 0:
                rare_percentages = {cat: (value_counts[cat] / valid_count) * 100 for cat in rare_categories}
                rare_cat_str = f"{col}: " + ", ".join(f"{cat}: {pct:.3f}%" for cat, pct in rare_percentages.items())
                rare_categories_cat_list.append(rare_cat_str)
            
            # Update Boolean Matrix
            outliers_matrix[col] = df[col].isin(rare_categories)
            
            ax.set_title(f"{plot_index + 1}) Pie Chart for {col}")
            
            # Display Missing Values
            na_percent = summary_cat_table.loc[col, "missing_pct"] if col in summary_cat_table.index else "N/A"
            ax.text(0.05, 0.92, f"NA: {na_percent}%", transform=ax.transAxes, fontsize=10, color='black')
            ax.set_ylabel('')

        # Numerical Logic (Box Plot)
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Retrieve Q1 and Q3 from the pre-calculated summary table
            q1 = summary_num_table.loc[col, "25%"]
            q3 = summary_num_table.loc[col, "75%"]
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Identify Outliers
            is_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers_matrix[col] = is_outlier
            
            count_outliers = is_outlier.sum()
            count_of_outliers[col] = count_outliers
            
            # Draw Box Plot
            sns.boxplot(x=df[col].dropna(), ax=ax, color='lightgreen')
            
            ax.set_title(f"{plot_index + 1}) Box Plot with Outliers for {col}")
            
            # Display Missing Values and Outlier Stats
            na_percent = summary_num_table.loc[col, "missing_pct"] if col in summary_num_table.index else "N/A"
            
            ax.text(0.05, 0.92, f"NA: {na_percent}%", transform=ax.transAxes, fontsize=10, color='black')
            
            outlier_pct = round(100 * count_outliers / len(df), 3)
            ax.text(0.05, 0.02, f"Outliers: {count_outliers} ({outlier_pct}%)", transform=ax.transAxes, fontsize=10, color='black')

    # Remove empty subplots
    for blank_idx in range(total_cols, len(axes)):
        fig.delaxes(axes[blank_idx])

    if save_plt:
        if output_path:
            plt.savefig(output_path + "/visualize_outliers_and_proportions.png", bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return outliers_matrix, count_of_outliers, rare_categories_cat_list


# ---------------------------------------------------------------------------
# Schema Enforcement Layer
# ---------------------------------------------------------------------------


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
# Data division by distribution type
# ---------------------------------------------------------------------------


def split_variables_by_type(df: pd.DataFrame, normal_vars: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Splits DataFrame variables into groups based on data type and normality assumption.

    The function categorizes columns into:
    - Categorical variables (category dtype)
    - Continuous variables assumed to be normally distributed
    - Continuous variables not assumed to be normally distributed
    - Datetime variables

    The classification is based on both pandas dtype inference and a provided list
    of normally distributed variables.

    :param df: Input pandas DataFrame.
    :param normal_vars: List of column names assumed to follow a normal distribution.
    :type df: pandas.DataFrame
    :type normal_vars: List[str]
    :return: Tuple containing:
            (categorical variables,
            non-normal continuous variables,
            normal continuous variables,
            datetime variables)
    :rtype: Tuple[List[str], List[str], List[str], List[str]]
    """
    cat_vars = df.select_dtypes(include=['category']).columns.tolist()
    all_cont_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    non_normal_cont_vars = [col for col in all_cont_vars if col not in normal_vars]
    normal_cont_vars = [col for col in all_cont_vars if col in normal_vars]
    datetime_vars = df.select_dtypes(include=['datetime']).columns.tolist()
    
    return cat_vars, non_normal_cont_vars, normal_cont_vars, datetime_vars


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------


def pearson_matrices(df: pd.DataFrame, normal_vars: list[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates Pearson correlation and p-value matrices for normally distributed
    continuous variables.

    The function computes pairwise Pearson correlation coefficients between all
    variables listed in normal_vars. It also calculates the statistical significance
    (p-value) for each correlation.

    The implementation calculates only the upper triangular part of the matrices
    (excluding the diagonal) and mirrors the results to the lower triangular part,
    taking advantage of correlation matrix symmetry.

    For each pair of variables:
    - Self-correlation values are set to 1 with p-value 0.
    - Missing values are removed using pairwise deletion.
    - Pairs with fewer than two valid observations return NaN values.

    :param df: Input pandas DataFrame containing the numerical variables.
    :param normal_vars: List of variables assumed to follow a normal distribution.
    :type df: pandas.DataFrame
    :type normal_vars: list[str]

    :return:
        Tuple containing:
        - Pearson p-value matrix.
        - Pearson correlation coefficient matrix.
    :rtype: Tuple[pandas.DataFrame, pandas.DataFrame]
    """
    corr_matrix = pd.DataFrame(index=normal_vars, columns=normal_vars, dtype=float)
    pval_matrix = pd.DataFrame(index=normal_vars, columns=normal_vars, dtype=float)

    for i, col1 in enumerate(normal_vars):
        for j, col2 in enumerate(normal_vars):
            
            if col1 == col2:
                corr_matrix.loc[col1, col2] = 1.0  
                pval_matrix.loc[col1, col2] = 0.0  
                
            elif i < j:
                pair = df[[col1, col2]].dropna()
    
                if len(pair) < 2:
                    corr_matrix.loc[col1, col2] = np.nan
                    pval_matrix.loc[col1, col2] = np.nan
                    corr_matrix.loc[col2, col1] = np.nan
                    pval_matrix.loc[col2, col1] = np.nan
                else:
                    corr, p = stats.pearsonr(pair[col1], pair[col2])
                
                    corr_matrix.loc[col1, col2] = corr
                    pval_matrix.loc[col1, col2] = p
                    corr_matrix.loc[col2, col1] = corr
                    pval_matrix.loc[col2, col1] = p

    return pval_matrix, corr_matrix


def spearman_matrices(df: pd.DataFrame, normal_vars: List[str], non_normal_vars: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates Spearman correlation and p-value matrices for variables that are
    not assumed to follow a normal distribution.

    The function combines normal and non-normal variable lists and computes
    pairwise Spearman rank correlations where applicable. Spearman correlation is
    used for non-normal variables because it does not require the assumption of
    normally distributed data.

    Behavior:
    - Non-normal variables are compared using Spearman correlation.
    - Correlations between two normal variables are skipped because they should
    be handled using Pearson correlation instead.
    - Self-correlation for non-normal variables is set to 1 with p-value 0.
    - Missing values are removed using pairwise deletion.
    - Results are mirrored across the matrix diagonal due to correlation symmetry.
    - Pairs with insufficient observations return NaN values.

    :param df: Input pandas DataFrame containing the variables.
    :param normal_vars: List of variables assumed to follow a normal distribution.
    :param non_normal_vars: List of variables that do not follow a normal distribution.
    :type df: pandas.DataFrame
    :type normal_vars: List[str]
    :type non_normal_vars: List[str]

    :return:
        Tuple containing:
        - Spearman p-value matrix.
        - Spearman correlation coefficient matrix.
    :rtype: Tuple[pandas.DataFrame, pandas.DataFrame]
    """
    all_vars = normal_vars + non_normal_vars
    corr_matrix = pd.DataFrame(index=all_vars, columns=all_vars, dtype=float)
    pval_matrix = pd.DataFrame(index=all_vars, columns=all_vars, dtype=float)

    for i, col1 in enumerate(all_vars):
        for j, col2 in enumerate(all_vars):
            
            
            if col1 == col2:

                if col1 in non_normal_vars:
                    corr_matrix.loc[col1, col2] = 1.0  
                    pval_matrix.loc[col1, col2] = 0.0  
                else:
                    # normal varb need to proccess person test
                    corr_matrix.loc[col1, col2] = np.nan
                    pval_matrix.loc[col1, col2] = np.nan
                    
            elif i < j:
                if (col1 in normal_vars) and (col2 in normal_vars):
                    corr_matrix.loc[col1, col2] = np.nan
                    pval_matrix.loc[col1, col2] = np.nan
                    corr_matrix.loc[col2, col1] = np.nan
                    pval_matrix.loc[col2, col1] = np.nan
                    continue
                    
                pair = df[[col1, col2]].dropna()
    
                if len(pair) < 2:
                    corr_matrix.loc[col1, col2] = np.nan
                    pval_matrix.loc[col1, col2] = np.nan
                    corr_matrix.loc[col2, col1] = np.nan
                    pval_matrix.loc[col2, col1] = np.nan
                else:
                    corr, p = stats.spearmanr(pair[col1], pair[col2])
                    
                    corr_matrix.loc[col1, col2] = corr
                    pval_matrix.loc[col1, col2] = p
                    corr_matrix.loc[col2, col1] = corr
                    pval_matrix.loc[col2, col1] = p

    return pval_matrix, corr_matrix


def mann_whitney_matrices(df: pd.DataFrame, non_normal_vars: List[str], binary_vars: List[str], rank_biserial_correlation:bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates Mann-Whitney U test matrices for comparing non-normal continuous
    variables against binary categorical variables.

    The function evaluates whether the distribution of each non-normal numerical
    variable differs significantly between the two groups defined by each binary
    variable.

    For each continuous-binary variable pair:
    - Missing values are removed using pairwise deletion.
    - The binary variable must contain exactly two unique groups.
    - Mann-Whitney U test is performed using a two-sided alternative hypothesis.
    - The result can be represented either as:
        - Rank-biserial correlation (effect size, range: -1 to 1)
        - Raw Mann-Whitney U statistic

    The p-value matrix contains statistical significance values, while the
    correlation matrix contains effect sizes or U statistics.

    :param df: Input pandas DataFrame containing the variables.
    :param non_normal_vars: List of numerical variables that do not follow a
                            normal distribution.
    :param binary_vars: List of binary categorical variables.
    :param rank_biserial_correlation: If True, converts U statistic to
                                    rank-biserial correlation.
    :type df: pandas.DataFrame
    :type non_normal_vars: List[str]
    :type binary_vars: List[str]
    :type rank_biserial_correlation: bool

    :return:
        Tuple containing:
        - Mann-Whitney p-value matrix.
        - Effect size / U statistic matrix.
    :rtype: Tuple[pandas.DataFrame, pandas.DataFrame]
    """
    corr_matrix = pd.DataFrame(index=non_normal_vars, columns=binary_vars, dtype=float)
    pval_matrix = pd.DataFrame(index=non_normal_vars, columns=binary_vars, dtype=float)

    for cont_var in non_normal_vars:
        for bin_var in binary_vars:
            
            pair = df[[cont_var, bin_var]].dropna()
            unique_groups = pair[bin_var].unique()
            
            if len(unique_groups) != 2:
                corr_matrix.loc[cont_var, bin_var] = np.nan
                pval_matrix.loc[cont_var, bin_var] = np.nan
                continue
                
            group_a = pair[pair[bin_var] == unique_groups[0]][cont_var]
            group_b = pair[pair[bin_var] == unique_groups[1]][cont_var]
            
            n1 = len(group_a)
            n2 = len(group_b)
            
            if n1 < 1 or n2 < 1:
                corr_matrix.loc[cont_var, bin_var] = np.nan
                pval_matrix.loc[cont_var, bin_var] = np.nan
                continue
                
            stat_u, p = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
            
            # Convert from U statistic to Rank-Biserial correlation (ranging from -1 to 1)

            if rank_biserial_correlation:
                r_rb = 1 - (2 * stat_u) / (n1 * n2)
                corr_matrix.loc[cont_var, bin_var] = r_rb
            
            else:
                corr_matrix.loc[cont_var, bin_var] = stat_u
            
            pval_matrix.loc[cont_var, bin_var] = p

    return pval_matrix, corr_matrix


def ttest_matrices(df: pd.DataFrame, normal_vars: List[str], binary_vars: List[str], point_biserial_correlation: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates independent samples t-test matrices between continuous normal variables
    and binary categorical variables.

    The function evaluates whether the means of each normally distributed continuous
    variable differ significantly between two groups defined by each binary variable.

    For each continuous-binary variable pair:
    - Missing values are removed using pairwise deletion.
    - The binary variable must contain exactly two unique groups.
    - Independent two-sample t-test is performed (equal variance assumed).
    - Requires at least two observations per group.

    The results include:
    - p-value matrix (statistical significance).
    - Effect size matrix, which can be either:
        - Point-biserial correlation (recommended, range -1 to 1), or
        - Raw t-statistic.

    The point-biserial correlation is computed using:
    r = t / sqrt(t² + df), where df = n1 + n2 - 2.

    :param df: Input pandas DataFrame containing the variables.
    :param normal_vars: List of continuous variables assumed to follow a normal distribution.
    :param binary_vars: List of binary categorical variables.
    :param point_biserial_correlation: If True, converts t-statistic to point-biserial correlation.
    :type df: pandas.DataFrame
    :type normal_vars: List[str]
    :type binary_vars: List[str]
    :type point_biserial_correlation: bool

    :return:
        Tuple containing:
        - t-test p-value matrix.
        - Effect size / t-statistic matrix.
    :rtype: Tuple[pandas.DataFrame, pandas.DataFrame]
    """
    corr_matrix = pd.DataFrame(index=normal_vars, columns=binary_vars, dtype=float)
    pval_matrix = pd.DataFrame(index=normal_vars, columns=binary_vars, dtype=float)

    for cont_var in normal_vars:
        for bin_var in binary_vars:
            
            pair = df[[cont_var, bin_var]].dropna()
            unique_groups = pair[bin_var].unique()
            
            if len(unique_groups) != 2:
                corr_matrix.loc[cont_var, bin_var] = np.nan
                pval_matrix.loc[cont_var, bin_var] = np.nan
                continue
                
            group_a = pair[pair[bin_var] == unique_groups[0]][cont_var]
            group_b = pair[pair[bin_var] == unique_groups[1]][cont_var]
            
            n1 = len(group_a)
            n2 = len(group_b)
            
            # T-test requires at least 2 observations per group to calculate variance
            if n1 < 2 or n2 < 2:
                corr_matrix.loc[cont_var, bin_var] = np.nan
                pval_matrix.loc[cont_var, bin_var] = np.nan
                continue
                
            # Perform standard independent 2-sample T-test
            stat_t, p = stats.ttest_ind(group_a, group_b, equal_var=True)
            
            # Convert T-statistic to Point-Biserial correlation (ranging from -1 to 1)
            # Formula: r = t / sqrt(t^2 + df) where df = n1 + n2 - 2
            if point_biserial_correlation:
                df_degrees = n1 + n2 - 2
                r_pb = stat_t / np.sqrt((stat_t ** 2) + df_degrees)
                corr_matrix.loc[cont_var, bin_var] = r_pb
            else:
                corr_matrix.loc[cont_var, bin_var] = stat_t
            
            pval_matrix.loc[cont_var, bin_var] = p

    return pval_matrix, corr_matrix


def _cramers_v(contingency_table: pd.DataFrame) -> float:
    """
    Helper function to compute Cramér's V for a contingency table.
    """
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.values.sum()
    min_dim = min(contingency_table.shape) - 1
    
    if min_dim == 0 or n == 0:
        return 0.0
        
    return float(np.sqrt(chi2 / (n * min_dim)))
    

def chi2_matrices(df: pd.DataFrame, cat_vars: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates Chi-square test matrices for categorical variables, including
    Cramér's V effect size and p-values.

    The function evaluates associations between all pairs of categorical variables
    using the Chi-square test of independence.

    For each pair of categorical variables:
    - Missing values are removed using pairwise deletion.
    - A contingency table is constructed using pd.crosstab.
    - The Chi-square test is applied to compute statistical significance.
    - Cramér's V is computed as a normalized measure of association strength.

    Behavior:
    - Diagonal values (self-comparisons) are set to NaN.
    - Symmetry is enforced: results are mirrored across the matrix diagonal.
    - Pairs with insufficient category variation (less than 2x2 table) are skipped.

    Outputs:
    - p-value matrix indicating statistical significance.
    - Cramér's V matrix indicating effect size (0 to 1).

    :param df: Input pandas DataFrame containing categorical variables.
    :param cat_vars: List of categorical variable names.
    :type df: pandas.DataFrame
    :type cat_vars: List[str]

    :return:
        Tuple containing:
        - Chi-square p-value matrix.
        - Cramér's V association matrix.
    :rtype: Tuple[pandas.DataFrame, pandas.DataFrame]
    """
    cramers_v = pd.DataFrame(index=cat_vars, columns=cat_vars, dtype=float)
    pval_matrix = pd.DataFrame(index=cat_vars, columns=cat_vars, dtype=float)

    for i, col1 in enumerate(cat_vars):
        for j, col2 in enumerate(cat_vars):
            
            if i == j:
                cramers_v.loc[col1, col2] = np.nan
                pval_matrix.loc[col1, col2] = np.nan
                
            elif i < j:
                pair = df[[col1, col2]].dropna()
                
                ct = pd.crosstab(pair[col1], pair[col2])
                
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    cramers_v.loc[col1, col2] = np.nan
                    pval_matrix.loc[col1, col2] = np.nan
                    cramers_v.loc[col2, col1] = np.nan
                    pval_matrix.loc[col2, col1] = np.nan
                    continue
                
                _, p, _, _ = stats.chi2_contingency(ct)
                
                cv = _cramers_v(ct)
                
                # הזנת הנתונים למשולש העליון
                cramers_v.loc[col1, col2] = cv
                pval_matrix.loc[col1, col2] = p
                
                # שכפול למשולש התחתון
                cramers_v.loc[col2, col1] = cv
                pval_matrix.loc[col2, col1] = p

    return pval_matrix, cramers_v


def anova_matrices(df: pd.DataFrame, normal_vars: List[str], cat_vars_gt2: List[str], eta_squared: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates one-way ANOVA matrices for normally distributed continuous variables
    across categorical variables with more than two groups.

    The function evaluates whether the means of each normally distributed continuous
    variable differ significantly across multiple independent categorical groups.

    For each continuous-categorical pair:
    - Missing values are removed using pairwise deletion.
    - Groups are formed based on categorical levels.
    - Only variables with at least two valid groups are analyzed.
    - One-way ANOVA (F-test) is applied.

    Effect size:
    - Optionally computes eta-squared (η²), a measure of explained variance:
    η² = (F * df_between) / (F * df_between + df_within)

    Outputs:
    - p-value matrix (statistical significance).
    - effect size matrix (η² or raw F statistic).

    :param df: Input pandas DataFrame containing variables.
    :param normal_vars: List of continuous variables assumed to follow a normal distribution.
    :param cat_vars_gt2: List of categorical variables with more than two groups.
    :param eta_squared: If True, computes eta-squared effect size instead of raw F statistic.
    :type df: pandas.DataFrame
    :type normal_vars: List[str]
    :type cat_vars_gt2: List[str]
    :type eta_squared: bool

    :return:
        Tuple containing:
        - ANOVA p-value matrix.
        - Effect size matrix (eta-squared or F statistic).
    :rtype: Tuple[pandas.DataFrame, pandas.DataFrame]
    """
    effect_matrix = pd.DataFrame(index=normal_vars, columns=cat_vars_gt2, dtype=float)
    pval_matrix = pd.DataFrame(index=normal_vars, columns=cat_vars_gt2, dtype=float)

    for cont_var in normal_vars:
        for cat_var in cat_vars_gt2:
            
            pair = df[[cont_var, cat_var]].dropna()
            
            groups = [group.values for name, group in pair.groupby(cat_var, observed=True)[cont_var] if len(group) > 0]
            
            if len(groups) < 2:
                effect_matrix.loc[cont_var, cat_var] = np.nan
                pval_matrix.loc[cont_var, cat_var] = np.nan
                continue
                
            stat_f, p = stats.f_oneway(*groups)
            
            if eta_squared:
                k = len(groups)
                n_total = sum(len(g) for g in groups)
                
                df_between = k - 1
                df_within = n_total - k
                
                if df_within > 0:
                    eta_sq = (stat_f * df_between) / ((stat_f * df_between) + df_within)
                    effect_matrix.loc[cont_var, cat_var] = eta_sq
                else:
                    effect_matrix.loc[cont_var, cat_var] = np.nan
            else:
                effect_matrix.loc[cont_var, cat_var] = stat_f
                
            pval_matrix.loc[cont_var, cat_var] = p

    return pval_matrix, effect_matrix


def htest_matrices(df: pd.DataFrame, non_normal_vars: List[str], cat_vars_gt2: List[str], epsilon_squared: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates Kruskal-Wallis test (H-Test) matrices for non-normal continuous variables
    across categorical variables with more than two groups.

    The function evaluates whether there are statistically significant differences
    in the distributions of each non-normal continuous variable across multiple
    independent categorical groups.

    For each continuous-categorical pair:
    - Missing values are removed using pairwise deletion.
    - Groups are formed based on categorical levels.
    - Only variables with at least two valid groups are analyzed.
    - The Kruskal-Wallis H-test is applied (non-parametric alternative to ANOVA).

    Effect size:
    - Optionally computes epsilon-squared (ε²) as a measure of effect size:
    ε² = H / (n - 1), where H is the Kruskal-Wallis statistic and n is total sample size.

    Outputs:
    - p-value matrix (statistical significance).
    - effect size matrix (ε² or raw H statistic).

    :param df: Input pandas DataFrame containing variables.
    :param non_normal_vars: List of continuous variables not assumed to be normally distributed.
    :param cat_vars_gt2: List of categorical variables with more than two groups.
    :param epsilon_squared: If True, computes epsilon-squared effect size instead of raw H statistic.
    :type df: pandas.DataFrame
    :type non_normal_vars: List[str]
    :type cat_vars_gt2: List[str]
    :type epsilon_squared: bool

    :return:
        Tuple containing:
        - Kruskal-Wallis p-value matrix.
        - Effect size matrix (epsilon-squared or H statistic).
    :rtype: Tuple[pandas.DataFrame, pandas.DataFrame]
    """
    effect_matrix = pd.DataFrame(index=non_normal_vars, columns=cat_vars_gt2, dtype=float)
    pval_matrix = pd.DataFrame(index=non_normal_vars, columns=cat_vars_gt2, dtype=float)
    
    for cont_var in non_normal_vars:
        for cat_var in cat_vars_gt2:
            
            pair = df[[cont_var, cat_var]].dropna()
            
            groups = [group.values for name, group in pair.groupby(cat_var, observed=True)[cont_var] if len(group) > 0]
            
            if len(groups) < 2:
                effect_matrix.loc[cont_var, cat_var] = np.nan
                pval_matrix.loc[cont_var, cat_var] = np.nan
                continue
                
            stat_h, p = stats.kruskal(*groups)
            
            if epsilon_squared:
                n = sum(len(g) for g in groups)
                if n > 1:
                    eps_sq = stat_h / (n - 1)
                    effect_matrix.loc[cont_var, cat_var] = eps_sq
                else:
                    effect_matrix.loc[cont_var, cat_var] = np.nan
            else:
                effect_matrix.loc[cont_var, cat_var] = stat_h
                
            pval_matrix.loc[cont_var, cat_var] = p

    return pval_matrix, effect_matrix


def plot_statistical_heatmaps(pval_matrix: pd.DataFrame, effect_matrix: Optional[pd.DataFrame] = None, plot_title: str = "Statistical Analysis",
    test_name: str = "Test", file_name: str = "statistical_heatmap", show: bool = False,save_plt: bool = True, output_path: str = c.OUTPUT_DIR) -> None:
    """
    Plots statistical heatmaps for p-value matrices and optional effect size matrices.

    The function visualizes results from statistical tests (e.g., correlation,
    t-test, ANOVA, chi-square) using seaborn heatmaps.

    It supports:
    - P-value heatmap (always plotted).
    - Optional effect size heatmap (e.g., Cramér's V, eta², correlation, etc.).

    Key features:
    - Automatically detects whether the matrix is symmetric.
    - Applies upper-triangle masking for symmetric matrices.
    - Dynamically adjusts figure size based on matrix dimensions.
    - Uses different color maps depending on effect size range:
    * Diverging colormap (coolwarm) for signed effects (-1 to 1).
    * Sequential colormap (Blues) for non-negative effects (0 to 1).

    Visualization settings:
    - Annotated heatmaps with numeric values.
    - Rotated axis labels for readability.
    - Tight layout for clean spacing.
    - Optional saving to file and display control.

    :param pval_matrix: Matrix of p-values.
    :param effect_matrix: Optional matrix of effect sizes.
    :param plot_title: Title of the plot.
    :param test_name: Name of the statistical test displayed in title.
    :param file_name: Output filename (without extension).
    :param show: Whether to display the plot.
    :param save_plt: Whether to save the plot to disk.
    :param output_path: Directory where plot is saved.
    :type pval_matrix: pandas.DataFrame
    :type effect_matrix: Optional[pandas.DataFrame]
    :type plot_title: str
    :type test_name: str
    :type file_name: str
    :type show: bool
    :type save_plt: bool
    :type output_path: str

    :return: None
    """
    is_sym = (pval_matrix.shape[0] == pval_matrix.shape[1]) and \
             (list(pval_matrix.index) == list(pval_matrix.columns))
    
    
    mask = np.triu(np.ones_like(pval_matrix, dtype=bool)) if is_sym else None

    
    cols = pval_matrix.shape[1]
    rows = pval_matrix.shape[0]
    
    fig_width = max(10, cols * 1.2)
    base_height = max(6, rows * 0.8)
    
    num_plots = 1 if effect_matrix is None else 2
    fig_height = base_height * num_plots
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(fig_width, fig_height))
    
    if num_plots == 1:
        axes = [axes]

    sns.heatmap(
        pval_matrix.astype(float),
        mask=mask,
        annot=True,
        fmt=".4f", 
        cmap="PiYG_r",
        cbar=True,
        square=is_sym,
        vmin=0.0,
        vmax=1.0, 
        center=0.05, 
        annot_kws={"size": 10},
        ax=axes[0]
    )
    
    sym_text = "(Lower Triangle)" if is_sym else "(Full Matrix)"
    axes[0].set_title(f"{plot_title} - P-Values | {test_name} {sym_text}", fontsize=16, pad=15)
    axes[0].tick_params(axis="x", rotation=45, labelsize=10)
    axes[0].tick_params(axis="y", rotation=0, labelsize=10)

    if effect_matrix is not None:

        min_val = effect_matrix.min().min()
        eff_vmin = -1.0 if min_val < -0.01 else 0.0
        eff_cmap = "coolwarm" if eff_vmin == -1.0 else "Blues"
        
        sns.heatmap(
            effect_matrix.astype(float),
            mask=mask,
            annot=True,
            fmt=".3f",
            cmap=eff_cmap,
            cbar=True,
            square=is_sym,
            vmin=eff_vmin,
            vmax=1.0,
            center=0.0 if eff_vmin == -1.0 else None,
            annot_kws={"size": 10},
            ax=axes[1]
        )
        axes[1].set_title(f"{plot_title} - Test Result | {test_name} {sym_text}", fontsize=16, pad=15)
        axes[1].tick_params(axis="x", rotation=45, labelsize=10)
        axes[1].tick_params(axis="y", rotation=0, labelsize=10)

    plt.tight_layout()

    if save_plt:
        plt.savefig(output_path + f"{file_name}.png", bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)



# I STOPED HERE 
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