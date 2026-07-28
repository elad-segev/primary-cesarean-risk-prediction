import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import config as c
from typing import List, Tuple, Optional, Dict

# EDA Pipeline Functions

# ---------------------------------------------------------------------------
# CSV Saving Method
# ---------------------------------------------------------------------------

def save_df(df:pd.DataFrame , file_name: str, path: str = c.OUTPUT_DIR_FOR_TABLES):
    """
    Saves a pandas DataFrame as a CSV file.

    The function constructs a CSV filename if the provided file name does not
    already include the '.csv' extension and saves the DataFrame to the specified
    directory.

    The file is written using pandas' `to_csv()` method.

    :param df: DataFrame to be saved.
    :param file_name: Output file name. The '.csv' extension is added automatically
                    if not already provided.
    :param path: Destination directory where the file will be saved.
    :type df: pandas.DataFrame
    :type file_name: str
    :type path: str

    :return: None
    :rtype: None
    """
    fn = file_name if file_name.lower().endswith(".csv") else file_name + ".csv"
    p = path + "/" + fn
    df.to_csv(p)


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


def visualize_feature_distributions(df: pd.DataFrame, show:bool=False, save_plt:bool=True, end_file_name:str = "" ,output_path:str=c.OUTPUT_DIR_FOR_GRAPH):
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
    :parm end_file_name: the end of the filename template
    :param output_path: Directory where the plot image will be saved.
    :type df: pandas.DataFrame
    :type show: bool
    :type save_plt: bool
    :type end_file_name: str
    :type output_path: str
    :return: None
    :rtype: None
"""
    summary_num_table = describe_numerical(df)
    summary_cat_table = describe_categorical(df)
    cols_to_plot = df.select_dtypes(exclude=["string"]).columns
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
        plt.savefig(output_path + f"/visualize_feature_distributions{end_file_name}.png", bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)



def visualize_outliers_and_proportions(df: pd.DataFrame, show: bool = False, save_plt: bool = True, output_path:str=c.OUTPUT_DIR_FOR_GRAPH) -> Tuple[pd.DataFrame, dict, list]:
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


def split_variables_by_type(df: pd.DataFrame, normal_vars: List[str] = []) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Splits DataFrame variables into groups based on data type and normality assumption.

    The function categorizes columns into:
    - Categorical variables (category dtype)
    - Binary categorical variables (exactly 2 categories)
    - Multi-class categorical variables (more than 2 categories)
    - Continuous variables assumed to be normally distributed
    - Continuous variables not assumed to be normally distributed
    - Datetime variables

    The classification is based on both pandas dtype inference and a provided list
    of normally distributed variables.

    :param df: Input pandas DataFrame.
    :param normal_vars: List of column names assumed to follow a normal distribution (diff []).
    :type df: pandas.DataFrame
    :type normal_vars: List[str]
    :return: Tuple containing:
            (categorical variables,
            binary categorical variables,
            multi-class categorical variables,
            non-normal continuous variables,
            normal continuous variables,
            datetime variables)
    :rtype: Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]
    """
    cat_vars = df.select_dtypes(include=['category']).columns.tolist()
    bin_cat = [col for col in cat_vars if df[col].dropna().nunique() == 2]
    multy_cat = [col for col in cat_vars if df[col].dropna().nunique() > 2]
    
    all_cont_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    non_normal_cont_vars = [col for col in all_cont_vars if col not in normal_vars]
    normal_cont_vars = [col for col in all_cont_vars if col in normal_vars]
    datetime_vars = df.select_dtypes(include=['datetime']).columns.tolist()
    
    return cat_vars, bin_cat, multy_cat, non_normal_cont_vars, normal_cont_vars, datetime_vars


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
                
                cramers_v.loc[col1, col2] = cv
                pval_matrix.loc[col1, col2] = p
                
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
    test_name: str = "Test", file_name: str = "statistical_heatmap", show: bool = False,save_plt: bool = True, output_path: str = c.OUTPUT_DIR_FOR_GRAPH) -> None:
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
        plt.savefig(output_path + "/" + f"{file_name}.png", bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)


def significance_table(pval_matrix: pd.DataFrame, stat_matrix: pd.DataFrame, test_name: str, alpha: float = 0.05, effect_threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Creates a consolidated significance table from statistical test result matrices.

    The function converts p-value and statistic matrices into a long-format summary
    table that is easier to inspect, filter, export, or report.

    Supported matrix structures:
    - Symmetric matrices (e.g., Pearson or Spearman correlation matrices).
    Only the upper triangular portion is processed to avoid duplicate pairs.
    - Rectangular matrices (e.g., t-tests or Mann-Whitney tests).
    All valid cells are processed.

    For each variable pair, the output includes:
    - Statistical test name.
    - Variable names.
    - Test statistic or effect size.
    - P-value.
    - Statistical significance indicator.
    - Optional practical significance indicator based on a user-defined
    effect size threshold.

    :param pval_matrix: Matrix containing p-values.
    :param stat_matrix: Matrix containing test statistics or effect sizes.
    :param test_name: Name of the statistical test used.
    :param alpha: Significance threshold used to determine statistical significance.
    :param effect_threshold: Minimum absolute effect size required to be
                            considered practically meaningful. If None,
                            practical significance is not evaluated.
    :type pval_matrix: pandas.DataFrame
    :type stat_matrix: pandas.DataFrame
    :type test_name: str
    :type alpha: float
    :type effect_threshold: Optional[float]

    :return: Long-format summary table containing all valid statistical results.
    :rtype: pandas.DataFrame
    """
    is_sym = (pval_matrix.shape[0] == pval_matrix.shape[1]) and \
             (list(pval_matrix.index) == list(pval_matrix.columns))
    
    if is_sym:
        mask = np.triu(np.ones(pval_matrix.shape, dtype=bool), k=1)
        pvals_flat = pval_matrix.where(mask).stack()
        stats_flat = stat_matrix.where(mask).stack()
    else:
        pvals_flat = pval_matrix.stack()
        stats_flat = stat_matrix.stack()

    df_summary = pd.DataFrame({
        'P_Value': pvals_flat,
        'Statistic': stats_flat
    }).reset_index()
    
    df_summary.columns = ['Variable_A', 'Variable_B', 'P_Value', 'Statistic']
    
    df_summary = df_summary.dropna(subset=['P_Value', 'Statistic'])

    df_summary['P_Value'] = df_summary['P_Value'].astype(float).round(5)
    df_summary['Statistic'] = df_summary['Statistic'].astype(float).round(5)
    
    df_summary['Test_Name'] = test_name
    
    df_summary['Is_Significant'] = df_summary['P_Value'] < alpha
    

    if effect_threshold is not None:
        df_summary['Is_Meaningful_Effect'] = df_summary['Is_Significant'] & \
                                             (df_summary['Statistic'].abs() >= effect_threshold)
    
    final_cols = ['Test_Name', 'Variable_A', 'Variable_B', 'Statistic', 'P_Value', 'Is_Significant']
    if effect_threshold is not None:
        final_cols.append('Is_Meaningful_Effect')
        
    return df_summary[final_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Table 1 — descriptive comparison stratified by the outcome
# ---------------------------------------------------------------------------


def table1(df: pd.DataFrame ,normal_vars: List[str], non_normal_vars: List[str],
    multy_cat_vars: List[str], bin_cat_vars: List[str], target:str=c.TARGET_VAR,
    decimals: int = 3, show_test_label: bool=True, show_smd_label: bool=True) -> pd.DataFrame:
    """
    Generates a publication-ready Table 1 summarizing baseline characteristics by outcome group.

    This function produces a descriptive characteristics table commonly used in
    clinical and epidemiological studies. Variables are summarized according to
    their data type: normally distributed continuous variables are reported as
    mean (SD), non-normally distributed variables as median (IQR), and categorical
    variables as counts and percentages.

    For each variable, the appropriate statistical hypothesis test is selected
    automatically based on the variable type and the target variable. The function
    also computes the Standardized Mean Difference (SMD) for two-group comparisons,
    providing a measure of effect size that is independent of sample size.

    The resulting table includes overall population statistics, group-specific
    summaries, p-values, SMD values (optional), and the statistical test used
    (optional), making it suitable for direct inclusion in academic publications.

    :param df: Input pandas DataFrame containing the study dataset.
    :param normal_vars: List of normally distributed continuous variables.
    :param non_normal_vars: List of non-normally distributed continuous variables.
    :param multy_cat_vars: List of multicategory categorical variables.
    :param bin_cat_vars: List of binary categorical variables.
    :param target: Name of the grouping (outcome) variable.
    :param decimals: Number of decimal places used when formatting numeric summaries.
    :param show_test_label: If True, includes a column indicating the statistical
                            test performed for each variable.
    :param show_smd_label: If True, includes a column containing the Standardized
                        Mean Difference (SMD) for each variable.

    :type df: pandas.DataFrame
    :type normal_vars: List[str]
    :type non_normal_vars: List[str]
    :type multy_cat_vars: List[str]
    :type bin_cat_vars: List[str]
    :type target: str
    :type decimals: int
    :type show_test_label: bool
    :type show_smd_label: bool

    :return: Publication-ready Table 1 containing overall and group-specific
            descriptive statistics, p-values, optional SMD values, and optional
            statistical test labels.
    :rtype: pandas.DataFrame
    """

    if target not in df.columns:
        raise ValueError(f"Target variable '{target}' is not present in the DataFrame.")

    type_map = {
        **{v: "normal" for v in normal_vars},
        **{v: "non_normal" for v in non_normal_vars},
        **{v: "categorical" for v in multy_cat_vars},
        **{v: "dichotomous" for v in bin_cat_vars},
    }
    
    # target var definition
    target_type = type_map.get(target, "dichotomous")

    levels = sorted(df[target].dropna().unique())
    # One column per outcome level, named by the level value (as in the R file).
    group_cols = [str(g) for g in levels]


    def test_label(ta: str, tb: str) -> str:
        """
        Determines the appropriate statistical test based on variable types.

        The function maps combinations of variable types to their corresponding
        statistical hypothesis tests, following common statistical analysis rules:

        Variable type categories:
        - "normal": continuous normally distributed variables
        - "non_normal": continuous non-normally distributed variables
        - "dichotomous": binary categorical variables
        - "categorical": general categorical variables

        Test selection logic:
        - Continuous vs Continuous:
        - Both normal → Pearson correlation
        - Otherwise → Spearman correlation

        - Continuous vs Dichotomous:
        - Normal → Independent t-test
        - Non-normal → Mann-Whitney U test

        - Continuous vs Categorical (multi-class):
        - Normal → One-way ANOVA
        - Non-normal → Kruskal-Wallis test

        - Categorical vs Categorical:
        - Chi-square test of independence

        :param ta: Type of first variable.
        :param tb: Type of second variable.
        :type ta: str
        :type tb: str

        :return: Name of the recommended statistical test.
        :rtype: str
        """
        types = {ta, tb}
        continuous = {"normal", "non_normal"}

        # continuous vs continuous  ->  Pearson (both normal) / Spearman
        if ta in continuous and tb in continuous:
            return "Pearson correlation" if ta == tb == "normal" else "Spearman correlation"
        
        # continuous vs dichotomous  ->  t-test (normal) / Mann-Whitney
        if types & continuous and "dichotomous" in types:
            cont_type = ta if ta in continuous else tb
            return "t-test" if cont_type == "normal" else "Mann-Whitney"
        
        # continuous vs categorical  ->  ANOVA (normal) / Kruskal-Wallis (H-test)
        if types & continuous and "categorical" in types:
            cont_type = ta if ta in continuous else tb
            
            # (di)categorical vs (di)categorical  ->  Chi-Square
            return "One-way ANOVA" if cont_type == "normal" else "Kruskal-Wallis"
        return "Chi-square"
    

    def fmt_p(p: float) -> str:
        """
        Formats a p-value for display.

        The function returns:
        - An empty string if the value is missing (NaN).
        - "<0.001" for p-values smaller than 0.001.
        - Otherwise, the p-value rounded to three decimal places.

        :param p: P-value to format.
        :type p: float

        :return: Formatted p-value string.
        :rtype: str
        """
        if pd.isna(p):
            return ""
        return "<0.001" if p < 0.001 else f"{p:.3f}"


    def calc_smd(var_data: pd.Series, target_data: pd.Series, vtype: str, target_levels: list) -> float:
            """
            Calculates the Standardized Mean Difference (SMD) between two comparison groups.

            This function quantifies the magnitude of imbalance for a single variable
            between two target groups. Continuous variables are evaluated using the
            difference in group means divided by the pooled standard deviation, while
            categorical variables are evaluated using standardized differences in category
            proportions.

            For multicategory variables, the SMD is computed for each category separately,
            and the maximum SMD is returned as a summary measure of imbalance. If the target
            contains more than two groups, one of the groups has no observations, or the
            SMD cannot be computed, the function returns `NaN`.

            :param var_data: Series containing the variable to evaluate.
            :param target_data: Series defining the grouping variable.
            :param vtype: Variable type. Supported values are "normal",
                        "non_normal", and categorical types.
            :param target_levels: List containing the two target group labels.

            :type var_data: pandas.Series
            :type target_data: pandas.Series
            :type vtype: str
            :type target_levels: List

            :return: Standardized Mean Difference (SMD) between the two groups, or
                    `NaN` if the calculation cannot be performed.
            :rtype: float
            """
            if len(target_levels) != 2:
                return np.nan 

            g1 = var_data[target_data == target_levels[0]].dropna()
            g2 = var_data[target_data == target_levels[1]].dropna()

            if g1.empty or g2.empty:
                return np.nan

            if vtype in ("normal", "non_normal"):
                var1, var2 = g1.var(), g2.var()
                if var1 + var2 == 0: 
                    return 0.0
                pooled_sd = np.sqrt((var1 + var2) / 2)
                return abs(g1.mean() - g2.mean()) / pooled_sd
                
            else:
                cats = set(g1.unique()).union(set(g2.unique()))
                if len(cats) <= 2:
                    c = list(cats)[0] 
                    p1 = (g1 == c).mean()
                    p2 = (g2 == c).mean()
                    var_pool = (p1 * (1 - p1) + p2 * (1 - p2)) / 2
                    if var_pool == 0: 
                        return 0.0
                    return abs(p1 - p2) / np.sqrt(var_pool)
                else:
                    max_smd = 0.0
                    for c in cats:
                        p1 = (g1 == c).mean()
                        p2 = (g2 == c).mean()
                        var_pool = (p1 * (1 - p1) + p2 * (1 - p2)) / 2
                        if var_pool > 0:
                            smd_c = abs(p1 - p2) / np.sqrt(var_pool)
                            max_smd = max(max_smd, smd_c)
                    return max_smd


    def fmt_numeric(s: pd.Series, vtype: str, dec:int=4) -> str:
        """
        Formats a numeric pandas Series into a human-readable statistical summary string.

        The function provides two formatting styles depending on the variable type:
        - For normally distributed variables:
        Returns mean and standard deviation in the format:
        "mean (std)"

        - For non-normal variables:
        Returns median and interquartile range (IQR) in the format:
        "median (Q1-Q3)"

        Missing values are ignored. If the series is empty after removing NaNs,
        an empty string is returned.

        :param s: Input numeric Series.
        :param vtype: Variable type indicator ("normal" or non-normal).
        :param dec: Number of decimal places to round to.
        :type s: pandas.Series
        :type vtype: str
        :type dec: int

        :return: Formatted statistical summary string.
        :rtype: str
        """
        s = s.dropna()
        if s.empty:
            return ""
        if vtype == "normal":
            return f"{s.mean():.{dec}f} ({s.std():.{dec}f})"
        return f"{s.median():.{dec}f} ({s.quantile(0.25):.{dec}f}-{s.quantile(0.75):.{dec}f})"


    def fmt_categorical(s: pd.Series, value) -> str:
        """
        Formats the frequency of a categorical value as a count and percentage.

        The function calculates:
        - The number of occurrences of the specified value.
        - The percentage of non-missing observations represented by that value.

        If the Series contains no valid (non-missing) observations, only the count is
        returned.

        :param s: Series containing categorical values.
        :param value: Category whose frequency should be summarized.
        :type s: pandas.Series
        :type value: Any

        :return: Formatted string containing the count and percentage, or only the
                count if no valid observations exist.
        :rtype: str
        """
        # for categorical cases
        denom = s.notna().sum()
        n = int((s == value).sum())
        return f"{n} ({n / denom * 100:.1f}%)" if denom else f"{n}"


    rows = []

    n_row = {"V1": "Individuals", "V2": "n", "Pop": str(len(df))}
    for g, col in zip(levels, group_cols):
        n_row[col] = str(int((df[target] == g).sum()))
    n_row["SMD"] = ""
    n_row["pval"] = ""
    n_row["Test"] = ""

    rows.append(n_row)

    for var, vtype in type_map.items():
        if var == target:
            continue

        pair = df[[var, target]].dropna()
        try:
            p = _pair_test(pair[var], pair[target], vtype, target_type)
        except Exception:
            p = np.nan
        p_str = fmt_p(p)
        test = test_label(vtype, target_type)

        smd_val = calc_smd(df[var], df[target], vtype, levels)
        smd_str = f"{smd_val:.3f}" if pd.notna(smd_val) else ""

        if vtype in ("normal", "non_normal"):
            label = "Mean (SD)" if vtype == "normal" else "Median (IQR)"
            row = {"V1": var, "V2": label, "Pop": fmt_numeric(df[var], vtype, dec=decimals)}
            for g, col in zip(levels, group_cols):
                row[col] = fmt_numeric(df.loc[df[target] == g, var], vtype,dec=decimals)
            row["SMD"] = smd_str
            row["pval"] = p_str
            row["Test"] = test
            rows.append(row)

        else:
            cats = list(df[var].dropna().unique())
            try:
                cats = sorted(cats)
            except TypeError:
                pass

            for idx, cval in enumerate(cats):
                row = {"V1": var, "V2": str(cval), "Pop": fmt_categorical(df[var], cval)}
                
                for g, col in zip(levels, group_cols):
                    row[col] = fmt_categorical(df.loc[df[target] == g, var], cval)

                if idx == 0:
                    row["SMD"] = smd_str
                    row["pval"] = p_str
                    row["Test"] = test
                else:
                    row["SMD"] = ""
                    row["pval"] = ""
                    row["Test"] = ""
                    
                rows.append(row)

    col_order = ["V1", "V2", "Pop", *group_cols,"SMD", "pval"] if show_smd_label else ["V1", "V2", "Pop", *group_cols, "pval"]
    col_order = col_order + ["Test"] if show_test_label else col_order

    return pd.DataFrame(rows)[col_order]


def sensitivity_table1(df: pd.DataFrame, target_col: str, target_name: str,
                       normal_vars: List[str], non_normal_vars: List[str], 
                       multy_cat_vars: List[str], bin_cat_vars: List[str]) -> pd.DataFrame:
    """
    Create a sensitivity analysis Table 1 comparing records with and without available outcome data.

    This function assesses the potential impact of missing outcome data by creating
    a binary indicator that distinguishes observations with available target values
    from those with missing target values. It then generates a standard Table 1
    comparing baseline characteristics between these two groups using the specified
    continuous and categorical variables.

    The resulting table is relabeled with descriptive group names to facilitate
    interpretation and reporting in sensitivity analyses.

    :param df: Input pandas DataFrame containing the study data.
    :param target_col: Name of the outcome column used to identify missing values.
    :param target_name: Descriptive name of the target variable used for column labels.
    :param normal_vars: List of normally distributed continuous variables.
    :param non_normal_vars: List of non-normally distributed continuous variables.
    :param multy_cat_vars: List of multicategory categorical variables.
    :param bin_cat_vars: List of binary categorical variables.

    :type df: pandas.DataFrame
    :type target_col: str
    :type target_name: str
    :type normal_vars: List[str]
    :type non_normal_vars: List[str]
    :type multy_cat_vars: List[str]
    :type bin_cat_vars: List[str]

    :return: Table 1 comparing baseline characteristics between records with
            available outcome data and records with missing outcome data.
    :rtype: pandas.DataFrame
    """

    df_sen = df.copy()
    df_sen['target_indicator'] = df_sen[target_col].notna().astype("int")

    table_one = table1(
        df=df_sen,
        normal_vars=normal_vars,
        non_normal_vars=non_normal_vars,
        multy_cat_vars=multy_cat_vars,
        bin_cat_vars=bin_cat_vars,
        target="target_indicator"
    )

    table_one.rename(columns={"0": f"Missing {target_name} Data", 
                           "1": f"{target_name} Data Available"}, inplace=True)

    table_one.reset_index(drop=True, inplace=True)
    return table_one


# ---------------------------------------------------------------------------
# Treatment of clinical abnormalities
# ---------------------------------------------------------------------------
def apply_clinical_logic(df: pd.DataFrame, execute: list = [1.1, 1.2, 1.3, 1.5, 2.1, 2.2]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies predefined clinical consistency rules to a dataset and generates a report of all performed validation steps.

    This function validates the input dataset against a collection of clinically
    defined logical rules, including pairwise contradictions, temporal
    inconsistencies, and derived variable verification. Depending on the selected
    rules, inconsistent values are replaced with missing values (NaN) or derived
    variables are recalculated to ensure internal consistency.

    Each validation rule is independently logged, producing a structured report
    that summarizes the detected issue, the corrective action, the number of
    affected records, and whether the rule was executed. This provides a transparent
    audit trail of all data cleaning operations.

    :param df: Input pandas DataFrame containing the clinical dataset.
    :param execute: List of rule identifiers specifying which validation rules
                    should be applied. Rules not included in this list are
                    documented in the report but are not executed.

    :type df: pandas.DataFrame
    :type execute: List[float]

    :return:
        Tuple containing:
        - df_clean: Cleaned DataFrame after applying the selected clinical logic rules.
        - report_df: DataFrame summarizing all validation rules, detected issues,
        corrective actions, affected rows, and execution status.

    :rtype: Tuple[pandas.DataFrame, pandas.DataFrame]
    """
    df_clean = df.copy()
    action_logs = []

    # ==========================================
    # 1. Pairwise Contradictions (Ref: Doc Sec 1 - methodology_justifications.txt)
    # ==========================================
    
    # Ref 1.1: Weight Invariant
    calc_gain = df_clean['weight_at_admission'] - df_clean['weight_pre_pregnancy']
    rec_gain = df_clean['weight_gain']
    
    invalid_calc = calc_gain < -10.0
    invalid_rec = rec_gain < -10.0
    
    invalid_any = invalid_calc | invalid_rec
    
    both_invalid = (invalid_calc & invalid_rec).sum()
    only_calc_invalid = (invalid_calc & ~invalid_rec).sum() 
    only_rec_invalid = (invalid_rec & ~invalid_calc).sum() 
    total_invalid = invalid_any.sum()
    
    action_logs.append({
        "Category": "1. Pairwise Contradictions & Derived Metrics",
        "Issue": "Severe weight loss (> 10kg) in calculated weights or reported weight_gain",
        "Action Taken": "Set weight_pre_pregnancy, weight_at_admission, and weight_gain to NaN",
        "Affected Rows": total_invalid,
        "Details": f"Overlap: {both_invalid} | Only Calc: {only_calc_invalid} | Only weight_gain: {only_rec_invalid}",
        "Is Done": 1.1 in execute
    })
    
    if 1.1 in execute:
        df_clean.loc[invalid_calc, ['weight_pre_pregnancy', 'weight_at_admission']] = np.nan
        df_clean.loc[invalid_rec, 'weight_gain'] = np.nan


    # Ref 1.2: Amniotic Fluid Invariant
    if 'polyhydramnios' in df_clean.columns and 'oligohydramnios' in df_clean.columns:
        invalid_fluid = (df_clean['polyhydramnios'] == 1) & (df_clean['oligohydramnios'] == 1)
        n_invalid = invalid_fluid.sum()
        action_logs.append({
            "Category": "1. Pairwise Contradictions",
            "Issue": "Both polyhydramnios and oligohydramnios == 1",
            "Action Taken": "Set both flags to NaN",
            "Affected Rows": n_invalid,
            "Is Done": 1.2 in execute
        })
        if 1.2 in execute:
            df_clean.loc[invalid_fluid, ['polyhydramnios', 'oligohydramnios']] = np.nan


    # Ref 1.3: Hypertension Type Invariant
    if 'chronic_htn' in df_clean.columns and 'gestational_htn' in df_clean.columns:
        invalid_htn = (df_clean['chronic_htn'] == 1) & (df_clean['gestational_htn'] == 1)
        n_invalid = invalid_htn.sum()
        action_logs.append({
            "Category": "1. Pairwise Contradictions",
            "Issue": "Both chronic and gestational HTN == 1",
            "Action Taken": "Set both HTN flags to NaN",
            "Affected Rows": n_invalid,
            "Is Done": 1.3 in execute
        })
        if 1.3 in execute:
            df_clean.loc[invalid_htn, ['chronic_htn', 'gestational_htn']] = np.nan

    # Ref 1.4: Delivery Pathway Invariant
    if 'was_planned_cs' in df_clean.columns and 'induction' in df_clean.columns:
        invalid_induction = (df_clean['was_planned_cs'] == 1) & (df_clean['induction'] == 1)
        n_invalid = invalid_induction.sum()
        action_logs.append({
            "Category": "1. Pairwise Contradictions",
            "Issue": "Planned CS marked with Induction",
            "Action Taken": f"Set induction to NaN",
            "Affected Rows": n_invalid,
            "Is Done": 1.4 in execute
        })
        if 1.4 in execute:
            df_clean.loc[invalid_induction, 'induction'] = np.nan


    # Ref 1.5: Temporal Invariant
    if 'admission_date' in df_clean.columns and 'birth_date' in df_clean.columns:
        invalid_dates = df_clean['admission_date'] > df_clean['birth_date']
        n_invalid = invalid_dates.sum()
        action_logs.append({
            "Category": "1. Pairwise Contradictions",
            "Issue": "Admission date > Birth date",
            "Action Taken": "Set both dates to NaN",
            "Affected Rows": n_invalid,
            "Is Done": 1.5 in execute
        })
        if 1.5 in execute:
            df_clean.loc[invalid_dates, ['admission_date', 'birth_date']] = np.nan

    # ==========================================
    # 2. Derived & Math Invariants (Ref: Doc Sec 2)
    # ==========================================
    
  
    # Ref 2.1: HTN Cluster Recalculation
    htn_components = ['chronic_htn', 'gestational_htn', 'preeclampsia']
    if all(col in df_clean.columns for col in htn_components) and 'any_htn' in df_clean.columns:
        original_any_htn = df_clean['any_htn'].copy()
        htn_numeric = df_clean[htn_components].apply(pd.to_numeric, errors='coerce')
        calculated_any_htn = htn_numeric.max(axis=1)
        n_changed = (original_any_htn != calculated_any_htn).sum()
        
        action_logs.append({
            "Category": "2. Derived Variables",
            "Issue": "'any_htn' did not match the maximum of its components",
            "Action Taken": "Recalculated 'any_htn' safely using numeric conversion",
            "Affected Rows": n_changed,
            "Is Done": 2.1 in execute
        })
        if 2.1 in execute:
            df_clean['any_htn'] = calculated_any_htn
    

    # Ref 2.2: BMI Mathematical Verification
    if 'weight_pre_pregnancy' in df_clean.columns and 'height_cm' in df_clean.columns:
        n_calculated = (df_clean['weight_pre_pregnancy'].notna() & df_clean['height_cm'].notna()).sum()            
        action_logs.append({
            "Category": "2. Derived Variables",
            "Issue": "Transparent BMI calculation preferred over original 'bmi_computed'",
            "Action Taken": "Created new 'bmi_pre_pregnancy' column based on pre-pregnancy weight",
            "Affected Rows": n_calculated,
            "Is Done": 2.2 in execute
        })
        if 2.2 in execute:
            df_clean['bmi_pre_pregnancy'] = df_clean['weight_pre_pregnancy'] / ((df_clean['height_cm'] / 100) ** 2)


    # If no issues were found, add a clean bill of health to the log
    if not action_logs:
        action_logs.append({
            "Category": "General",
            "Issue": "No clinical logic violations found",
            "Action Taken": "None",
            "Affected Rows": 0,
            "Is Done": False,
        })

    report_df = pd.DataFrame(action_logs)
    
    return df_clean, report_df


# ---------------------------------------------------------------------------
# Quick outliers-value overview
# ---------------------------------------------------------------------------


def plot_outliers_heatmap(df: pd.DataFrame, outliers_matrix: pd.DataFrame, rand: bool = False, save_plt: bool = True, show: bool = False, output_path: str = c.OUTPUT_DIR_FOR_GRAPH, end: str = "") -> None:
    """
    Creates and optionally saves a heatmap visualization of detected outliers and
    rare categorical values.

    The function visualizes a boolean matrix where:
    - Rows represent observations.
    - Columns represent variables.
    - True values indicate detected outliers or rare categories.

    The heatmap provides an overview of data quality issues by showing the
    distribution of flagged values across the dataset.

    The function calculates:
    - Total number of flagged cells.
    - Percentage of flagged cells relative to the entire matrix.

    For large datasets, optional random sampling can be applied to limit the number
    of displayed rows while maintaining visualization readability.

    :param df: Original DataFrame (used for compatibility and future extensions).
    :param outliers_matrix: Boolean DataFrame indicating outliers or rare categories.
    :param rand: Whether to randomly sample rows before visualization.
    :param save_plt: Whether to save the generated heatmap image.
    :param show: Whether to display the plot.
    :param output_path: Directory path where the image will be saved.
    :param end: Optional suffix added to the output filename.
    :type df: pandas.DataFrame
    :type outliers_matrix: pandas.DataFrame
    :type rand: bool
    :type save_plt: bool
    :type show: bool
    :type output_path: str | Path
    :type end: str

    :return: None
    :rtype: None
    """
    plot_matrix = outliers_matrix.copy()
    
    n_outliers = plot_matrix.sum().sum()
    pct_outliers = round(100 * n_outliers / plot_matrix.size, 2)
    
    rand_msg = ""
    rand_file_suffix = ""
    max_visual_rows = 10000 

    if rand and len(plot_matrix) > max_visual_rows:
        plot_matrix = plot_matrix.sample(n=max_visual_rows, random_state=42).sort_index()
        rand_msg = f" | Sampled {max_visual_rows:,} rows"
        rand_file_suffix = "_sampled" # Safer for filenames than using rand_msg

    # Dynamically adjust width based on number of columns
    plt.figure(figsize=(max(12, len(plot_matrix.columns) // 2), 6))
    
    # cmap="Reds" creates a good visual cue for outliers (warnings)
    sns.heatmap(plot_matrix, cbar=False, cmap="Reds", yticklabels=False)
    
    plt.title(
        f"Outliers & Rare Categories Map  |  {n_outliers:,} flagged cells ({pct_outliers}% of total {end}{rand_msg})", 
        fontsize=13, pad=15
    )
    
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()

    if save_plt:
        plt.savefig(output_path + f"/outliers_heatmap{end}{rand_file_suffix}.png", bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------------
# Missingness analysis  (MCAR vs MAR)
# ---------------------------------------------------------------------------

def missingness_mechanism_table(df: pd.DataFrame, normal_vars: List[str], non_normal_vars: List[str],
    multy_cat_vars: List[str], bin_cat_vars: List[str], alpha: float = 0.05, min_obs: int = 3,
    optional_labels:list=["num of affecting columns", "associated_variables", "structural_variables", "MAR"]) -> pd.DataFrame:
    """
    Evaluates the missing-data mechanism for variables containing missing values.

    The function investigates whether the missingness of each variable is associated
    with other variables in the dataset by creating a binary missingness indicator
    (1 = missing, 0 = observed) and performing an appropriate statistical test
    against every other variable.

    Each comparison is performed using an internal helper function that selects
    the appropriate statistical test according to the variable types.

    Variables whose missingness is significantly associated with at least one
    other variable (p ≤ alpha) are classified as MAR (Missing At Random).
    Variables with no detected associations are classified as MCAR
    (Missing Completely At Random).

    Additionally, variables that cannot be tested because one of the compared
    variables contains only a single observed value are recorded as structural
    variables.

    The returned table can optionally include:
    - Number of associated variables.
    - List of statistically associated variables with p-values.
    - List of structural variables.
    - MCAR indicator.
    - MAR indicator.

    :param df: Input pandas DataFrame.
    :param normal_vars: List of normally distributed continuous variables.
    :param non_normal_vars: List of non-normally distributed continuous variables.
    :param multy_cat_vars: List of multi-category categorical variables.
    :param bin_cat_vars: List of binary categorical variables.
    :param alpha: Significance level used to determine statistical association.
    :param min_obs: Minimum number of paired observations required to perform a test.
    :param optional_labels: Columns to include in the returned summary table.
    :type df: pandas.DataFrame
    :type normal_vars: List[str]
    :type non_normal_vars: List[str]
    :type multy_cat_vars: List[str]
    :type bin_cat_vars: List[str]
    :type alpha: float
    :type min_obs: int
    :type optional_labels: list

    :return: Summary table describing the estimated missing-data mechanism for
            each variable containing missing values.
    :rtype: pandas.DataFrame
    """
    type_map = {
        **{v: "normal" for v in normal_vars},
        **{v: "non_normal" for v in non_normal_vars},
        **{v: "categorical" for v in multy_cat_vars},
        **{v: "dichotomous" for v in bin_cat_vars},
    }
    
    variables = list(type_map.keys())
    counts = {}
    p_vals: Dict[Tuple[str, str], float] = {}
    structural_texts: Dict[str, List[str]] = {}

    for var in variables:
        # Skip if the column has no missing values
        if not df[var].isnull().any():
            continue
            
        # Create dichotomous indicator for missingness (1 = missing, 0 = not missing)
        indicator = df[var].isnull().astype(int)

        n_associated = 0
        for other in variables:
            if other == var:
                continue
                
            pair = pd.concat([indicator, df[other]], axis=1).dropna()
            if len(pair) < min_obs:
                continue
                
            if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
                if var not in structural_texts:
                    structural_texts[var] = []
                structural_texts[var].append(other)
                continue 

            p = _pair_test(pair.iloc[:, 0], pair.iloc[:, 1], "dichotomous", type_map[other])
            
            if pd.notna(p) and p <= alpha:
                n_associated += 1
                p_vals[(var, other)] = p

        counts[var] = n_associated

    # Build the final DataFrame
    table = pd.DataFrame({"num of affecting columns": pd.Series(counts, dtype=int)})
    table["MAR"] = table["num of affecting columns"] > 0
    table["MCAR"] = ~table["MAR"]
    table = table.reset_index().rename(columns={"index": "Variable"})
   

    assoc_texts:Dict[str:List[str]] = {}
    for (var, other), p in p_vals.items():
        text = f"{other} (p={p:.4f})"
        if var not in assoc_texts:
            assoc_texts[var] = []
        assoc_texts[var].append(text)

    assoc_strings = {var: ", ".join(items) for var, items in assoc_texts.items()}

    table["associated_variables"] = table["Variable"].map(assoc_strings).fillna("")

    structural_strings = {var: ", ".join(items) for var, items in structural_texts.items()}
    table["structural_variables"] = table["Variable"].map(structural_strings).fillna("")

    master_order = ["Variable", "num of affecting columns", "associated_variables", "structural_variables", "MCAR", "MAR"]
    user_requests = set(optional_labels) if optional_labels else set()
    user_requests.update(["Variable", "MCAR"])
    final_cols = [col for col in master_order if col in user_requests and col in table.columns]

    return table[final_cols]


def _pair_test(a: pd.Series, b: pd.Series, type_a: str, type_b: str) -> float:
    """
    Performs the appropriate statistical test between two variables based on their
    data types and returns the resulting p-value.

    The function automatically selects the statistical test according to the
    variable types:

    - Continuous vs Continuous:
        - Pearson correlation for two normally distributed variables.
        - Spearman correlation when at least one variable is non-normal.

    - Continuous vs Dichotomous:
        - Independent t-test for normal continuous variables.
        - Mann-Whitney U test for non-normal continuous variables.

    - Continuous vs Categorical (>2 groups):
        - One-way ANOVA for normal continuous variables.
        - Kruskal-Wallis test for non-normal continuous variables.

    - Categorical vs Categorical:
        - Chi-square test of independence.

    The function is mainly used for missingness mechanism analysis, where one
    variable is often a binary missingness indicator.

    :param a: First variable series.
    :param b: Second variable series.
    :param type_a: Data type classification of the first variable.
    :param type_b: Data type classification of the second variable.
    :type a: pandas.Series
    :type b: pandas.Series
    :type type_a: str
    :type type_b: str

    :return:
        P-value obtained from the selected statistical test.
    :rtype: float
    """
    types = {type_a, type_b}
    continuous = {"normal", "non_normal"}

    # continuous vs continuous  ->  Pearson (both normal) / Spearman
    if type_a in continuous and type_b in continuous:
        if type_a == "normal" and type_b == "normal":
            return stats.pearsonr(a, b)[1]
        return stats.spearmanr(a, b)[1]

    # continuous vs dichotomous  ->  t-test (normal) / Mann-Whitney
    if types & continuous and "dichotomous" in types:
        (cont, cont_type, grp) = (a, type_a, b) if type_a in continuous else (b, type_b, a)
        groups = [cont[grp == g] for g in pd.unique(grp)]

        if len(pd.unique(grp)) < 2:
            return np.nan

        if cont_type == "normal":
            return stats.ttest_ind(*groups, equal_var=True)[1]
        return stats.mannwhitneyu(*groups, alternative="two-sided")[1]

    # continuous vs categorical  ->  ANOVA (normal) / Kruskal-Wallis (H-test)
    if types & continuous and "categorical" in types:
        (cont, cont_type, grp) = (a, type_a, b) if type_a in continuous else (b, type_b, a)
        groups = [cont[grp == g] for g in pd.unique(grp)]
        if cont_type == "normal":
            return stats.f_oneway(*groups)[1]
        return stats.kruskal(*groups)[1]

    # (di)categorical vs (di)categorical  ->  Chi-Square
    return stats.chi2_contingency(pd.crosstab(a, b))[1]


# ---------------------------------------------------------------------------
# Quick missing-value overview
# ---------------------------------------------------------------------------

def plot_simple_missing_heatmap(df: pd.DataFrame, rand:bool = False, save_plt:bool = True, show:bool = False, output_path:str=c.OUTPUT_DIR_FOR_GRAPH, end:str = "")-> None:
    """
    Plots a heatmap visualization of missing values in a DataFrame.

    The function generates a binary heatmap where:
    - Missing values (NaN) are highlighted.
    - Observed values are shown as empty/neutral cells.

    This visualization helps quickly identify:
    - Missing data patterns
    - Structural missingness across rows/columns
    - Random vs clustered missing behavior

    Behavior:
    - Optionally samples the dataset for large DataFrames to improve performance.
    - Displays summary statistics in the plot title:
        - Total missing cells
        - Percentage of missing data
    - Supports saving the plot to disk with optional labeling.

    :param df: Input pandas DataFrame.
    :param rand: If True, randomly samples up to 10,000 rows for visualization when dataset is large.
    :param save_plt: Whether to save the generated plot as an image file.
    :param show: Whether to display the plot interactively.
    :param output_path: Directory where the plot image will be saved.
    :param end: Optional suffix added to filename and plot title (e.g., version or timestamp tag).
    :type df: pandas.DataFrame
    :type rand: bool
    :type save_plt: bool
    :type show: bool
    :type output_path: str
    :type end: str

    :return: None
    """
    plot_df = df.copy()
    n_missing = plot_df.isnull().sum().sum()
    pct_missing = round(100 * n_missing / df.size, 2)
    rand_msg = ""
    max_visual_rows = 10000 

    if rand and len(df) > max_visual_rows:
        plot_df = plot_df.sample(n=max_visual_rows, random_state=42).sort_index()
        rand_msg = f" | Sampled {max_visual_rows:,} rows"
    

    plt.figure(figsize=(max(12, len(plot_df.columns) // 2), 6))
    sns.heatmap(plot_df.isnull(), cbar=False, cmap="crest", yticklabels=False)
    plt.title(
        f"Missing-value map  |  {n_missing:,} missing cells ({pct_missing}% of total {end}{rand_msg})", fontsize=13, pad=15)
    
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()

    if save_plt:
        plt.savefig(output_path + f"/missing_heatmap{end}{rand_msg}.png", bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close()



# ---------------------------------------------------------------------------
# Rare-category detection
# ---------------------------------------------------------------------------

def categorical_frequencies_table(df: pd.DataFrame, cat_vars: List[str], threshold: float = 0.05) -> pd.DataFrame:
    """
    Creates a frequency distribution table for categorical variables.

    The function computes absolute counts and relative percentages for each category
    within the specified categorical variables. Missing values are treated as a
    separate explicit category ("Missing") to ensure full representation of the data
    distribution.

    For each variable:
    - Counts are computed using value_counts(dropna=False).
    - Percentages are computed relative to the full column size.
    - Categories below a given frequency threshold are flagged as rare.

    Behavior:
    - Missing values are included as a distinct category labeled "Missing".
    - Rare categories are identified when their percentage is below the threshold.
    - Output is a long-format table (one row per category per variable).

    :param df: Input pandas DataFrame.
    :param cat_vars: List of categorical variable names to analyze.
    :param threshold: Proportion threshold for defining rare categories (e.g., 0.05 = 5%).
    :type df: pandas.DataFrame
    :type cat_vars: List[str]
    :type threshold: float

    :return:
        DataFrame containing:
        - Variable name
        - Category value
        - Count of occurrences
        - Percentage of total
        - Rare category flag
    :rtype: pandas.DataFrame
    """
    rows = []

    for col in cat_vars:
        if col not in df.columns:
            continue

        # Using dropna=False ensures missing values are counted as a distinct category
        counts = df[col].value_counts(dropna=False)
        percentages = df[col].value_counts(dropna=False, normalize=True) * 100

        for cat_val, count in counts.items():
            pct = percentages[cat_val]
            
            # Identify if the category is the missing values group
            is_missing = pd.isna(cat_val)
            cat_name = "Missing" if is_missing else str(cat_val)
            
            # Flag as rare if it's below the threshold (usually we don't flag 'Missing' as rare to consolidate)
            is_rare = bool(pct < (threshold * 100))
            
            rows.append({
                "Variable": col,
                "Category": cat_name,
                "Count": int(count),
                "Percentage (%)": round(pct, 2),
                "Is_Rare": is_rare
            })

    report_df = pd.DataFrame(rows)
    
    return report_df


def set_log(df:pd.DataFrame, numeric_cols:list, drop: bool=False) -> pd.DataFrame:
    """
    Applies a natural logarithmic transformation to selected numeric columns.

    This function creates new features by applying the natural logarithm
    transformation using `np.log1p` (i.e., log(1 + x)), which safely handles zero
    values and is therefore well suited for clinical and tabular datasets. The
    transformed columns are added to the DataFrame with the naming convention
    `log(column_name)`, while the original columns are preserved unless explicitly
    removed.

    If any specified column does not exist in the DataFrame, a `KeyError` is raised.

    :param df: Input pandas DataFrame containing the numeric variables.
    :param numeric_cols: List of numeric column names to transform.
    :param drop: If True, removes the original columns after creating the
                transformed versions. Defaults to False.

    :type df: pandas.DataFrame
    :type numeric_cols: List[str]
    :type drop: bool

    :return: DataFrame containing the logarithmically transformed features.
    :rtype: pandas.DataFrame
    """
    df_transformed = df.copy()
    
    for col in numeric_cols:
        if col not in df_transformed.columns:
            raise KeyError(f"Column '{col}' not found in the DataFrame.")
            
        new_col_name = f"log({col})"
        
        df_transformed[new_col_name] = np.log1p(df_transformed[col])
        
    if drop:
        df_transformed = df_transformed.drop(columns=numeric_cols)
        
    return df_transformed