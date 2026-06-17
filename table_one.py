# Table 1 - Baseline characteristics table stratified by a target variable.
# Designed to be pasted directly into eda_utils.py: it relies only on names already
# available there (np, pd, stats, _cramers_v) and uses pd.api.types.is_numeric_dtype.
# The design follows the conventions used in first_submission_in_data_science_part_1:
#   - numeric / categorical split via pd.api.types.is_numeric_dtype
#   - numeric variables are treated as non-normal -> median [Q1, Q3] and
#     non-parametric tests (Mann-Whitney U for 2 groups, Kruskal-Wallis for >2)
#   - categorical variables -> n (%) per category with a Chi-Square test and Cramer's V
#   - results are collected as a list of dicts and returned as a DataFrame


def missing_percent_str(series):
    # Same missing-percent formatting convention used in describe_by_type_numerical / _category
    pct = (series.isnull().sum() / len(series)) * 100
    return f"{int(pct)}%" if pct % 10 == 0 else f"{round(pct, 3)}%"


def median_iqr_str(series, decimals=2):
    # "median [Q1, Q3]" summary for a numeric series (non-normal convention)
    data = series.dropna()
    if data.empty:
        return "-"
    q1, med, q3 = data.quantile([0.25, 0.50, 0.75])
    return f"{round(med, decimals)} [{round(q1, decimals)}, {round(q3, decimals)}]"


def count_percent_str(count, total, decimals=2):
    # "n (pct%)" summary for a category count relative to its valid total
    if total == 0:
        return f"{count} (-)"
    return f"{count} ({round(100 * count / total, decimals)}%)"


def table_one(df, target, decimals=2):
    """
    Build a clinical 'Table 1' of baseline characteristics stratified by the target variable.

    Numeric variables are summarised as median [Q1, Q3] for the whole cohort and for
    each target group, and compared with Mann-Whitney U (2 groups) or Kruskal-Wallis (>2 groups).

    Categorical variables are summarised as n (%) per category for the whole cohort and for
    each target group, and compared with a Chi-Square test of independence (with Cramer's V).

    The missing percentage of every variable is reported as well.
    Returns the table as a pandas DataFrame.
    """
    # Split the variables by type, just like in the EDA notebook
    num_vars = [var for var in df.columns if var != target and is_numeric_dtype(df[var])]
    cat_vars = [var for var in df.columns if var != target and not is_numeric_dtype(df[var])]

    # The groups are defined by the categories of the target variable
    groups = sorted(df[target].dropna().unique().tolist(), key=lambda x: str(x))
    group_labels = [f"{target}={g}" for g in groups]

    results_data = []

    # Header row with the sample size of the whole cohort and of every group
    header = {"Variable": "N", "Category": "", "Overall": df[target].notna().sum()}
    for g, label in zip(groups, group_labels):
        header[label] = int((df[target] == g).sum())
    header.update({"Missing": "", "Test Type": "", "p value": "", "Cramer's V": ""})
    results_data.append(header)

    # Numeric variables: median [Q1, Q3] per group + non-parametric test
    for num_var in num_vars:
        groups_collection = [df[df[target] == g][num_var].dropna() for g in groups]
        valid_groups = [group for group in groups_collection if len(group) > 0]

        # Choose the statistical test by the number of groups (same logic as the notebook)
        if len(valid_groups) < 2 or any(len(group) < 2 for group in valid_groups):
            test_type, p_val = "-", np.nan
        elif len(valid_groups) == 2:
            _, p_val = stats.mannwhitneyu(*valid_groups)
            test_type = "Mann-Whitney U"
        else:
            _, p_val = stats.kruskal(*valid_groups)
            test_type = "Kruskal-Wallis"

        row = {"Variable": num_var, "Category": "median [Q1, Q3]", "Overall": median_iqr_str(df[num_var], decimals)}
        for g, label in zip(groups, group_labels):
            row[label] = median_iqr_str(df[df[target] == g][num_var], decimals)
        row.update({
            "Missing": missing_percent_str(df[num_var]),
            "Test Type": test_type,
            "p value": "-" if pd.isna(p_val) else f"{p_val:.3e}",
            "Cramer's V": "none",
        })
        results_data.append(row)

    # Categorical variables: n (%) per category + Chi-Square test and Cramer's V
    for cat_var in cat_vars:
        filtered_data = df[[cat_var, target]].dropna()
        contingency_table = pd.crosstab(filtered_data[cat_var], filtered_data[target])

        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            p_val, cv = np.nan, np.nan
        else:
            _, p_val, _, _ = stats.chi2_contingency(contingency_table)
            cv = _cramers_v(contingency_table)

        # Parent row holding the test result for the whole variable
        parent = {"Variable": cat_var, "Category": "n (%)", "Overall": ""}
        for label in group_labels:
            parent[label] = ""
        parent.update({
            "Missing": missing_percent_str(df[cat_var]),
            "Test Type": "Chi-Square",
            "p value": "-" if pd.isna(p_val) else f"{p_val:.3e}",
            "Cramer's V": "-" if pd.isna(cv) else round(cv, 4),
        })
        results_data.append(parent)

        # One indented row per category with its count and percentage
        for category in df[cat_var].dropna().value_counts().index.tolist():
            child = {"Variable": "", "Category": f"   {category}",
                     "Overall": count_percent_str(int((df[cat_var] == category).sum()), df[cat_var].notna().sum(), decimals)}
            for g, label in zip(groups, group_labels):
                sub = df[df[target] == g][cat_var]
                child[label] = count_percent_str(int((sub == category).sum()), sub.notna().sum(), decimals)
            child.update({"Missing": "", "Test Type": "", "p value": "", "Cramer's V": ""})
            results_data.append(child)

    # Assemble the final table (list of dicts -> DataFrame), keeping a readable column order
    col_order = ["Variable", "Category", "Overall"] + group_labels + ["Missing", "Test Type", "p value", "Cramer's V"]
    table = pd.DataFrame(results_data)[col_order]

    return table
