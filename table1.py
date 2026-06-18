# Table 1 - Baseline characteristics table stratified by a target variable.
#
# This is a Python port of the R `Table1()` function (see "Table1 R.txt"). It keeps
# the same layout and formatting conventions as the original:
#
#   Layout (columns):
#       Variables | Categories | Population | <one column per target level> | p-value [| Test]
#   Rows:
#       - first row  -> "Individuals / n / N"  (overall N and N per target level)
#       - numeric    -> "Mean (SD)", "Median (IQR)", "Missing (%)"
#       - categorical-> one "n (%)" row per category (+ a "Missing" category, see `catmiss`)
#
#   Formatting (matching the R defaults):
#       - thousands separator ("," big.mark) on every number
#       - `decimals` decimal places (R default = 1)
#       - numeric summaries: "mean (sd)" and "median (q25-q75)"
#       - category summaries: "count (pct%)"
#       - dichotomous "negative" rows (No/0/None/...) are dropped when `delzero=True`
#
# Designed to be pasted directly into eda_utils.py: it relies only on names already
# available there (np, pd, stats, and the existing `_pair_test`) and uses
# pd.api.types.is_numeric_dtype. The statistical test for every feature-vs-target
# pair is delegated to `_pair_test`; `_pair_test_name` mirrors the exact same
# branching so the (optional) "Test" column reports which test was run.


# Labels treated as the "negative"/zero level of a dichotomous variable. Mirrors the
# R check (nm == "No" | "no" | 0 | "0" | "None" | "none"); these rows are removed when
# `delzero=True`, so a yes/no variable only contributes its single "positive" row.
_ZERO_LABELS = {"no", "0", "0.0", "none", "false", "negative", "absent", "n"}


# ---------------------------------------------------------------------------
# Small formatting helpers (reproduce the R big.mark / nsmall formatting)
# ---------------------------------------------------------------------------

def _fmt_num(value, decimals):
    # Number with a thousands separator and a fixed number of decimals (R big.mark + nsmall).
    if value is None or pd.isna(value):
        return "NA"
    return f"{value:,.{decimals}f}"


def _fmt_int(value):
    # Integer count with a thousands separator (R big.mark, nsmall=0).
    if value is None or pd.isna(value):
        return "NA"
    return f"{int(round(value)):,}"


def _fmt_pval(p):
    # P-value formatting following the R convention of rounding to 3 decimals.
    if p is None or pd.isna(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{round(p, 3)}"


def _mean_sd(series, decimals):
    # "mean (sd)" summary for a numeric series (the R g1 helper).
    data = series.dropna()
    if data.empty:
        return " -- "
    return f"{_fmt_num(data.mean(), decimals)} ({_fmt_num(data.std(), decimals)})"


def _median_iqr(series, decimals):
    # "median (q25-q75)" summary for a numeric series (the R g2 helper).
    data = series.dropna()
    if data.empty:
        return " -- "
    q1, med, q3 = data.quantile([0.25, 0.50, 0.75])
    return f"{_fmt_num(med, decimals)} ({_fmt_num(q1, decimals)}-{_fmt_num(q3, decimals)})"


def _count_pct(count, total, decimals):
    # "count (pct%)" summary for a single category relative to its valid total.
    if total == 0:
        return " -- "
    return f"{_fmt_int(count)} ({_fmt_num(100 * count / total, decimals)}%)"


# ---------------------------------------------------------------------------
# Variable typing + test-name resolution (kept consistent with _pair_test)
# ---------------------------------------------------------------------------

def _var_type(series, normal_vars, cat_vars):
    """
    Classify a column into one of the four type strings understood by `_pair_test`:
    "normal", "non_normal" (continuous) or "dichotomous", "categorical".

    Follows the eda_utils convention (see split_variables_by_type): numeric dtype is
    continuous, the `category`/`bool`/object dtypes are categorical. A continuous
    variable is "normal" only when explicitly listed in `normal_vars`. Columns listed
    in `cat_vars` are forced to be categorical (mirrors the R `factorVars` argument).
    """
    name = series.name
    is_categorical = (
        name in cat_vars
        or series.dtype == bool
        or isinstance(series.dtype, pd.CategoricalDtype)
        or not pd.api.types.is_numeric_dtype(series)
    )
    if is_categorical:
        return "dichotomous" if series.dropna().nunique() == 2 else "categorical"
    return "normal" if name in normal_vars else "non_normal"


def _pair_test_name(type_a, type_b):
    """
    Return the name of the test that `_pair_test` would run for the given type pair.

    This mirrors the branching inside `_pair_test` one-to-one, so the "Test" column
    is guaranteed to match the p-value actually produced:
        - continuous vs continuous   -> Pearson / Spearman correlation
        - continuous vs dichotomous  -> Independent t-test / Mann-Whitney U
        - continuous vs categorical  -> One-way ANOVA / Kruskal-Wallis
        - categorical vs categorical -> Chi-square test
    """
    continuous = {"normal", "non_normal"}
    types = {type_a, type_b}

    if type_a in continuous and type_b in continuous:
        if type_a == "normal" and type_b == "normal":
            return "Pearson correlation"
        return "Spearman correlation"

    if types & continuous and "dichotomous" in types:
        cont_type = type_a if type_a in continuous else type_b
        return "Independent t-test" if cont_type == "normal" else "Mann-Whitney U"

    if types & continuous and "categorical" in types:
        cont_type = type_a if type_a in continuous else type_b
        return "One-way ANOVA" if cont_type == "normal" else "Kruskal-Wallis"

    return "Chi-square test"


# ---------------------------------------------------------------------------
# Row builders (one helper per kind of row in the R table)
# ---------------------------------------------------------------------------

def _numeric_row(var, label, series, target_series, groups, group_labels, summarise):
    # A numeric summary row (Mean (SD) / Median (IQR)) for the whole cohort and per group.
    row = {"Variables": var, "Categories": label, "Population": summarise(series)}
    for g, label_g in zip(groups, group_labels):
        row[label_g] = summarise(series[target_series == g])
    return row


def _missing_row(var, series, target_series, groups, group_labels, decimals):
    # The "Missing (%)" row, reported overall and per group (R Missing handling).
    def missing(s):
        n_missing = s.isna().sum()
        return _count_pct(n_missing, len(s), decimals) if n_missing > 0 else " -- "

    row = {"Variables": var, "Categories": "Missing (%)", "Population": missing(series)}
    for g, label_g in zip(groups, group_labels):
        row[label_g] = missing(series[target_series == g])
    return row


def _category_row(var, category, work, target_series, groups, group_labels, decimals):
    # One "n (%)" row for a single category of a categorical variable.
    def count_pct(s):
        return _count_pct(int((s == category).sum()), int(s.notna().sum()), decimals)

    row = {"Variables": var, "Categories": str(category), "Population": count_pct(work)}
    for g, label_g in zip(groups, group_labels):
        row[label_g] = count_pct(work[target_series == g])
    return row


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def table_one(df, target=None, variables=None, normal_vars=None, cat_vars=None,
              decimals=1, miss=3, catmiss=True, delzero=True, show_test=False):
    """
    Build a clinical 'Table 1' of baseline characteristics, optionally stratified by a
    target variable. This is a Python port of the R `Table1()` and keeps the same
    layout and formatting (see the module docstring).

    Numeric variables are summarised as Mean (SD), Median (IQR) and Missing (%).
    Categorical variables are summarised as n (%) per category. When a `target` is
    given, every feature is compared against it with the test chosen by `_pair_test`
    (driven by the eda_utils typing convention, i.e. the `normal_vars` list).

    :param df: Input data (your df.csv loaded with pandas).
    :param target: Name of the stratifying / target column. If None, only the
        "Population" column is produced (no group columns, p-values or tests).
    :param variables: Columns to include (in order). Defaults to every column except `target`.
    :param normal_vars: Continuous columns assumed to be normally distributed. These
        get parametric tests (t-test / ANOVA / Pearson) inside `_pair_test`; all other
        numeric columns get the non-parametric equivalent.
    :param cat_vars: Columns to force-treat as categorical (mirrors the R `factorVars`).
    :param decimals: Number of decimals used when formatting numbers (R default = 1).
    :param miss: Missing-value reporting level. >=1 adds a "Missing (%)" row to numeric
        variables; >=2 (with catmiss=False) adds one to categorical variables.
    :param catmiss: If True, missing values of a categorical variable are shown as a
        dedicated "Missing" category instead of a separate "Missing (%)" row.
    :param delzero: If True, drop the "negative" row (No/0/None/...) of dichotomous
        variables so only the informative "positive" row remains.
    :param show_test: Internal boolean flag. If True, an extra "Test" column is added
        reporting the name of the statistical test run between the target and each
        feature (the test that `_pair_test` performed).
    :return: The Table 1 as a pandas DataFrame.
    :rtype: pandas.DataFrame
    """
    normal_vars = set(normal_vars or [])
    cat_vars = set(cat_vars or [])
    if variables is None:
        variables = [col for col in df.columns if col != target]

    # Define the stratification groups from the target's levels (it is treated as a
    # grouping factor, exactly like `factor(y)` in R).
    if target is not None and target in df.columns:
        target_series = df[target]
        target_type = "dichotomous" if target_series.dropna().nunique() == 2 else "categorical"
        groups = sorted(target_series.dropna().unique().tolist(), key=lambda v: str(v))
        group_labels = [str(g) for g in groups]
    else:
        target = None
        target_series = None
        groups, group_labels = [], []

    rows = []

    # First row: "Individuals / n / N" with the overall N and the N of each group.
    individuals = {"Variables": "Individuals", "Categories": "n", "Population": _fmt_int(len(df))}
    for g, label_g in zip(groups, group_labels):
        individuals[label_g] = _fmt_int(int((target_series == g).sum()))
    rows.append(individuals)

    for var in variables:
        # Mirror the R behaviour: skip missing columns and columns with no usable data.
        if var not in df.columns:
            continue
        series = df[var]
        if series.dropna().nunique() < 1:
            continue

        vtype = _var_type(series, normal_vars, cat_vars)

        # Feature-vs-target test (delegated to _pair_test) computed once per variable.
        p_value, test_name = np.nan, ""
        if target is not None and series.dropna().nunique() > 1:
            pair = pd.concat([series, target_series], axis=1).dropna()
            feature_col, target_col = pair.iloc[:, 0], pair.iloc[:, 1]
            try:
                p_value = _pair_test(feature_col, target_col, vtype, target_type)
            except Exception:
                p_value = np.nan
            test_name = _pair_test_name(vtype, target_type)

        block = []

        if vtype in ("normal", "non_normal"):
            block.append(_numeric_row(var, "Mean (SD)", series, target_series, groups,
                                      group_labels, lambda s: _mean_sd(s, decimals)))
            block.append(_numeric_row(var, "Median (IQR)", series, target_series, groups,
                                      group_labels, lambda s: _median_iqr(s, decimals)))
            if miss >= 1:
                block.append(_missing_row(var, series, target_series, groups, group_labels, decimals))
        else:
            # Categorical: optionally fold missing values into a "Missing" category.
            work = series
            if catmiss and series.isna().any():
                work = series.astype("object").where(series.notna(), "Missing")

            # Category order follows R `table()`: sorted levels, with "Missing" last.
            categories = sorted([c for c in work.dropna().unique() if c != "Missing"], key=lambda v: str(v))
            if (work == "Missing").any():
                categories.append("Missing")

            for category in categories:
                # Drop the dichotomous "negative" row when delzero is on (R delzero logic).
                if (delzero and target is not None and vtype == "dichotomous"
                        and str(category).strip().lower() in _ZERO_LABELS):
                    continue
                block.append(_category_row(var, category, work, target_series, groups, group_labels, decimals))

            if not catmiss and miss >= 2:
                block.append(_missing_row(var, series, target_series, groups, group_labels, decimals))

        # Stamp the p-value (and optional test name) on the first row of the variable block.
        if block and target is not None:
            block[0]["p-value"] = _fmt_pval(p_value)
            if show_test:
                block[0]["Test"] = test_name
        rows.extend(block)

    table = pd.DataFrame(rows)

    # Assemble the final column order, matching the R layout.
    col_order = ["Variables", "Categories", "Population"] + group_labels
    if target is not None:
        col_order.append("p-value")
        if show_test:
            col_order.append("Test")

    for col in col_order:
        if col not in table.columns:
            table[col] = ""

    return table[col_order].fillna("")
