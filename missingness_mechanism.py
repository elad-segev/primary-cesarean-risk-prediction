# -*- coding: utf-8 -*-
"""Missingness mechanism classification (MCAR vs MAR).

* the seven statistical tests of the project:
  Pearson, Spearman, Mann-Whitney U, Student's t-test, Chi-Square,
  ANOVA, Kruskal-Wallis,
* Shapiro-Wilk to decide whether a continuous variable is normal,

Idea
----
For a variable ``v`` that contains missing values we build a **dichotomous**
indicator ``R = v.isnull()`` (1 = missing, 0 = present) and test whether that
indicator is *associated* with every other variable ``x``:

* a significant association (p <= alpha) means the probability of ``v`` being
  missing depends on ``x``  ->  evidence for **MAR**;
* if no other variable is associated with the missingness of ``v`` we have no
  evidence that it depends on anything  ->  we call it **MCAR**
  (anything that is not MAR is treated as MCAR).

The seven tests are selected according to the **three variable types** of the
columns being compared:

    * Dichotomous variables  (exactly 2 categories)
    * Categorical variables  (more than 2 categories)
    * Continuous variables

+------------------+------------------+--------------------------------------+
| variable A type  | variable B type  | test (normal -> / non-normal ->)     |
+==================+==================+======================================+
| continuous       | continuous       | Pearson      / Spearman              |
| continuous       | dichotomous      | t-test       / Mann-Whitney U        |
| continuous       | categorical      | ANOVA        / Kruskal-Wallis        |
| dichotomous      | dichotomous      | Chi-Square                           |
| dichotomous      | categorical      | Chi-Square                           |
| categorical      | categorical      | Chi-Square                           |
+------------------+------------------+--------------------------------------+

Because a missing-value indicator is always **dichotomous**, the missingness
analysis itself fires the dichotomous rows of the table: t-test / Mann-Whitney
against continuous columns and Chi-Square against (di)categorical columns.
The dispatcher below, however, implements *all seven* tests so it can be reused
for any pair of variables.

All public functions take the variables to analyse **as inputs** (the three
type lists), so nothing about a specific dataset is hard-coded.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- #
# Variable-type constants                                                      #
# --------------------------------------------------------------------------- #
CONTINUOUS = "continuous"
CATEGORICAL = "categorical"     # > 2 categories
DICHOTOMOUS = "dichotomous"     # exactly 2 categories


# --------------------------------------------------------------------------- #
# Normality helper (Shapiro-Wilk, mirrors the notebook logic)                 #
# --------------------------------------------------------------------------- #
def infer_normal_vars(
    df: pd.DataFrame,
    continuous_vars: Sequence[str],
    alpha: float = 0.05,
    max_sample: int = 5000,
    random_state: int = 42,
) -> List[str]:
    """Return the subset of ``continuous_vars`` that look normally distributed.

    Uses the Shapiro-Wilk test exactly like the original notebook: when a
    column has more than ``max_sample`` non-missing values it is randomly
    sub-sampled (``random_state`` fixed for reproducibility) before testing.
    A column is considered normal when ``p >= alpha``.
    """
    normal_vars: List[str] = []
    for col in continuous_vars:
        data = df[col].dropna()
        if len(data) < 3:                       # Shapiro needs >= 3 points
            continue
        sample = (
            data.sample(n=max_sample, random_state=random_state)
            if len(data) > max_sample
            else data
        )
        _, p_val = stats.shapiro(sample)
        if p_val >= alpha:
            normal_vars.append(col)
    return normal_vars


# --------------------------------------------------------------------------- #
# Type lookup                                                                  #
# --------------------------------------------------------------------------- #
def _build_type_map(
    continuous_vars: Sequence[str],
    categorical_vars: Sequence[str],
    dichotomous_vars: Sequence[str],
) -> Dict[str, str]:
    """Map every supplied variable name to one of the three type constants."""
    type_map: Dict[str, str] = {}
    for col in continuous_vars:
        type_map[col] = CONTINUOUS
    for col in categorical_vars:
        type_map[col] = CATEGORICAL
    for col in dichotomous_vars:
        type_map[col] = DICHOTOMOUS
    return type_map


# ------------------------------------- #
# The seven-test dispatcher                                                    #
# ------------------------------------- #
def run_test_by_type(
    series_a: pd.Series,
    series_b: pd.Series,
    type_a: str,
    type_b: str,
    a_normal: bool = False,
    b_normal: bool = False,
    min_group_size: int = 2,
) -> Tuple[str, float, float]:
    """Pick and run one of the seven tests for a pair of variables.

    Parameters
    ----------
    series_a, series_b : aligned (same index) pandas Series.
    type_a, type_b     : one of CONTINUOUS / CATEGORICAL / DICHOTOMOUS.
    a_normal, b_normal : whether the corresponding continuous series is normal
                         (ignored for non-continuous variables).
    min_group_size     : groups smaller than this are dropped before group
                         comparison tests.

    Returns
    -------
    (test_name, statistic, p_value).  ``p_value`` is ``np.nan`` when the test
    could not be computed (too little data / degenerate contingency table).
    """
    # Pairwise drop of missing values so both series stay aligned.
    pair = pd.concat([series_a, series_b], axis=1).dropna()
    if len(pair) < 3:
        return ("insufficient data", np.nan, np.nan)
    a = pair.iloc[:, 0]
    b = pair.iloc[:, 1]

    types = {type_a, type_b}

    # 1/2 — continuous vs continuous -> Pearson (normal) / Spearman (else)
    if type_a == CONTINUOUS and type_b == CONTINUOUS:
        if a_normal and b_normal:
            stat, p = stats.pearsonr(a, b)
            return ("Pearson", float(stat), float(p))
        stat, p = stats.spearmanr(a, b)
        return ("Spearman", float(stat), float(p))

    # 3/4 — continuous vs dichotomous -> t-test (normal) / Mann-Whitney (else)
    if CONTINUOUS in types and DICHOTOMOUS in types:
        if type_a == CONTINUOUS:
            cont, grp, cont_normal = a, b, a_normal
        else:
            cont, grp, cont_normal = b, a, b_normal
        groups = [cont[grp == g] for g in pd.unique(grp)]
        groups = [g for g in groups if len(g) >= min_group_size]
        if len(groups) < 2:
            return ("insufficient groups", np.nan, np.nan)
        if cont_normal:
            stat, p = stats.ttest_ind(*groups, equal_var=True)
            return ("T-test", float(stat), float(p))
        stat, p = stats.mannwhitneyu(*groups, alternative="two-sided")
        return ("Mann-Whitney U", float(stat), float(p))

    # 5/6 — continuous vs categorical -> ANOVA (normal) / Kruskal-Wallis (else)
    if CONTINUOUS in types and CATEGORICAL in types:
        if type_a == CONTINUOUS:
            cont, grp, cont_normal = a, b, a_normal
        else:
            cont, grp, cont_normal = b, a, b_normal
        groups = [cont[grp == g] for g in pd.unique(grp)]
        groups = [g for g in groups if len(g) >= min_group_size]
        if len(groups) < 2:
            return ("insufficient groups", np.nan, np.nan)
        if cont_normal:
            stat, p = stats.f_oneway(*groups)
            return ("ANOVA", float(stat), float(p))
        stat, p = stats.kruskal(*groups)
        return ("Kruskal-Wallis", float(stat), float(p))

    # 7 — (di)categorical vs (di)categorical -> Chi-Square
    contingency = pd.crosstab(a, b)
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return ("Chi-Square", np.nan, np.nan)
    stat, p, _, _ = stats.chi2_contingency(contingency)
    return ("Chi-Square", float(stat), float(p))


# ---------------------------------------------------------------- #
# Function 1 — classify the missingness of each variable #
# ---------------------------------------------------------------- #
def classify_missingness(
    df: pd.DataFrame,
    continuous_vars: Sequence[str],
    categorical_vars: Sequence[str],
    dichotomous_vars: Sequence[str],
    normal_vars: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    target_vars: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Test, for every variable with missing data, whether it is MAR or MCAR.

    For each focal variable a dichotomous missing-value indicator is built and
    tested against every *other* supplied variable, using :func:`run_test_by_type`
    so the correct one of the seven tests is chosen per variable type.

    Parameters
    ----------
    df :
        The data.
    continuous_vars, categorical_vars, dichotomous_vars :
        The variables to consider, split by type.  These are the variables
        passed *in* — nothing is auto-detected from the frame.
    normal_vars :
        Continuous variables that are normally distributed (drives the
        parametric / non-parametric choice).  If ``None`` it is inferred with
        Shapiro-Wilk via :func:`infer_normal_vars`.
    alpha :
        Significance threshold (default 0.05).
    target_vars :
        Restrict the analysis to these focal variables.  Defaults to every
        supplied variable that actually has at least one missing value.

    Returns
    -------
    dict with three DataFrames (rows = focal variables, columns = predictors):

    * ``"p_values"`` : raw p-value of each association test.
    * ``"significant"`` : boolean matrix, True where p <= alpha (MAR signal).
    * ``"tests"`` : name of the test used for each cell.
    """
    type_map = _build_type_map(continuous_vars, categorical_vars, dichotomous_vars)
    all_vars = list(type_map.keys())

    if normal_vars is None:
        normal_vars = infer_normal_vars(df, continuous_vars, alpha=alpha)
    normal_set = set(normal_vars)

    # Focal variables: those (among the supplied ones) that have missing values.
    if target_vars is None:
        focal_vars = [v for v in all_vars if df[v].isnull().any()]
    else:
        focal_vars = list(target_vars)

    p_values: Dict[str, Dict[str, float]] = {}
    test_names: Dict[str, Dict[str, str]] = {}

    for focal in focal_vars:
        p_values[focal] = {}
        test_names[focal] = {}

        # The missing-value indicator is, by construction, dichotomous.
        indicator = df[focal].isnull().astype(int)

        for other in all_vars:
            if other == focal:
                continue

            test_name, _, p_val = run_test_by_type(
                series_a=indicator,
                series_b=df[other],
                type_a=DICHOTOMOUS,
                type_b=type_map[other],
                a_normal=False,
                b_normal=(other in normal_set),
            )
            p_values[focal][other] = p_val
            test_names[focal][other] = test_name

    pval_df = pd.DataFrame(p_values).T.reindex(index=focal_vars)
    test_df = pd.DataFrame(test_names).T.reindex(index=focal_vars)
    sig_df = pval_df.le(alpha)          # NaN -> False (test could not run)

    return {"p_values": pval_df, "significant": sig_df, "tests": test_df}


# ------------------------------------------ #
# Function 2 — the summary table 
# ------------------------------------------ #
def missingness_mechanism_table(
    df: pd.DataFrame,
    continuous_vars: Sequence[str],
    categorical_vars: Sequence[str],
    dichotomous_vars: Sequence[str],
    normal_vars: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    target_vars: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build the MCAR / MAR summary table.

    Mirrors the table in the uploaded image (``num of affecting columns`` +
    mechanism flag) and additionally exposes **both** an ``MCAR`` and a ``MAR``
    column, as requested:

    * ``num of affecting columns`` — how many other variables are significantly
      associated with the missingness of the focal variable.
    * ``MAR``  — True when at least one variable is associated (the missingness
      depends on observed data).
    * ``MCAR`` — True otherwise (anything that is not MAR is treated as MCAR).

    The arguments are identical to :func:`classify_missingness`; this function
    simply aggregates its boolean matrix into one row per focal variable.
    """
    result = classify_missingness(
        df=df,
        continuous_vars=continuous_vars,
        categorical_vars=categorical_vars,
        dichotomous_vars=dichotomous_vars,
        normal_vars=normal_vars,
        alpha=alpha,
        target_vars=target_vars,
    )
    sig_df = result["significant"]

    table = pd.DataFrame(index=sig_df.index)
    table["num of affecting columns"] = sig_df.sum(axis=1).astype(int)
    table["MAR"] = table["num of affecting columns"] > 0
    table["MCAR"] = ~table["MAR"]
    return table
