# -*- coding: utf-8 -*-
"""Missingness mechanism classification (MCAR vs MAR).
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from scipy import stats

# Four variable types ------------------------------------------------------- #
NORMAL = "normal"
NON_NORMAL = "non_normal"
CATEGORICAL = "categorical"
DICHOTOMOUS = "dichotomous"
CONTINUOUS = {NORMAL, NON_NORMAL}


def _pair_test(a: pd.Series, b: pd.Series, type_a: str, type_b: str) -> float:
    """Return the p-value of the appropriate one of the seven tests.

    ``a`` and ``b`` must already be aligned and free of missing values (the
    single ">= 3 observations" check is done by the caller, once).
    """
    types = {type_a, type_b}

    # continuous vs continuous  ->  Pearson (both normal) / Spearman
    if type_a in CONTINUOUS and type_b in CONTINUOUS:
        if type_a == NORMAL and type_b == NORMAL:
            return stats.pearsonr(a, b)[1]
        return stats.spearmanr(a, b)[1]

    # continuous vs dichotomous  ->  t-test (normal) / Mann-Whitney
    if types & CONTINUOUS and DICHOTOMOUS in types:
        (cont, cont_type, grp) = (a, type_a, b) if type_a in CONTINUOUS else (b, type_b, a)
        groups = [cont[grp == g] for g in pd.unique(grp)]
        if cont_type == NORMAL:
            return stats.ttest_ind(*groups, equal_var=True)[1]
        return stats.mannwhitneyu(*groups, alternative="two-sided")[1]

    # continuous vs categorical  ->  ANOVA (normal) / Kruskal-Wallis
    if types & CONTINUOUS and CATEGORICAL in types:
        (cont, cont_type, grp) = (a, type_a, b) if type_a in CONTINUOUS else (b, type_b, a)
        groups = [cont[grp == g] for g in pd.unique(grp)]
        if cont_type == NORMAL:
            return stats.f_oneway(*groups)[1]
        return stats.kruskal(*groups)[1]

    # (di)categorical vs (di)categorical  ->  Chi-Square
    return stats.chi2_contingency(pd.crosstab(a, b))[1]


def missingness_mechanism_table(
    df: pd.DataFrame,
    normal_vars: Sequence[str],
    non_normal_vars: Sequence[str],
    categorical_vars: Sequence[str],
    dichotomous_vars: Sequence[str],
    alpha: float = 0.05,
    min_obs: int = 3,
) -> pd.DataFrame:
    """Classify the missingness of every variable as MCAR or MAR.

    Parameters
    ----------
    df :
        The data.
    normal_vars, non_normal_vars, categorical_vars, dichotomous_vars :
        The variables to analyse, grouped by their (already known) type.
    alpha :
        Significance threshold for an association (default 0.05).
    min_obs :
        Minimum paired observations required to run a test (default 3).

    Returns
    -------
    pd.DataFrame indexed by the variables that have missing values, with:

    * ``num of affecting columns`` — how many variables are associated with the
      missingness of the focal variable;
    * ``MAR``  — True when that count > 0;
    * ``MCAR`` — True otherwise (everything that is not MAR is MCAR).
    """
    type_map = {
        **{v: NORMAL for v in normal_vars},
        **{v: NON_NORMAL for v in non_normal_vars},
        **{v: CATEGORICAL for v in categorical_vars},
        **{v: DICHOTOMOUS for v in dichotomous_vars},
    }
    variables = list(type_map)

    counts = {}
    for focal in variables:
        if not df[focal].isnull().any():
            continue
        indicator = df[focal].isnull().astype(int)  # the indicator is dichotomous

        n_associated = 0
        for other in variables:
            if other == focal:
                continue
            pair = pd.concat([indicator, df[other]], axis=1).dropna()
            if len(pair) < min_obs:                 # the only observation-count check
                continue
            p = _pair_test(pair.iloc[:, 0], pair.iloc[:, 1], DICHOTOMOUS, type_map[other])
            if pd.notna(p) and p <= alpha:
                n_associated += 1
        counts[focal] = n_associated

    table = pd.DataFrame({"num of affecting columns": pd.Series(counts, dtype=int)})
    table["MAR"] = table["num of affecting columns"] > 0
    table["MCAR"] = ~table["MAR"]
    return table
