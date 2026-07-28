import warnings
import numpy as np
import config as c
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler
import statsmodels.discrete.discrete_model as sm_discrete
from typing import Any, Callable, Dict, List, Optional, Tuple
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegressionCV

# functions for predictive modeling and internal validation

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe_logit(p: np.ndarray) -> np.ndarray:
    """
    Computes the logit transformation while avoiding numerical instability.

    The function clips probability values to a small interval near (0, 1) before
    applying the logit transformation. This prevents division-by-zero errors and
    infinite values when probabilities are exactly 0 or 1.

    The logit transformation is defined as:

        logit(p) = log(p / (1 - p))

    :param p: Array of probability values.
    :type p: numpy.ndarray

    :return: Array containing the logit-transformed values.
    :rtype: numpy.ndarray
    """
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def _hosmer_lemeshow(y_true: np.ndarray, y_prob: np.ndarray, groups: int = 10) -> Tuple[float, float]:
    """
    Calculates the Hosmer-Lemeshow goodness-of-fit test for a binary
    classification model.

    The Hosmer-Lemeshow test evaluates how well predicted probabilities agree
    with the observed binary outcomes. Predicted probabilities are divided into
    approximately equal-sized groups (deciles by default), and observed and
    expected event counts are compared using a chi-squared statistic.

    A large p-value indicates no evidence of poor model calibration, whereas a
    small p-value suggests that the predicted probabilities do not adequately
    fit the observed data.

    :param y_true: Array of observed binary outcomes.
    :param y_prob: Array of predicted probabilities for the positive class.
    :param groups: Number of probability groups used in the test.
    :type y_true: numpy.ndarray
    :type y_prob: numpy.ndarray
    :type groups: int

    :return:
        Tuple containing:
        - Hosmer-Lemeshow chi-squared statistic.
        - Corresponding p-value.
    :rtype: Tuple[float, float]
    """
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df["decile"] = pd.qcut(df["p"], q=groups, duplicates="drop")
    grouped = df.groupby("decile", observed=True).agg(
        observed=("y", "sum"),
        expected=("p", "sum"),
        n=("y", "count"),
    )
    denominator = grouped["expected"] * (1 - grouped["expected"] / grouped["n"])
    denominator = denominator.replace(0, np.nan)
    hl_stat = float(((grouped["observed"] - grouped["expected"]) ** 2 / denominator).sum())
    dof = max(len(grouped) - 2, 1)
    p_value = float(1 - stats.chi2.cdf(hl_stat, dof))
    return hl_stat, p_value


def _bootstrap_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Calculates a compact set of binary classification performance metrics.

    The function is intended for repeated evaluation inside a bootstrap procedure.
    It computes a minimal set of performance measures that summarize model
    discrimination, calibration, and overall prediction accuracy.

    Computed metrics:
    - Area Under the ROC Curve (AUC).
    - Brier score.
    - Calibration slope estimated by fitting a logistic regression model between
    the observed outcomes and the logit-transformed predicted probabilities.

    If the response variable contains only a single class, the function returns
    None because the performance metrics cannot be computed.

    :param y_true: Array of observed binary outcomes.
    :param y_prob: Array of predicted probabilities for the positive class.
    :type y_true: numpy.ndarray
    :type y_prob: numpy.ndarray

    :return:
        Dictionary containing the calculated metrics, or None if the metrics
        cannot be computed.
    :rtype: Optional[Dict[str, float]]
    """
    if len(np.unique(y_true)) < 2:
        return None
    auc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    logit_p = _safe_logit(y_prob)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal = sm.Logit(y_true, sm.add_constant(logit_p)).fit(disp=False, maxiter=100)
        cal_slope = float(np.asarray(cal.params).ravel()[1])
    except Exception:
        cal_slope = float("nan")
    return {"auc": auc, "brier": brier, "calibration_slope": cal_slope}


# ---------------------------------------------------------------------------
# Stage 1a — Missing-value handling (existing, extended type hint)
# ---------------------------------------------------------------------------

def apply_complete_case_analysis(df: pd.DataFrame, required_cols: List[str] = c.MAIN_MODEL_VARS) -> Tuple[pd.DataFrame, Tuple[int, float]]:
    """
    Performs complete case analysis by removing rows with missing values.

    The function filters the input DataFrame and retains only rows that contain
    non-missing values across all variables specified in 'required_cols'.
    This approach is commonly used when a strict no-imputation strategy is
    required.

    The function also reports how many rows were removed and what proportion of
    the dataset was affected.

    Note:
    This aligns with a complete-case (listwise deletion) assumption where missing
    values are not imputed but instead excluded from the analysis.

    :param df: Input pandas DataFrame.
    :param required_cols: List of columns that must contain non-missing values
                        for a row to be retained.
    :type df: pandas.DataFrame
    :type required_cols: List[str]

    :return:
        Tuple containing:
        - DataFrame after complete case filtering.
        - Tuple of (number of dropped rows, percentage of dropped rows).
    :rtype: Tuple[pandas.DataFrame, Tuple[int, float]]
    """
    original_n = len(df)
    df_complete = df.dropna(subset=required_cols).copy()
    dropped_n = original_n - len(df_complete)
    return df_complete, (dropped_n, round((dropped_n / original_n) * 100, 2))


# ---------------------------------------------------------------------------
# Stage 1b — Cohort splitting
# ---------------------------------------------------------------------------

# run and test df['was_planned_cs'].isna().sum() before run this function
def split_cohort(df: pd.DataFrame,filter_col: str = "was_planned_cs",filter_val: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a dataset into two analytical cohorts based on a filtering condition.

    The function creates two separate DataFrames:
    1. Full cohort: includes all observations in the dataset.
    2. Sub-cohort: includes only rows that satisfy a specific condition
    defined by 'filter_col == filter_val'.

    This is commonly used in clinical or cohort-based analysis pipelines where
    different modeling tasks are performed on the full population versus a
    restricted subgroup.

    :param df: Input pandas DataFrame containing the full dataset.
    :param filter_col: Column used to define the filtering condition.
    :param filter_val: Value used to select the subset cohort.
    :type df: pandas.DataFrame
    :type filter_col: str
    :type filter_val: int (default = 0)

    :return:
        Tuple containing:
        - Full cohort DataFrame (unfiltered copy).
        - Filtered cohort DataFrame where filter condition is satisfied.
    :rtype: Tuple[pandas.DataFrame, pandas.DataFrame]
    """
    full_cohort = df.copy()
    labor_cohort = df[df[filter_col] == filter_val].copy()
    return full_cohort, labor_cohort


# ---------------------------------------------------------------------------
# Stage 2 — Events-Per-Variable adequacy gate
# ---------------------------------------------------------------------------

def check_epv(df: pd.DataFrame, features: List[str],
               target: str, min_epv: int = 10) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Evaluates whether a dataset satisfies the Events Per Variable (EPV) criterion for logistic regression.

    This function assesses the adequacy of the available outcome events relative to
    the number of candidate predictors. The EPV rule is widely used to estimate
    whether a regression model is likely to have sufficient statistical support and
    to reduce the risk of overfitting.

    The function calculates the total number of outcome events, the maximum
    recommended number of predictors based on the specified minimum EPV threshold,
    and the actual EPV ratio. It also separates the candidate features into
    admissible and excluded groups according to the allowable number of predictors.

    The results are returned both as a dictionary for programmatic use and as a
    DataFrame for convenient reporting or visualization.

    :param df: Input pandas DataFrame containing features and the target variable.
    :param features: List of candidate predictor column names.
    :param target: Name of the binary target column.
    :param min_epv: Minimum required number of events per predictor variable
                    (default is 10).

    :type df: pandas.DataFrame
    :type features: List[str]
    :type target: str
    :type min_epv: int

    :return:
        Tuple containing:
        - res_dict: Dictionary summarizing the EPV evaluation, including the number
        of events, number of features, actual EPV, maximum allowable predictors,
        pass/fail status, admissible features, and excluded features.
        - res_df: DataFrame representation of the EPV evaluation results.

    :rtype: Tuple[Dict[str, Any], pandas.DataFrame]
    """
    n_events = int(df[target].sum())
    max_variables = n_events // min_epv
    actual_epv = n_events / len(features) if features else 0.0

    admissible = list(features[:max_variables])
    excluded = list(features[max_variables:])

    res_dict = {
        "n_events": n_events,
        "n_features": len(features),
        "actual_epv": round(actual_epv, 1),
        "max_variables": max_variables,
        "passes": len(features) <= max_variables,
        "admissible": admissible,
        "excluded": excluded,
    } 

    res_df = [
        {"Metric": "Outcome events", "Key": "n_events", "Value": res_dict['n_events']},
        {"Metric": "Candidate features", "Key": "n_features", "Value": res_dict['n_features']},
        {"Metric": "Actual EPV", "Key": "actual_epv", "Value": f"{res_dict['actual_epv']:.1f}"},
        {"Metric": "Max allowable vars", "Key": "max_variables", "Value": res_dict['max_variables']},
        {"Metric": "EPV check passes", "Key": "passes", "Value": res_dict['passes']}, 
        {"Metric": "Admissible", "Key": "admissible", "Value": ", ".join(res_dict['admissible']) if res_dict['admissible'] else "None"},
        {"Metric": "Excluded", "Key": "excluded", "Value": ", ".join(res_dict['excluded']) if res_dict['excluded'] else "None"}
    ]
    res_df = pd.DataFrame(res_df)
    
    return res_dict, res_df


# ---------------------------------------------------------------------------
# Stage 3a — LASSO feature selection
# ---------------------------------------------------------------------------

def lasso_feature_selection(df: pd.DataFrame, features: List[str], target: str, cv: int = 5,random_state: int = 42) -> Dict[str, Any]:
    """
    Performs feature selection using L1-regularized logistic regression
    (LASSO-style selection).

    The function standardizes the input features and fits a cross-validated
    logistic regression model with L1 penalty. Features with non-zero
    coefficients are considered selected, while those with zero coefficients
    are eliminated.

    This approach is commonly used for automatic feature selection in
    classification problems, as L1 regularization induces sparsity in the model
    coefficients.

    Steps:
    1. Extract feature matrix X and target vector y.
    2. Standardize features using StandardScaler.
    3. Fit LogisticRegressionCV with L1 penalty and cross-validation.
    4. Extract learned coefficients.
    5. Split features into selected and eliminated based on coefficient values.

    :param df: Input pandas DataFrame containing features and target.
    :param features: List of feature column names used for training.
    :param target: Name of the target column.
    :param cv: Number of cross-validation folds.
    :param random_state: Random seed for reproducibility.
    :type df: pandas.DataFrame
    :type features: List[str]
    :type target: str
    :type cv: int
    :type random_state: int

    :return:
        Dictionary containing:
        - selected: List of selected features (non-zero coefficients).
        - eliminated: List of eliminated features (zero coefficients).
        - coef: Dictionary mapping feature → coefficient.
        - best_C: Best inverse regularization strength chosen by CV.
    :rtype: Dict[str, Any]
    """
    X = df[features].values
    y = df[target].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegressionCV(
        penalty="l1",
        solver="liblinear",
        cv=cv,
        max_iter=10_000,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_scaled, y)

    coef = dict(zip(features, model.coef_[0].tolist()))
    selected = [f for f, c_ in coef.items() if c_ != 0.0]
    eliminated = [f for f, c_ in coef.items() if c_ == 0.0]

    return {
        "selected": selected,
        "eliminated": eliminated,
        "coef": coef,
        "best_C": float(model.C_[0]),
    }


# ---------------------------------------------------------------------------
# Stage 3b — Backward elimination
# ---------------------------------------------------------------------------

def backward_elimination(df: pd.DataFrame, features: List[str], target: str, p_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Performs backward feature elimination using Wald p-values from a logistic regression model.

    This function implements a stepwise backward elimination procedure where all
    candidate features are initially included in a logistic regression model. At each
    iteration, the model is refit and the feature with the highest p-value (least
    statistically significant) is identified. If this p-value exceeds the specified
    significance threshold (`p_threshold`), the corresponding feature is removed.

    The process repeats until all remaining features are statistically significant
    (p-values below or equal to the threshold) or until no features remain.

    This method is commonly used in statistical modeling to reduce feature space
    while maintaining only variables that contribute significantly to the predictive
    power of the model.

    :param df: Input pandas DataFrame containing feature columns and target variable.
    :param features: List of feature column names to evaluate.
    :param target: Name of the binary target column used for logistic regression.
    :param p_threshold: Significance threshold for feature removal (default is 0.05).

    :type df: pandas.DataFrame
    :type features: List[str]
    :type target: str
    :type p_threshold: float

    :return:
        Dictionary containing:
        - selected: List of features that remain after elimination.
        - eliminated: List of removed features in the order they were eliminated.
        - steps: Audit trail of elimination steps, each containing:
            * step: Iteration index
            * removed_var: Feature removed at that step
            * p_value: Corresponding p-value of the removed feature

    :rtype: Dict[str, Any]
    """
    remaining = list(features)
    eliminated = []
    steps = []

    step = 0
    while remaining:
        X = sm.add_constant(df[remaining], has_constant="add")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.Logit(df[target], X).fit(disp=False, maxiter=200)

        pvals = model.pvalues.drop("const", errors="ignore")
        max_p = float(pvals.max())

        if max_p <= p_threshold:
            break

        worst_var = str(pvals.idxmax())
        steps.append({"step": step, "removed_var": worst_var, "p_value": round(max_p, 6)})
        remaining.remove(worst_var)
        eliminated.append(worst_var)
        step += 1

    return {
        "selected": remaining,
        "eliminated": eliminated,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Stage 4a — Model training
# ---------------------------------------------------------------------------

def summarize_categorical_by_target(df: pd.DataFrame, cat_features: List[str], target: str) -> pd.DataFrame:
    summary_list = []
    
    for feature in cat_features:
        crosstab = pd.crosstab(df[feature], df[target]).reset_index()
        
        crosstab.rename(columns={feature: 'Category_Value'}, inplace=True)
        
        crosstab.insert(0, 'Feature', feature)
        
        summary_list.append(crosstab)
        
    final_df = pd.concat(summary_list, ignore_index=True)
    
    final_df.columns.name = None 
    
    return final_df


def train_logistic_regression(df: pd.DataFrame, features: List[str], target: str = c.TARGET_VAR) -> sm_discrete.LogitResults:
    """
    Fits a standard (unweighted) logistic regression model using maximum likelihood estimation.

    The function trains a logistic regression model on the provided dataset using
    statsmodels' Logit implementation. It includes an intercept term and uses all
    supplied features without applying class weighting or resampling. This ensures
    that the model preserves the natural distribution of the outcome variable,
    which is important for calibration when the target reflects real-world prevalence.

    Before training, the function validates that the target variable is binary
    (contains only 0 and 1 values). If invalid values are detected, it raises an error.

    The fitted model retains full statistical output, including coefficients,
    p-values, confidence intervals, and log-likelihood, making it suitable for
    interpretability and statistical inference.

    :param df: Input pandas DataFrame containing features and target variable.
    :param features: List of column names to be used as predictors.
    :param target: Name of the binary target column (must contain only 0/1 values).

    :type df: pandas.DataFrame
    :type features: List[str]
    :type target: str

    :return: Fitted statsmodels logistic regression model (LogitResults object).
    :rtype: sm_discrete.LogitResults
    """
    unique_vals = set(df[target].dropna().unique())
    if not unique_vals.issubset({0, 1}):
        raise ValueError(
            f"target column '{target}' must contain only 0 and 1; "
            f"found: {sorted(unique_vals)}"
        )

    X = sm.add_constant(df[features], has_constant="add")
    y = df[target]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.Logit(y, X).fit(disp=False, maxiter=200)
    return model


# ---------------------------------------------------------------------------
# Stage 4b — Odds-ratio table
# ---------------------------------------------------------------------------

def compute_odds_ratios(model: sm_discrete.LogitResults, conf_level: float = 0.95) -> pd.DataFrame:
    """
    Computes Odds Ratios (OR) with confidence intervals and Wald p-values from a fitted logistic regression model.

    The function converts the logistic regression coefficients into odds ratios by
    applying the exponential transformation. It also derives confidence intervals
    by exponentiating the model's confidence bounds and includes the corresponding
    Wald p-values for statistical significance assessment.

    The intercept term is removed, and only predictor variables are returned to
    focus on interpretable clinical or explanatory effects of features.

    This output format is commonly used in medical and epidemiological analysis
    to express the magnitude and direction of associations in an interpretable way.

    :param model: Fitted statsmodels logistic regression results object.
    :param conf_level: Confidence level for interval estimation (default is 0.95).

    :type model: sm_discrete.LogitResults
    :type conf_level: float

    :return:
        pandas DataFrame indexed by feature names containing:
        - OR: Odds ratio (exp(coefficient))
        - CI_lower: Lower bound of confidence interval
        - CI_upper: Upper bound of confidence interval
        - p_value: Wald p-value for each coefficient

    :rtype: pandas.DataFrame
    """
    alpha = 1 - conf_level
    conf_int = model.conf_int(alpha=alpha)

    result = pd.DataFrame(
        {
            "OR": np.exp(model.params),
            "CI_lower": np.exp(conf_int.iloc[:, 0]),
            "CI_upper": np.exp(conf_int.iloc[:, 1]),
            "p_value": model.pvalues,
        }
    )
    result = result.drop(index="const", errors="ignore")
    result = result.round({"OR": 3, "CI_lower": 3, "CI_upper": 3, "p_value": 4})
    return result


# ---------------------------------------------------------------------------
# Stage 5a — Bootstrap validation loop
# ---------------------------------------------------------------------------

def bootstrap_validate(df: pd.DataFrame, features: List[str],
                       target: str = c.TARGET_VAR,n_iterations: int = 1000,
                         random_state: Optional[int] = None, print_cause:bool=False) -> Tuple[Dict[str, float], List[Dict[str, float]]]:

    rng = np.random.RandomState(random_state)

    X_with_const = sm.add_constant(df[features].values, has_constant="add")
    y = df[target].values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        apparent_model = sm.Logit(y, X_with_const).fit(disp=False, maxiter=200)
    
    apparent_prob = apparent_model.predict(X_with_const)
    apparent_metrics = _bootstrap_metrics(y, apparent_prob)

    bootstrap_optimisms: List[Dict[str, float]] = []
    failed_iterations = 0
    failure_reasons = {
        "missing_class_1": 0, 
        "missing_class_0": 0, 
        "model_convergence_error": 0,
        "metrics_calculation_error": 0
    }
    for _ in range(n_iterations):
        indices = rng.choice(len(X_with_const), size=len(X_with_const), replace=True)
        X_boot, y_boot = X_with_const[indices], y[indices]

        unique_classes = np.unique(y_boot)
        if len(unique_classes) < 2:
            if unique_classes[0] == 0:
                failure_reasons["missing_class_1"] += 1
            else:
                failure_reasons["missing_class_0"] += 1
            failed_iterations += 1
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boot_model = sm.Logit(y_boot, X_boot).fit(disp=False, maxiter=200)
        except Exception as e:
            if print_cause:
                print(f"Convergence Error Preview: {str(e)}")
            failure_reasons["model_convergence_error"] += 1
            failed_iterations += 1
            continue

        prob_boot = boot_model.predict(X_boot)
        prob_orig = boot_model.predict(X_with_const)

        m_boot = _bootstrap_metrics(y_boot, prob_boot)
        m_orig = _bootstrap_metrics(y, prob_orig)

        if m_boot is None or m_orig is None:
            failure_reasons["metrics_calculation_error"] += 1
            failed_iterations += 1
            continue

        optimism = {k: m_boot[k] - m_orig[k] for k in m_boot}
        bootstrap_optimisms.append(optimism)

    failed_percentage = (failed_iterations / n_iterations) * 100 if n_iterations > 0 else 0.0
    failure_stats = (failed_iterations, failed_percentage, failure_reasons)

    return apparent_metrics, bootstrap_optimisms, failure_stats


# ---------------------------------------------------------------------------
# Stage 5b — Optimism correction
# ---------------------------------------------------------------------------

def compute_optimism_correction(apparent_metrics: Dict[str, float], bootstrap_optimisms: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Computes optimism-corrected performance metrics using bootstrap validation results.

    This function aggregates per-iteration bootstrap optimism estimates and computes
    their mean for each performance metric. The mean optimism is then subtracted
    from the apparent (training-set) performance to obtain corrected estimates that
    better reflect expected out-of-sample performance.

    This correction addresses model overfitting by adjusting inflated training
    performance metrics downward (or upward, depending on the metric direction),
    following standard bootstrap optimism correction methodology used in predictive
    model validation.

    The function returns the original apparent metrics, the computed mean optimism,
    and the final corrected metrics.

    :param apparent_metrics: Dictionary of performance metrics computed on the full dataset.
    :param bootstrap_optimisms: List of dictionaries containing per-iteration optimism values.

    :type apparent_metrics: Dict[str, float]
    :type bootstrap_optimisms: List[Dict[str, float]]

    :return:
        Dictionary containing:
        - apparent: Original performance metrics from training data.
        - mean_optimism: Average optimism per metric across bootstrap iterations.
        - corrected: Optimism-corrected performance metrics.

    :rtype: Dict[str, Dict[str, float]]
    """
    keys = list(apparent_metrics.keys())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_optimism = {
            k: float(np.nanmean([m[k] for m in bootstrap_optimisms if k in m]))
            for k in keys
        }
    corrected = {k: apparent_metrics[k] - mean_optimism[k] for k in keys}

    return {
        "apparent": apparent_metrics,
        "mean_optimism": {k: round(v, 4) for k, v in mean_optimism.items()},
        "corrected": {k: round(v, 4) for k, v in corrected.items()},
    }


# ---------------------------------------------------------------------------
# Stage 6a — Full model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Evaluates a probabilistic binary classification model using both discrimination
    and calibration metrics.

    The function computes standard performance measures to assess how well the
    predicted probabilities align with observed outcomes. Discrimination is
    measured using AUROC and Brier score, while calibration is evaluated using the
    Hosmer–Lemeshow test and a calibration regression (slope and intercept of
    log-odds of predicted probabilities vs observed outcomes).

    These metrics together provide a complete view of model quality:
    discrimination reflects ranking ability, while calibration reflects the
    accuracy of predicted probabilities.

    :param y_true: Ground-truth binary labels (0/1).
    :param y_prob: Predicted probabilities for the positive class.

    :type y_true: np.ndarray
    :type y_prob: np.ndarray

    :return:
        Dictionary containing:
        - auc: Area Under the ROC Curve (higher = better discrimination).
        - brier: Brier score (lower = better probabilistic accuracy).
        - hl_stat: Hosmer-Lemeshow chi-squared statistic for calibration.
        - hl_p: Hosmer-Lemeshow p-value (higher suggests better calibration).
        - calibration_slope: Slope of calibration regression (ideal = 1.0).
        - calibration_intercept: Intercept of calibration regression (ideal = 0.0).

    :rtype: Dict[str, float]
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    auc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    hl_stat, hl_p = _hosmer_lemeshow(y_true, y_prob)

    logit_p = _safe_logit(y_prob)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal = sm.Logit(y_true, sm.add_constant(logit_p, has_constant="add")).fit(
            disp=False, maxiter=200
        )
    _cal_params = np.asarray(cal.params).ravel()
    cal_intercept = float(_cal_params[0])
    cal_slope = float(_cal_params[1])

    return {
        "auc": round(auc, 4),
        "brier": round(brier, 4),
        "hl_stat": round(hl_stat, 4),
        "hl_p": round(hl_p, 4),
        "calibration_slope": round(cal_slope, 4),
        "calibration_intercept": round(cal_intercept, 4),
    }


# ---------------------------------------------------------------------------
# Stage 6b — Decision Curve Analysis
# ---------------------------------------------------------------------------

def decision_curve_analysis(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Performs Decision Curve Analysis (DCA) to evaluate the clinical utility of a predictive model across decision thresholds.

    This function computes the net benefit of a model over a range of probability thresholds and compares it against two default clinical strategies: treating all patients and treating none. Net benefit is calculated by balancing true positives against false positives using a threshold-dependent weighting derived from the odds at each threshold.

    The analysis helps determine whether using the model improves decision-making compared to standard reference strategies, and identifies the lowest threshold at which the model provides superior net benefit.

    If no threshold exists where the model outperforms both reference strategies, no optimal threshold is returned.

    :param y_true: Ground-truth binary labels (0/1).
    :param y_prob: Predicted probabilities for the positive class.
    :param thresholds: Optional array of probability thresholds to evaluate.
                    If not provided, defaults to 0.01-0.50 with step 0.01.

    :type y_true: np.ndarray
    :type y_prob: np.ndarray
    :type thresholds: Optional[np.ndarray]

    :return:
        Tuple containing:
        - dca_df: DataFrame with columns:
            * threshold: Decision threshold
            * net_benefit_model: Net benefit of the model
            * net_benefit_all: Net benefit of treating all patients
            * net_benefit_none: Net benefit of treating none (0 baseline)
        - optimal_threshold: Lowest threshold where the model outperforms both
        reference strategies, or None if no such threshold exists.

    :rtype: Tuple[pd.DataFrame, Optional[float]]
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    prevalence = float(y_true.mean())

    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 0.51, 0.01), 2)

    rows = []
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        odds = pt / (1 - pt)

        nb_model = tp / n - fp / n * odds
        nb_all = prevalence - (1 - prevalence) * odds
        nb_none = 0.0

        rows.append(
            {
                "threshold": pt,
                "net_benefit_model": round(nb_model, 6),
                "net_benefit_all": round(nb_all, 6),
                "net_benefit_none": nb_none,
            }
        )

    dca_df = pd.DataFrame(rows)

    reference_max = dca_df[["net_benefit_all", "net_benefit_none"]].max(axis=1)
    dominant = dca_df[dca_df["net_benefit_model"] > reference_max]
    optimal_threshold = float(dominant["threshold"].iloc[0]) if len(dominant) > 0 else None

    return dca_df, optimal_threshold


# ---------------------------------------------------------------------------
# Stage 7a — ROC curve plot
# ---------------------------------------------------------------------------

_PALETTE = {"model": "#1565C0", "reference": "#9E9E9E", "fill": "#BBDEFB"}
_SPINE_COLOR = "#BDBDBD"
_FONT = {"family": "DejaVu Sans"}


def _apply_academic_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """
    Applies a consistent academic visual style to a Matplotlib Axes object.

    This helper function standardizes the appearance of plots by configuring the
    title, axis labels, tick formatting, grid, axis spines, and background color.
    Using a common styling function ensures that all generated figures share a
    uniform, publication-quality appearance throughout the project.

    :param ax: Matplotlib Axes object to be styled.
    :param title: Title displayed at the top of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.

    :type ax: matplotlib.axes.Axes
    :type title: str
    :type xlabel: str
    :type ylabel: str

    :return: None.
    :rtype: None
    """
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12, **_FONT)
    ax.set_xlabel(xlabel, fontsize=11, **_FONT)
    ax.set_ylabel(ylabel, fontsize=11, **_FONT)
    ax.tick_params(labelsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, color=_SPINE_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE_COLOR)
        spine.set_linewidth(0.8)
    ax.set_facecolor("#FAFAFA")


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   title: str = "ROC Curve", show: bool = False,
                   save_plt: bool = True,
                   output_path: str = c.OUTPUT_DIR_FOR_GRAPH) -> None:
    """
    Plot the Receiver Operating Characteristic (ROC) curve with AUC annotation.

    The diagonal no-skill reference line is shown for comparison. The area
    under the curve is displayed in the legend.

    :param y_true: Ground-truth binary labels (0/1).
    :param y_prob: Predicted probabilities for the positive class.
    :param title: Figure title.
    :param show: If True, calls plt.show() to display the plot interactively.
    :param save_plt: If True, saves the figure to the specified output directory.
    :param output_path: Base directory path for saving the generated image.

    :type y_true: numpy.ndarray
    :type y_prob: numpy.ndarray
    :type title: str
    :type show: bool
    :type save_plt: bool
    :type output_path: str

    :return: None
    :rtype: None
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr, tpr,
        color=_PALETTE["model"], linewidth=2.0,
        label=f"Model (AUC = {auc:.3f})",
    )
    ax.fill_between(fpr, tpr, alpha=0.08, color=_PALETTE["fill"])
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2,
            color=_PALETTE["reference"], label="No skill (AUC = 0.500)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    _apply_academic_style(ax, title, "1 - Specificity (FPR)", "Sensitivity (TPR)")
    fig.tight_layout()

    if save_plt:
        fig.savefig(f"{output_path}/roc_curve.png", bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)



# ---------------------------------------------------------------------------
# Stage 7b — Calibration curve plot
# ---------------------------------------------------------------------------

def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, title: str = "Calibration Plot",
                           show: bool = False, save_plt: bool = True, output_path: str = c.OUTPUT_DIR_FOR_GRAPH, end:str = "") -> None:
    """
    Plot a calibration curve (reliability diagram) comparing predicted
    probabilities against observed outcome rates.

    Predicted probabilities are grouped into *n_bins* equal-width bins.
    The mean predicted probability and the observed event fraction are
    computed per bin and plotted.  The perfect-calibration diagonal is
    shown as a reference.  A rug histogram of predicted probabilities is
    displayed below the main panel.

    :param y_true: Ground-truth binary labels (0/1).
    :param y_prob: Predicted probabilities for the positive class.
    :param n_bins: Number of probability bins (default = 10).
    :param title: Figure title.
    :param show: If True, calls plt.show() to display the plot interactively.
    :param save_plt: If True, saves the figure to the specified output directory.
    :param output_path: Base directory path for saving the generated image.
    :param end: Add to the end of the file name

    :type y_true: numpy.ndarray
    :type y_prob: numpy.ndarray
    :type n_bins: int
    :type title: str
    :type show: bool
    :type save_plt: bool
    :type output_path: str
    :type end: str

    :return: None
    :rtype: NoneType
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges[1:-1])
    mean_pred, mean_obs, counts = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        mean_pred.append(y_prob[mask].mean())
        mean_obs.append(y_true[mask].mean())
        counts.append(mask.sum())

    mean_pred = np.array(mean_pred)
    mean_obs = np.array(mean_obs)
    counts = np.array(counts)

    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1, figsize=(6, 6),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08},
        sharex=True,
        constrained_layout=True,
    )

    ax_main.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2,
                 color=_PALETTE["reference"], label="Perfect calibration")
    sc = ax_main.scatter(
        mean_pred, mean_obs,
        s=counts / counts.max() * 200 + 20,
        color=_PALETTE["model"], edgecolors="white", linewidth=0.8,
        zorder=3, label="Observed fraction per bin",
    )
    ax_main.plot(mean_pred, mean_obs, color=_PALETTE["model"],
                 linewidth=1.4, alpha=0.6)
    ax_main.set_xlim([-0.01, 1.01])
    ax_main.set_ylim([-0.01, 1.01])
    ax_main.legend(fontsize=9, loc="upper left", framealpha=0.9)
    _apply_academic_style(
        ax_main, title,
        xlabel="",
        ylabel="Observed event fraction",
    )
    ax_main.set_xlabel("")

    ax_hist.hist(y_prob, bins=40, color=_PALETTE["model"], alpha=0.55,
                 edgecolor="white", linewidth=0.4)
    ax_hist.set_ylabel("Count", fontsize=9)
    ax_hist.set_xlabel("Predicted probability", fontsize=11, **_FONT)
    ax_hist.tick_params(labelsize=8)
    ax_hist.set_facecolor("#FAFAFA")
    for spine in ax_hist.spines.values():
        spine.set_edgecolor(_SPINE_COLOR)

    if save_plt:
        fig.savefig(f"{output_path}/calibration_curve{end}.png", bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Stage 7c — Decision curve plot
# ---------------------------------------------------------------------------

def plot_decision_curve(dca_df: pd.DataFrame, title: str = "Decision Curve Analysis",
                        optimal_threshold: Optional[float] = None,
                        show: bool = False, save_plt: bool = True,
                        output_path: str = c.OUTPUT_DIR_FOR_GRAPH) -> None:
    """
    Plot the Net Benefit curves from a Decision Curve Analysis.

    Takes the DataFrame returned by :func:`decision_curve_analysis` and
    displays three curves: the model's net benefit, the 'treat all' strategy,
    and the 'treat none' baseline (always zero).  The region where the model
    dominates both reference strategies is shaded.  If *optimal_threshold* is
    supplied, a vertical marker is drawn at that point.

    :param dca_df: DataFrame with columns threshold, net_benefit_model,
                   net_benefit_all, net_benefit_none — as returned by
                   :func:`decision_curve_analysis`.
    :param title: Figure title.
    :param optimal_threshold: Optional threshold to mark on the plot.
    :param show: If True, calls plt.show() to display the plot interactively.
    :param save_plt: If True, saves the figure to the specified output directory.
    :param output_path: Base directory path for saving the generated image.

    :type dca_df: pd.DataFrame
    :type title: str
    :type optimal_threshold: Optional[float]
    :type show: bool
    :type save_plt: bool
    :type output_path: str

    :return: None
    :rtype: NoneType

    Alignment: Section 8.2, for_claude.txt req 4.3 (Decision Curve Analysis
               for threshold selection).
    """
    th = dca_df["threshold"].values
    nb_model = dca_df["net_benefit_model"].values
    nb_all   = dca_df["net_benefit_all"].values
    nb_none  = np.zeros(len(th))

    reference_max = np.maximum(nb_all, nb_none)
    model_dominant = nb_model > reference_max

    fig, ax = plt.subplots(figsize=(7, 5))

    if model_dominant.any():
        ax.fill_between(
            th, nb_model, reference_max,
            where=model_dominant,
            alpha=0.12, color=_PALETTE["model"],
            label="_nolegend_",
        )

    ax.plot(th, nb_model, color=_PALETTE["model"],
            linewidth=2.2, label="Model")
    ax.plot(th, nb_all, color="#E53935", linewidth=1.6,
            linestyle="-.", label="Treat all")
    ax.plot(th, nb_none, color=_PALETTE["reference"],
            linewidth=1.2, linestyle=":", label="Treat none")

    if optimal_threshold is not None:
        ax.axvline(
            optimal_threshold, color=_PALETTE["model"],
            linestyle="--", linewidth=1.0, alpha=0.7,
        )
        ax.annotate(
            f"Optimal\n{optimal_threshold:.2f}",
            xy=(optimal_threshold, ax.get_ylim()[1] * 0.85),
            xytext=(optimal_threshold + 0.02, ax.get_ylim()[1] * 0.85),
            fontsize=8, color=_PALETTE["model"],
        )

    ax.axhline(0, color=_SPINE_COLOR, linewidth=0.8)
    ax.set_xlim([th.min() - 0.005, th.max() + 0.005])
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    _apply_academic_style(
        ax, title,
        xlabel="Threshold probability",
        ylabel="Net benefit",
    )

    if save_plt:
        fig.savefig(f"{output_path}/decision_curve.png", bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Stage 8 — Heuristic Shrinkage and Optimism Correction
# ---------------------------------------------------------------------------

def apply_model_shrinkage(model: sm_discrete.LogitResults, X_eval: pd.DataFrame, shrinkage_factor: float) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Applies uniform coefficient shrinkage to a fitted logistic regression model to reduce overfitting and optimism.

    This function adjusts the regression coefficients by multiplying them by a
    user-specified shrinkage factor, typically obtained from bootstrap-based
    calibration. The intercept is then re-estimated to preserve the original event
    prevalence, ensuring that the average predicted probability remains consistent
    with the observed outcome frequency.

    Using the adjusted coefficients and intercept, the function computes
    optimism-corrected predicted probabilities and constructs an updated Odds Ratio
    table. The table contains the shrunk Odds Ratios, shrinkage-adjusted confidence
    intervals, and the original Wald p-values for each predictor.

    :param model: Fitted statsmodels logistic regression results object.
    :param X_eval: Evaluation feature matrix including the intercept ("const") column.
    :param shrinkage_factor: Multiplicative factor applied uniformly to all
                            predictor coefficients.

    :type model: sm_discrete.LogitResults
    :type X_eval: pd.DataFrame
    :type shrinkage_factor: float

    :return:
        Tuple containing:
        - shrunk_probs: NumPy array of predicted probabilities after coefficient shrinkage.
        - shrunk_or_table: DataFrame containing the shrunk Odds Ratios,
        shrinkage-adjusted confidence intervals, and original p-values for each predictor.

    :rtype: Tuple[numpy.ndarray, pandas.DataFrame]
"""

    params = model.params.copy()
    features = params.drop('const')
    shrunk_features = features * shrinkage_factor

    X_no_const = X_eval.drop(columns=['const'])
    linear_predictor_shrunk = X_no_const @ shrunk_features

    target_prevalence = model.model.endog.mean()

    def loss_func(intercept):
        p = 1 / (1 + np.exp(-(intercept + linear_predictor_shrunk)))
        return (p.mean() - target_prevalence)**2

    res = minimize_scalar(loss_func)
    new_intercept = res.x

    final_logit = new_intercept + linear_predictor_shrunk
    shrunk_probs = 1 / (1 + np.exp(-final_logit))

    conf_int = model.conf_int(alpha=0.05).drop(index="const", errors="ignore")
    
    shrunk_or_table = pd.DataFrame({
        "OR_Shrunk": np.exp(shrunk_features),
        "CI_lower_Shrunk": np.exp(conf_int.iloc[:, 0] * shrinkage_factor),
        "CI_upper_Shrunk": np.exp(conf_int.iloc[:, 1] * shrinkage_factor),
        "p_value": model.pvalues.drop("const", errors="ignore")
    })
    
    shrunk_or_table = shrunk_or_table.round({"OR_Shrunk": 3, "CI_lower_Shrunk": 3, "CI_upper_Shrunk": 3, "p_value": 4})

    return shrunk_probs, shrunk_or_table, new_intercept


def table2(p_val_tolerance:float = 1e-4,norm_txt:bool = False, **kwargs) -> pd.DataFrame:
    """
    Generates a publication-ready Odds Ratio table by combining one or more logistic regression result tables.

    This function formats Odds Ratios and their 95% confidence intervals into a
    single academic-style column for each supplied model. It automatically detects
    the Odds Ratio, confidence interval, and p-value columns in each input table,
    allowing results from multiple models to be presented side by side.

    When the p-values are identical across all models (within the specified
    tolerance), they are reported in a single shared column. Otherwise, a separate
    p-value column is created for each model. Optionally, feature names can be
    normalized to improve readability by replacing underscores with spaces and
    capitalizing each entry.

    :param p_val_tolerance: Absolute tolerance used to determine whether p-values
                            from different models should be considered identical.
                            Defaults to 1e-4.
    :param norm_txt: If True, reformats the index labels by replacing underscores
                    with spaces and capitalizing the first letter. Defaults to
                    False.
    :param kwargs: Keyword arguments where each key is a model name (e.g.,
                "Apparent", "Corrected") and each value is a pandas DataFrame
                containing Odds Ratios, confidence intervals, and p-values.

    :type p_val_tolerance: float
    :type norm_txt: bool
    :type kwargs: pandas.DataFrame

    :return: Formatted pandas DataFrame containing Odds Ratios with 95%
            confidence intervals and appropriately displayed p-value column(s),
            ready for academic reporting.
    :rtype: pandas.DataFrame
    """
    tables = kwargs
    
    if not tables:
        raise ValueError("No tables were provided to generate the academic table.")

    formatted_dfs = []
    p_values_dict = {}

    for model_name, df in tables.items():
        
        cols = df.columns.str.lower()
        
        or_col = df.columns[cols.str.startswith('or')][0]
        ci_lower_col = df.columns[cols.str.contains('ci_lower')][0]
        ci_upper_col = df.columns[cols.str.contains('ci_upper')][0]
        p_col = df.columns[cols.str.contains('p_value')][0]

        # Format OR and 95% CI into a single academic string
        formatted_or_ci = df.apply(
            lambda row: f"{row[or_col]:.3f} ({row[ci_lower_col]:.3f} - {row[ci_upper_col]:.3f})", 
            axis=1
        )
        formatted_or_ci.name = f"{model_name} OR (95% CI)"
        formatted_dfs.append(formatted_or_ci)

        # Store p-values for comparison
        p_values_dict[model_name] = df[p_col]

    # Combine all formatted OR (CI) columns
    final_table = pd.concat(formatted_dfs, axis=1)

    # Check if all p-values are identical across models
    p_vals_identical = True
    first_p_vals = list(p_values_dict.values())[0]
    
    if len(tables) > 1:
        for p_vals in list(p_values_dict.values())[1:]:
            if not np.allclose(first_p_vals, p_vals, atol=p_val_tolerance):
                p_vals_identical = False
                break

    # Helper function to format p-values (e.g., 0.0000 -> <0.001)
    def format_p_val(x):
        return "<0.001" if x < 0.001 else f"{x:.4f}"

    # Append p-value column(s)
    if p_vals_identical or len(tables) == 1:
        final_table["P-value"] = first_p_vals.apply(format_p_val)
    else:
        for model_name, p_vals in p_values_dict.items():
            final_table[f"{model_name} P-value"] = p_vals.apply(format_p_val)

    # Clean up index names for academic presentation (replace underscores, capitalize) - optional
    if norm_txt:
        final_table.index = final_table.index.str.replace('_', ' ').str.capitalize()

    return final_table