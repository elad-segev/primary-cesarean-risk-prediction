# Prediction of Primary Cesarean Delivery Following Admission in Labor

A clinical risk model for multiparous women with diabetes in pregnancy and no history of cesarean section, using only data available at admission in labor.

Final undergraduate project  B.Sc. Digital Medical Technologies

## Overview

Cesarean delivery is one of the most common procedures in modern obstetrics and represents an important clinical decision point because of its implications for current and future pregnancies. Most prediction studies have focused on women with a previous cesarean delivery or first-time mothers, while the risk of primary cesarean delivery among multiparous women without a prior uterine scar remains less well characterized.

This project develops and internally validates a clinical prediction model for primary cesarean delivery among multiparous women with diabetes in pregnancy and no history of previous cesarean section (Robson groups 3 and 4). The model was designed as a decision-support tool using only demographic, obstetric, and clinical variables available at admission in labor, allowing individualized risk estimation before labor progression.

The final model is based on logistic regression fitted using maximum likelihood estimation and includes seven admission-time predictors identified through a structured model-development process. The model was evaluated using Harrell’s bootstrap optimism correction, calibration assessment, and Decision Curve Analysis to assess both predictive performance and potential clinical utility.

The development cohort included 3,690 deliveries from 2,897 women, with 294 primary cesarean deliveries observed (8.0% event rate). The final model demonstrated good discrimination (optimism-corrected AUC = 0.788) and strong calibration, providing an interpretable risk estimate based on multiple independent clinical predictors rather than a single dominant factor.

## Data confidentiality

This repository contains **code only**. No patient-level data, derived datasets, or analytical outputs are included, in line with clinical confidentiality requirements — `data/` and `output/` are gitignored, and all notebooks have been intentionally cleared of their cell outputs. The notebooks are therefore published as a record of methodology and are not executable as-is. Reproduction requires an approved dataset matching the schema defined in `config.py`.

## Pipeline

```
EDA and schema enforcement → Clinical logic validation → Missingness analysis (MCAR/MAR)
→ Complete-case analysis → Feature selection (LASSO / backward elimination)
→ Logistic regression → Bootstrap optimism correction → Shrinkage → DCA
```

## Repository structure

```
.
├── docs/                  # Methodology documentation
├── EDA.ipynb              # Exploratory analysis, clinical validation, Table 1
├── Model.ipynb            # Modeling, internal validation, clinical utility
├── config.py              # Paths, data schema, variable groups, thresholds
├── eda_utils.py           # EDA function library
├── model_utils.py         # Modeling & validation function library
├── auto_mod.py            # Automated multi-experiment runner
├── eda_pipline_test.py    # Ad-hoc sanity checks (not part of the pipeline)
├── installations.py       # Environment setup and dependency verification
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

Or run `python installations.py`, which additionally scaffolds the `data/` and `output/` directory tree and verifies that every required package is importable.

## Quick start

1. Place the source extraction in `data/` and set `DATA_PATH` in `config.py`
2. Run `EDA.ipynb` → produces `data/df_for_model.csv`
3. Run `Model.ipynb` → primary model, internal validation, clinical utility
4. Optional: `python auto_mod.py` → executes the experimental grid

## Main methodological choices

- **No imputation.** Missingness mechanism is diagnosed empirically (MCAR/MAR) before complete-case analysis; contradictory records are invalidated rather than guessed at.
- **No resampling or class weighting.** Native outcome prevalence is preserved so predicted probabilities remain calibrated to real-world risk.
- **Logistic regression by MLE**, for interpretability and direct translation of coefficients into Odds Ratios.
- **Rule-based clinical validation** with a full audit trail, in addition to statistical outlier detection.
- **Dual significance criteria** — statistical significance *and* a pre-registered effect-size threshold.
- **Bootstrap optimism correction** (1,000 iterations) for internal validation, followed by shrinkage with prevalence-preserving intercept re-estimation.
- **Decision Curve Analysis** as a first-class result: whether acting on the model helps, not only whether it discriminates.

## Documentation

The `docs/` directory contains [`methodology_justifications.txt`](docs/methodology_justifications.txt), which documents the clinical rationale behind the implemented data validation rules, including contradiction checks, derived-variable verification, and prevention of data leakage.

## Tech stack

`pandas` · `numpy` · `scipy` · `statsmodels` · `scikit-learn` · `matplotlib` · `seaborn` · `sweetviz`

## Authors

**Elad Segev** and **Chaya Brain** - joint design, implementation, and analysis.

## Academic supervision

Supervised by **Dr. Kirill Vasilchenko**, who provided academic guidance throughout the project.

## License & use

Published for academic and educational purposes. This project was conducted in collaboration with Maayanei HaYeshua Medical Center using an approved clinical dataset. This repository contains code only and does not include patient-level data or identifiable information.

The developed model is a research prototype and has not been externally validated or evaluated prospectively. It must not be used to inform clinical care. Any reuse with patient data requires appropriate institutional approval and data governance procedures.
