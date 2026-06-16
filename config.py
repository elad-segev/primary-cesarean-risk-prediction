# -*- coding: utf-8 -*-
"""
Project-level configuration — single source of truth for all constants.
No logic lives here; only paths, variable names, and scalar settings.
"""
import pandas as pd

OUTPUT_DIR: str = "output"


DATA_PATH: str = "data/df.csv"

HOLY_DATA = pd.read_csv(DATA_PATH)

TARGET_VAR: str = "primary_cesarean"

SENSITIVITY_FLAG: str = "was_planned_cs"

gdm_schema = {
    # Identifiers and Dates
    "mother_medical_record": {"type": "identifier"},
    "patient_id": {"type": "identifier"},
    "birth_date": {"type": "datetime"},
    "admission_date": {"type": "datetime"},
    
    # Demographics and Anthropometry (Continuous)
    "mother_age": {"type": "continuous"},
    "parity": {"type": "continuous"}, # we need to test it
    "height_cm": {"type": "continuous"},
    "weight_pre_pregnancy": {"type": "continuous"},
    "weight_at_admission": {"type": "continuous"},
    "bmi_computed": {"type": "continuous"},
    "weight_gain": {"type": "continuous"},
    
    # Laboratory (Continuous)
    "hemoglobin_first": {"type": "continuous"},
    "hemoglobin_min": {"type": "continuous"},
    "glucose_lab_any": {"type": "continuous"},
    "glucose_poc_any": {"type": "continuous"},
    "glucose_any": {"type": "continuous"},
    "glucose_max": {"type": "continuous"},
    
    # Intrapartum (Continuous)
    "gestational_age_weeks": {"type": "continuous"},
    "birth_weight_g": {"type": "continuous"},
    
    # Binary Flags
    "primary_cesarean": {"type": "binary"},
    "was_planned_cs": {"type": "binary"},
    "anemia": {"type": "binary"},
    "hba1c_recorded": {"type": "binary"},
    "oxytocin_recorded": {"type": "binary"},
    "insulin_recorded": {"type": "binary"},
    "metformin_recorded": {"type": "binary"},
    "antihypertensive_recorded": {"type": "binary"},
    "induction": {"type": "binary"},
    "meconium": {"type": "binary"},
    "ctg_performed": {"type": "binary"},
    "chronic_htn": {"type": "binary"},
    "gestational_htn": {"type": "binary"},
    "preeclampsia": {"type": "binary"},
    "any_htn": {"type": "binary"},
    "polyhydramnios": {"type": "binary"},
    "oligohydramnios": {"type": "binary"},
    "prom": {"type": "binary"},
    
    # Nominal / Categorical
    "start_mode_raw": {"type": "nominal"},
    "fetal_presentation": {"type": "nominal"},
    "membranes_color_raw": {"type": "nominal"},
    "membranes_type_raw": {"type": "nominal"}
}




# Fill in before running EDA — do not leave empty in production pipelines.
NUMERICAL_VARS: list[str] = []
CATEGORICAL_VARS: list[str] = []

# Feature subsets for cohort splitting (fill before modelling).
# MAIN_MODEL_VARS: high-coverage variables included in the primary model
#   (e.g. age, BMI, haemoglobin — present for nearly all 3,690 records).
# SENSITIVITY_VARS: partial-coverage variables included only in the
#   sensitivity analysis cohort (smaller effective N after complete-case drop).
MAIN_MODEL_VARS: list[str] = []
SENSITIVITY_VARS: list[str] = []

# Directory where split / cleaned datasets are written by data_prep.py.
PROCESSED_DATA_DIR: str = "data/processed/"

ALPHA_LEVEL: float = 0.05
