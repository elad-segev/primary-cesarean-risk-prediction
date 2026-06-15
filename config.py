# -*- coding: utf-8 -*-
"""
Project-level configuration — single source of truth for all constants.
No logic lives here; only paths, variable names, and scalar settings.
"""

DATA_PATH: str = "data/gdm_cohort_extract.csv"

TARGET_VAR: str = "primary_cesarean"

SENSITIVITY_FLAG: str = "was_planned_cs"

# Fill in before running EDA — do not leave empty in production pipelines.
NUMERICAL_VARS: list[str] = []
CATEGORICAL_VARS: list[str] = []

ALPHA_LEVEL: float = 0.05
