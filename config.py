import pandas as pd
# configuration

OUTPUT_DIR: str = "output"

OUTPUT_DIR_FOR_TABLES = "output/Dataframes"

OUTPUT_DIR_FOR_GRAPH = "output/graphs"

DATA_DIR:str = "data"

DATA_PATH:str = DATA_DIR + "/_GDM_COHORT_EXTRACTION_PRIMARY_CESAREAN_PREDICTION_IN_DIABETIC_M_202606021746_posefix.csv"

HOLY_DATA = pd.read_csv(DATA_PATH)

DF_FOR_MODEL = pd.read_csv(DATA_DIR + "/df_for_model.csv")

TARGET_VAR: str = "primary_cesarean"

FILTER_VAR: str = "was_planned_cs"



# bin_cat = [primary_cesarean, was_planned_cs, anemia, hba1c_recorded, oxytocin_recorded, 
#           insulin_recorded, metformin_recorded, antihypertensive_recorded, induction,meconium, ctg_performed, chronic_htn,
#           gestational_htn, preeclampsia, any_htn, polyhydramnios, oligohydramnios, prom, glucose_poc_any, glucose_any,
#           glucose_lab_any, fetal_presentation]

# multy_cat = [parity,start_mode_raw,membranes_color_raw,membranes_type_raw]

# non_normal_cont_vars = [mother_age,weight_pre_pregnancy,weight_at_admission,bmi_computed,weight_gain,hemoglobin_min,glucose_max,gestational_age_weeks]

# normal_cont_vars  = [height_cm, hemoglobin_first, birth_weight_g]

# datetime_vars = [birth_date, admission_date]



gdm_schema = {
    # Identifiers and Dates
    "mother_medical_record": {"type": "identifier"},
    "patient_id": {"type": "identifier"},
    "birth_date": {"type": "datetime"},
    "admission_date": {"type": "datetime"},
    
    # Demographics and Anthropometry (Continuous)
    "mother_age": {"type": "continuous"},
    "height_cm": {"type": "continuous"},
    "weight_pre_pregnancy": {"type": "continuous"},
    "weight_at_admission": {"type": "continuous"},
    "bmi_computed": {"type": "continuous"},
    "weight_gain": {"type": "continuous"},
    
    # Laboratory (Continuous)
    "hemoglobin_first": {"type": "continuous"},
    "hemoglobin_min": {"type": "continuous"},
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
    "glucose_poc_any": {"type": "binary"},
    "glucose_any": {"type": "binary"},
    "glucose_lab_any": {"type": "binary"},
    
    
    # Nominal / Categorical
    "start_mode_raw": {"type": "nominal"},
    "fetal_presentation": {"type": "nominal"},
    "membranes_color_raw": {"type": "nominal"},
    "membranes_type_raw": {"type": "nominal"},
    "parity": {"type": "nominal"}
}


# -----------------------------
MAIN_MODEL_VARS: list[str] =  ["mother_age", 'gestational_age_weeks','fetal_presentation', 'prom', 'glucose_any', 'polyhydramnios', 'preeclampsia']
SENSITIVITY_VARS: list[str] = ["bmi_computed", 'weight_at_admission', 'weight_gain', 'height_cm', 'weight_pre_pregnancy',"glucose_max", "was_plan_cs"] 
LIMITATION_VARS: list[str] = ['hemoglobin_first', 'hemoglobin_min', 'birth_weight_g', 'bmi_pre_pregnancy', 'start_mode(clean)', 'membranes_color(clean)',
                               'parity(clean)', 'membranes_type(clean)', 'log(mother_age)', 'log(weight_pre_pregnancy)', 'log(weight_at_admission)',
                                 'log(bmi_computed)', 'log(hemoglobin_min)', 'log(glucose_max)', 'log(gestational_age_weeks)', 'parity_5G', 'parity_1_3',
                                   'mother_medical_record', 'patient_id', 'birth_date', 'admission_date', 'parity', 'primary_cesarean', 'was_planned_cs',
                                     'anemia', 'glucose_lab_any', 'glucose_poc_any', 'hba1c_recorded', 'oxytocin_recorded', 'insulin_recorded', 'metformin_recorded',
                                       'antihypertensive_recorded', 'start_mode_raw', 'induction', 'membranes_color_raw', 'meconium', 'membranes_type_raw',
                                         'ctg_performed', 'chronic_htn', 'gestational_htn', 'any_htn', 'oligohydramnios']


ALPHA_LEVEL: float = 0.05

GLOBAL_STAT_THRESHOLDS = {
    "pearson": 0.20,
    "spearman": 0.25,
    "mannwhitney": 0.45,
    "ttest": 0.20,
    "chi2": 0.40,
    "anova": 0.04,
    "htest": 0.04
}