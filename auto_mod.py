import os
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
import config as c
import model_utils as mu
warnings.filterwarnings("ignore")

# Automated training and validation pipeline for multiple predictive models


OUTPUT_DIR = "output/Experiments_Output"

DATA_PATH = c.DATA_DIR + "/df_for_model.csv" 

experiments_dict = {
"1": ['parity(clean)', 'gestational_age_weeks', 'hemoglobin_min', 'insulin_recorded', 'glucose_any', 'any_htn', 'polyhydramnios', 'preeclampsia', 'prom'],
"2": ['parity(clean)', 'gestational_age_weeks', 'hemoglobin_min', 'insulin_recorded', 'glucose_any', 'any_htn', 'polyhydramnios', 'preeclampsia', 'prom', "ctg_performed"],
"3": ['parity(clean)', 'gestational_age_weeks', 'hemoglobin_min', 'insulin_recorded', 'glucose_any', 'any_htn', 'polyhydramnios', 'preeclampsia', 'prom', "ctg_performed", "induction"],
"4": ['parity(clean)', 'gestational_age_weeks', 'hemoglobin_min', 'insulin_recorded', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "ctg_performed"],
"5": ['parity(clean)', 'gestational_age_weeks', 'hemoglobin_min', 'insulin_recorded', 'glucose_any', 'polyhydramnios', 'any_htn', 'prom', 'ctg_performed'],
"6": ['gestational_age_weeks', 'hemoglobin_min', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', 'ctg_performed'],
"7": ['gestational_age_weeks', 'hemoglobin_min', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', 'ctg_performed', "mother_age"],
"8": ['gestational_age_weeks', 'hemoglobin_min', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', 'ctg_performed', "mother_age", "membranes_color(clean)"],
"9": ['gestational_age_weeks', 'hemoglobin_min', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', 'ctg_performed' , "mother_age", "meconium"],
"10": ["log(mother_age)", 'gestational_age_weeks', 'hemoglobin_min', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', 'ctg_performed'],
"11": ['mother_age', 'log(gestational_age_weeks)', 'hemoglobin_min', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', 'ctg_performed'],
"12": ["mother_age", 'gestational_age_weeks', 'log(hemoglobin_min)', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', 'ctg_performed'],
"13": ["mother_age", 'gestational_age_weeks', 'anemia', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', 'ctg_performed'],
"14": ["mother_age", 'gestational_age_weeks','hemoglobin_first', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', 'ctg_performed'],
"15": ["mother_age", 'gestational_age_weeks','hemoglobin_min', 'glucose_poc_any' ,'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', 'ctg_performed'],
"16": ["mother_age", 'gestational_age_weeks','hemoglobin_min', 'glucose_poc_any', 'polyhydramnios', 'preeclampsia', 'prom', 'ctg_performed'],
"17": ["mother_age", 'gestational_age_weeks','hemoglobin_min', 'glucose_lab_any', 'polyhydramnios', 'preeclampsia', 'prom', 'ctg_performed'],
"18": ["mother_age", 'gestational_age_weeks','hemoglobin_min', 'glucose_any', 'oligohydramnios', 'preeclampsia', 'prom', 'ctg_performed'],
"19": ["mother_age", 'gestational_age_weeks','hemoglobin_min', 'glucose_any', 'polyhydramnios',"hba1c_recorded" , 'preeclampsia', 'prom', 'ctg_performed'],
"20": ["mother_age", 'gestational_age_weeks','hemoglobin_min', 'glucose_any', 'polyhydramnios',"oxytocin_recorded" , 'preeclampsia', 'prom', 'ctg_performed'],
"21": ["mother_age", 'gestational_age_weeks','hemoglobin_min', 'glucose_any', 'polyhydramnios',"oxytocin_recorded", 'prom', 'ctg_performed'],
"22": ["mother_age", 'gestational_age_weeks','hemoglobin_min', 'glucose_any', 'polyhydramnios','preeclampsia',"metformin_recorded", 'prom', 'ctg_performed'],
"23": ["mother_age",'hemoglobin_min', 'glucose_any', 'polyhydramnios','preeclampsia',"metformin_recorded", 'prom', 'ctg_performed'],
"24": ['antihypertensive_recorded', "mother_age", 'gestational_age_weeks', 'hemoglobin_min', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', 'ctg_performed'],
"25": ["antihypertensive_recorded", "mother_age", "gestational_age_weeks", "glucose_any", "polyhydramnios", "preeclampsia", "prom", "ctg_performed"],
"26": ['antihypertensive_recorded', "mother_age",'hemoglobin_min', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', 'ctg_performed'],
"27": ['antihypertensive_recorded', "mother_age", 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', 'ctg_performed'],
"28": ["parity_5G", "mother_age", 'gestational_age_weeks', 'hemoglobin_min', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', 'ctg_performed'],
"29": ["parity_1_3", "mother_age", 'gestational_age_weeks', 'hemoglobin_min', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', 'ctg_performed'],
"30": ['mother_age', 'gestational_age_weeks', 'hemoglobin_min', 'insulin_recorded', 'polyhydramnios', 'preeclampsia', 'prom', 'ctg_performed'],
"38_htn": ['any_htn', "preeclampsia", "gestational_htn", "chronic_htn"],
"39_7_without_ctg_performed": ['gestational_age_weeks', 'hemoglobin_min', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
}

new_sen_models = {
    "1_bmi_computed_sen_39": ["bmi_computed", "mother_age", 'gestational_age_weeks', 'hemoglobin_min', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom'],
    "2_weight_gain_sen_39": ["weight_gain" ,"mother_age", 'gestational_age_weeks', 'hemoglobin_min', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom'],
    "3_weight_at_admission_sen_39": ["weight_at_admission" ,"mother_age", 'gestational_age_weeks', 'hemoglobin_min', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom'],
    "4_height_cm_sen_39": ["height_cm" ,"mother_age", 'gestational_age_weeks', 'hemoglobin_min', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom'],
    "5_weight_pre_pregnancy_sen_39": ["weight_pre_pregnancy" ,"mother_age", 'gestational_age_weeks', 'hemoglobin_min', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom'],
    "6_any_htn_sen_39": ["any_htn" ,"mother_age", 'gestational_age_weeks', 'hemoglobin_min', 'glucose_any', 'polyhydramnios', 'prom'],
}


experiments_dict_after_hb_min_fall_A = {
    "A0_base_model": ['gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
    "A1": ['height_cm', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
    "A2_no_mother_age": ['parity(clean)' , 'hemoglobin_first', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom'],
    "A3": ['hba1c_recorded', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
    "A4_no_gestational_age_weeks": ['metformin_recorded', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
    "A5_no_gestational_age_weeks": ['antihypertensive_recorded', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
    "A6_no_gestational_age_weeks": ['fetal_presentation', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
    "A7": ['induction', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
    "A8": ['membranes_color(clean)', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
    "A9_no_prom": ['membranes_type(clean)', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', "mother_age"],
    "A10": ['oxytocin_recorded', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
    "A11": ['insulin_recorded', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
    "A12": ['meconium', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
    "A13": ['any_htn', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"], 
    "A14_no_preeclampsia": ['hemoglobin_first', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'prom', "mother_age"],
    "A15_no_preeclampsia": ['any_htn', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'prom', "mother_age"],
    "A16_no_mother_age": ['parity(clean)', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom'],
    "A17_no_glucose_any" : ['gestational_age_weeks', 'hba1c_recorded', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
    "A18_(A5)": ['antihypertensive_recorded','gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
    "A19_(A6)": ['fetal_presentation', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
    "A20": ['parity(clean)', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
    "A21": ['gestational_htn' , 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
    "A22": ['chronic_htn' , 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"]
    }

experiments_dict_after_hb_min_fall_B = {
    "B0_base_model(A10)" : ['gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age", "oxytocin_recorded"],
    "B1": ['induction','gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age", "oxytocin_recorded"],
    "B2" : ['hba1c_recorded', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age", "oxytocin_recorded"],
    "B3" : ['antihypertensive_recorded','gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age", "oxytocin_recorded"],
    "B4" : ['fetal_presentation','gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age", "oxytocin_recorded"],
    "B5" : ['membranes_type(clean)','gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age", "oxytocin_recorded"],
    "B6" : ['any_htn','gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age", "oxytocin_recorded"],
    "B7_no_preeclampsia" : ['any_htn','gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'prom', "mother_age", "oxytocin_recorded"],
    "B8" : ['hemoglobin_first','gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age", "oxytocin_recorded"],
    "B9_no_preeclampsia" : ['hemoglobin_first','gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'prom', "mother_age", "oxytocin_recorded"],
    "B10" : ['parity(clean)','gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age", "oxytocin_recorded"],
    "B11_no_mother_age" : ['parity(clean)','gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "oxytocin_recorded"],
    }

experiments_dict_after_hb_min_fall_C = {
    "C0_base_model(A18)" : ['gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age", "antihypertensive_recorded"],
    "C1": ['induction','gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age", "antihypertensive_recorded"],
    "C2" : ['hba1c_recorded', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age", "antihypertensive_recorded"],
    "C3" : ['fetal_presentation','gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age", "antihypertensive_recorded"],
    "C4" : ['membranes_type(clean)','gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age", "antihypertensive_recorded"],
    "C5" : ['any_htn','gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age", "antihypertensive_recorded"],
    "C6_no_preeclampsia" : ['any_htn','gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'prom', "mother_age", "antihypertensive_recorded"],
    "C7" : ['hemoglobin_first','gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age", "antihypertensive_recorded"],
    "C8_no_preeclampsia" : ['hemoglobin_first','gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'prom', "mother_age", "antihypertensive_recorded"],
    "C9" : ['parity(clean)','gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age", "antihypertensive_recorded"],
    "C10_no_mother_age" : ['parity(clean)','gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "antihypertensive_recorded"],
    }

experiments_dict_after_hb_min_fall_D = {
    "D0_base_model(C9)" : ['parity(clean)', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age", "antihypertensive_recorded"],
    "D1_no_preeclampsia":  ['parity(clean)', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'prom', "mother_age", "antihypertensive_recorded"]
    }

experiments_dict_after_hb_min_fall_E = {
    "E1_no_was_planned_cs_rows(C9)" : ['parity(clean)', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age", "antihypertensive_recorded"]
    }

experiments_dict_after_hb_min_fall_F = {
    "F0_base_model(A21)": ['gestational_htn' , 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
    "F1":  ['parity(clean)', 'gestational_htn' , 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"]
    }


final_model = ['fetal_presentation', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"] # G1
experiments_dict_after_hb_min_fall_G = {
    "G0_base_model(A0)": ['gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
    "G1": ['fetal_presentation', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
    "G2": final_model + ['parity(clean)'],
    "G3_no_mother_age": ['parity(clean)', 'fetal_presentation', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom'],
    "G4": final_model + ['any_htn'],
    "G5_no_preeclampsia":  ['fetal_presentation', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'prom', "mother_age"] + ['any_htn'],
    "G9": ['fetal_presentation', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age", 'antihypertensive_recorded'],
    "G10_no_preeclampsia":  ['fetal_presentation', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'prom', "mother_age"] + ['antihypertensive_recorded'],
    }

experiments_dict_after_hb_min_fall_G_no_was_planned = {"G6_no_was_planned(G1)":  ['fetal_presentation', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
                                                       "G7_no_was_planned(G4)":  final_model + ['any_htn'],
                                                       "G8_no_preeclampsia_no_was_planned(G5)":  ['fetal_presentation', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'prom', "mother_age"] + ['any_htn'],
}


final_experiments = {"Z0_base_model(G1)": ['fetal_presentation', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
                      "Z1_weight_pre_pregnancy":['weight_pre_pregnancy', 'fetal_presentation', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
                      "Z2_weight_at_admission":['weight_at_admission', 'fetal_presentation', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
                      "Z3_weight_gain":['weight_gain', 'fetal_presentation', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
                      "Z4_height_cm":['height_cm', 'fetal_presentation', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
                      "Z5_glucose_max":['glucose_max', 'fetal_presentation', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
                      "Z7_bmi_computed": ['bmi_computed'] + ['fetal_presentation', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
                      "Z8_weight_pre_pregnancy_no_fetal_presentation": ['weight_pre_pregnancy'] + ['gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
                      "Z9_weight_at_admission_no_fetal_presentation": ['weight_at_admission'] + ['gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
                      "Z10_weight_gain_no_fetal_presentation": ['weight_gain'] + ['gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
                      "Z11_bmi_computed_no_fetal_presentation": ['bmi_computed'] + ['gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],








                      }
final_experiments_sen_was_planned_eq_0 = {"Z6_eq_0" : ['fetal_presentation', 'gestational_age_weeks', 'glucose_any', 'polyhydramnios', 'preeclampsia', 'prom', "mother_age"],
                                          }



def run_automated_experiments(experiments_dict_:dict=experiments_dict, model_plan:str = "main"):
    """
    Runs an automated end-to-end modeling pipeline for one or more predefined feature sets.

    This function executes a complete predictive modeling workflow for each
    experiment defined in the supplied experiment dictionary. Each experiment is
    performed independently and documented in a dedicated text report, while a
    summary of all experiments is saved as a CSV file.

    For every experiment, the pipeline performs data preparation, complete-case
    analysis, EPV assessment, feature selection, logistic regression model fitting,
    bootstrap internal validation, model performance evaluation, coefficient
    shrinkage, and generation of publication-ready Odds Ratio tables. Intermediate
    results, model diagnostics, and validation statistics are written to a
    structured report to provide a complete audit trail of the modeling process.

    If an experiment fails at any stage, the error is recorded and execution
    continues with the remaining experiments.

    :param experiments_dict_: Dictionary mapping experiment identifiers to lists
                            of candidate predictor variables.
    :param model_plan: Analysis mode to execute. The default ("main") performs the
                    primary analysis, while any other value is interpreted as a
                    sensitivity analysis plan and filters the cohort
                    accordingly.

    :type experiments_dict_: Dict[Any, List[str]]
    :type model_plan: str

    :return: None.
    :rtype: None
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading dataset...")
    working_df = pd.read_csv(DATA_PATH)

    summary_results = []

    for exp_num, features in experiments_dict_.items():

        file_name = f"Experiment_{exp_num}.txt" if model_plan == 'main' else f"Experiment_{exp_num}_sen_{model_plan}.txt"
        file_path = os.path.join(OUTPUT_DIR, file_name)
        
        print(f"Running Experiment {exp_num}... Saving to {file_name}")
        
        exp_summary = {
            "Experiment_ID": exp_num,
            "Input_Features": ", ".join(features),
            "Num_Input_Features": len(features),
            "Retained_Rows": None,
            "EPV_Passed": None,
            "Features_vs_Max_EPV": None,
            "LASSO_Eliminated": None,
            "Backward_Eliminated_Details": "None",
            "Log_Likelihood": None,
            "AIC": None,
            "BIC": None,
            "AUC_Corrected": None,
            "Calibration_Corrected": None,
            "Status": "Failed"
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            
            def write_section(title, content):
                f.write(f"--- {title} ---\n")
                if isinstance(content, pd.DataFrame):
                    
                    f.write(content.to_string())
                else:
                    f.write(str(content))
                f.write("\n\n\n")

            f.write(f"====================================\n")
            f.write(f"        EXPERIMENT NUMBER {exp_num}\n")
            f.write(f"====================================\n\n")
            
            write_section("Input Features", features)
            
            # ---------------------------------------------------------
            # Step 1: Opptional - Sensitivity Analysis
            # ---------------------------------------------------------
            if model_plan == 'main':
                p_part = f"all rows shape: {working_df.shape} ; mode = {model_plan}"

            else:
                _, working_df = mu.split_cohort(working_df, filter_col=model_plan)
                p_part = f"without `{model_plan} == 1` rows shape: {working_df.shape} ; mode = {model_plan}"

            write_section("Step 1: Opptional - Sensitivity Analysis", p_part)


            try:
                # ---------------------------------------------------------
                # Step 2: Complete Case Analysis
                # ---------------------------------------------------------
                df_complete, (n_dropped, pct_dropped) = mu.apply_complete_case_analysis(working_df, required_cols=features)
                exp_summary["Retained_Rows"] = len(df_complete)

                p_tab = "\n".join([
                    f"Dropped  : {n_dropped:,} rows ({pct_dropped:.3f}%)",
                    f"Retained : {len(df_complete):,} rows",
                    f"Events   : {int(df_complete[c.TARGET_VAR].sum()):,} primary cesareans {df_complete[c.TARGET_VAR].mean():.3%}",
                    f"\n\nshape: {df_complete.shape}"
                ])

                write_section("Step 2: Complete Case Analysis", p_tab)
                


                # ---------------------------------------------------------
                # Step 3: EPV Check
                # ---------------------------------------------------------
                _ , epv_stats = mu.check_epv(df_complete, features=features, target=c.TARGET_VAR, min_epv=10)
                passes_val = epv_stats.loc[epv_stats['Key'] == 'passes', 'Value'].values[0]
                max_vars = epv_stats.loc[epv_stats['Key'] == 'max_variables', 'Value'].values[0]
                n_feat = epv_stats.loc[epv_stats['Key'] == 'n_features', 'Value'].values[0]
                
                exp_summary["EPV_Passed"] = passes_val
                exp_summary["Features_vs_Max_EPV"] = f"{n_feat} / {max_vars}"
                
                write_section("Step 3: EPV Check", epv_stats)
                
                     
                # ---------------------------------------------------------
                # Step 4: Feature Selection (LASSO / Backward)
                # ---------------------------------------------------------
                lasso_result = mu.lasso_feature_selection(df_complete, features=features, target=c.TARGET_VAR, cv=5)
                selected_features = lasso_result["selected"]
                
                eliminated_lasso = lasso_result['eliminated']
                exp_summary["LASSO_Eliminated"] = ", ".join(eliminated_lasso) if eliminated_lasso else "None"

                if len(selected_features) == 0:
                    write_section("Step 4: LASSO Selection", "LASSO eliminated all features (Null Model Collapse).\n* Transitioning to Backward Elimination.")
                    backward_result = mu.backward_elimination(df_complete, features=features,
                                               target=c.TARGET_VAR, p_threshold=0.05)
                    selected_features = backward_result["selected"]

                    if len(selected_features) == 0:
                        write_section("Step 4.5: Backward Elimination", f"Backward Elimination failed (0 features left); Experiment {exp_num} aborted.")
                        exp_summary["Backward_Eliminated_Details"] = "All"
                        summary_results.append(exp_summary)
                        continue

                    else:
                        details = [f"{step['removed_var']} (p={step['p_value']:.4f})" for step in backward_result["steps"]]
                        exp_summary["Backward_Eliminated_Details"] = " | ".join(details)

                        res_t = "\n".join(
                            [f"  Step {step['step']+1}: Removed '{step['removed_var']}' (p-value: {step['p_value']:.4f})" 
                            for step in backward_result["steps"]
                        ])

                        selection_text = f"Selected ({len(selected_features)}): {selected_features}\nEliminated: {backward_result['eliminated']}\nElimination Audit Trail:\n{res_t}"
                        write_section("Step 4.5: Backward Elimination Results", selection_text)

                else:
                    selection_text = f"Selected ({len(selected_features)}): {selected_features}\nEliminated: {lasso_result['eliminated']}\nBest regularisation C: {lasso_result['best_C']:.4f}"
                    write_section("Step 4: LASSO Selection Results", selection_text)
                
                # ---------------------------------------------------------
                # Step 5: Model Training
                # ---------------------------------------------------------
                model = mu.train_logistic_regression(df_complete, features=selected_features)
                exp_summary["Log_Likelihood"] = round(model.llf, 2)
                exp_summary["AIC"] = round(model.aic, 2)
                exp_summary["BIC"] = round(model.bic, 2)

                selection_text ="\n".join(["Model summary (log-likelihood, AIC, BIC):",
                                           f"  Log-likelihood : {model.llf:.2f}",
                                           f"  AIC            : {model.aic:.2f}",
                                           f"  BIC            : {model.bic:.2f}"])
                write_section("Step 5: Model Summary (Log-likelihood, AIC, BIC)", selection_text)
                
                or_table = mu.compute_odds_ratios(model, conf_level=0.95)
                write_section("Step 5: Odds Ratio Table", or_table)
                
                # ---------------------------------------------------------
                # Step 6: Internal Validation (Bootstrap)
                # ---------------------------------------------------------
                apparent, optimisms, n_failed = mu.bootstrap_validate(df_complete, features=selected_features, random_state=42)
                failed_count, failed_pct, failure_reasons= n_failed
                correction = mu.compute_optimism_correction(apparent, optimisms)
                exp_summary["AUC_Corrected"] = correction["corrected"].get("auc")
                exp_summary["Calibration_Corrected"] = correction["corrected"].get("calibration_slope")
                metrics_df = pd.DataFrame({
                    "Apparent": correction["apparent"],
                    "Optimism": correction["mean_optimism"],
                    "Corrected": correction["corrected"]
                })
                reasons_text = (
                    f"  - Missing Target Class (No Events): {failure_reasons['missing_class_1']}\n"
                    f"  - Missing Negative Class: {failure_reasons['missing_class_0']}\n"
                    f"  - Model Convergence Errors: {failure_reasons['model_convergence_error']}\n"
                    f"  - Metrics Computation Errors: {failure_reasons['metrics_calculation_error']}"
                )
                step6_text = (
                    metrics_df.round(4).to_string() + 
                    f"\n\nBootstrap Iterations Failed: {failed_count} ({failed_pct:.1f}%)\n"
                    "Breakdown of Failures:\n" + reasons_text
                )

                write_section("Step 6: Apparent vs Optimism vs Corrected Performance", step6_text)
                
                # ---------------------------------------------------------
                # Step 7:  Clinical Utility and Performance Visualizations
                # ---------------------------------------------------------
                X_eval = sm.add_constant(df_complete[selected_features], has_constant="add")
                y_prob = model.predict(X_eval)
                y_true = df_complete[c.TARGET_VAR].values
                # Full discrimination + calibration metrics 
                metrics = mu.evaluate_model(y_true, y_prob.values)

                metrics_df = pd.DataFrame(list(metrics.items()), columns=["Performance Metric", "Value"])

                selection_text = f"Model Performance Summary\n{metrics_df}"
                write_section("Step 7: Clinical Utility and Performance Visualizations", selection_text)




                # ---------------------------------------------------------
                # Step 8: Shrinkage & Final Table
                # ---------------------------------------------------------
                optimism_slope = correction["corrected"]["calibration_slope"]

                _, shrunk_or_table, _ = mu.apply_model_shrinkage(model, X_eval, shrinkage_factor=optimism_slope)
                
                write_section("Step 8: Optimism-Corrected Odds Ratios (Shrunk OR)", shrunk_or_table)
                
                
                # ---------------------------------------------------------
                # Step 9: Table 2
                # ---------------------------------------------------------
                table2 = mu.table2(Apparent=or_table, Corrected=shrunk_or_table, p_val_tolerance=0)
                write_section("Step 9: Table 2 Apparent  with Corrected ", table2)

                exp_summary["Status"] = "Success"
                summary_results.append(exp_summary)

                f.write(">>> EXPERIMENT COMPLETED SUCCESSFULLY <<<\n")
                
            except Exception as e:
                exp_summary["Status"] = f"Error"
                summary_results.append(exp_summary)
                write_section("CRITICAL ERROR IN EXPERIMENT", f"Experiment {exp_num} failed due to an error:\n{str(e)}")

    summary_df = pd.DataFrame(summary_results)
    summary_csv_path = os.path.join(OUTPUT_DIR, "Experiments_Summary.csv")
    if os.path.exists(summary_csv_path):
        existing_df = pd.read_csv(summary_csv_path)
        combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
        combined_df.drop_duplicates(subset=['Experiment_ID'], keep='last', inplace=True)
        combined_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    else:
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    
    print(f"\nAll experiments finished. Check the '{OUTPUT_DIR}' directory for text files.")


if __name__ == "__main__":

    for key in final_experiments.keys():
        if key in ['Z3_weight_gain']:
            run_automated_experiments({key: final_experiments[key]})
