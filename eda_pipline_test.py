import eda_utils as eu
import config as c
import pandas as pd
import matplotlib.pyplot as plt
import model_utils as mu
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import itertools
# EDA pipeline test


"""
SANITY CHECKS and TROUBLESHOOTING SCRIPT

Description:
This script is an independent testing module and is not part of the main EDA 
or Modeling pipelines. It was developed to perform ad-hoc, granular sanity 
checks based on intermediate model outputs. The primary objective is to 
deeply understand the mathematical and clinical mechanics of the dataset, 
investigate anomalies, and refine the data processing logic accordingly.

Key Tests & Validations Performed:

* Parity & Missing Weights: Validates parity categories and weight_gain missingness.

* Weight Consistency Check: Verifies recorded weight gain against calculated weight differences.

* Weight Correlation: Evaluates agreement between recorded and calculated weights using Pearson and Spearman correlations.

* Extreme Weight Outliers: Detects logical contradictions in weight variables indicating possible data errors.

* Missing Overlap Checker: Assesses missingness patterns across weight and BMI variables to evaluate imputation feasibility.

* BMI Alignment: Validates calculated BMI against recorded pre-pregnancy BMI using correlation analyses.

* Clinical Bivariate Analysis: Tests associations between clinical indicators and primary cesarean delivery.

* Feature Interactions & Multicollinearity: Evaluates predictor overlap and potential data leakage.

* Chronological Boundaries Check: Confirms that birth and admission dates fall within the cohort timeline.

* Outlier Deep-Dive: Reviews extreme physiological values to distinguish errors from genuine clinical cases.

* Longitudinal Cohort Tracking: Identifies repeated deliveries to assess patient-level clustering effects.
"""




"""
pre_df = c.HOLY_DATA.copy()


df = eu.apply_data_schema(pre_df, c.gdm_schema)
parity_order = [1, 2, 3, 4, 5, 6, 7, 8]

df['parity'] = pd.Categorical(df['parity'],
                              categories=parity_order, ordered=True)


"""




post_df = c.DF_FOR_MODEL.copy()
df = eu.apply_data_schema(post_df, c.gdm_schema)
parity_order = [1, 2, 3, 4, 5, 6]
df['parity'] = pd.Categorical(df['parity'],
                              categories=parity_order, ordered=True)


print(df['weight_gain'].isna().sum() / df.shape[0])











"""
# mismatches = (df['weight_gain'] == (df['weight_at_admission'] - df['weight_pre_pregnancy'])).sum()
# print(mismatches)

a = df['weight_at_admission'] - df['weight_pre_pregnancy']
b = df['weight_gain']

wiegh_df = df.loc[:,['weight_at_admission', 'weight_pre_pregnancy', 'weight_gain']].copy()

c = a == b 

wiegh_df['gain'] = a

print(wiegh_df[c])

# d = c.sum()
# print(d, a.shape[0], b.shape[0])



"""
"""
valid_data = pd.concat([a, b], axis=1, keys=['a', 'b']).dropna()

pearson_corr, pearson_p = pearsonr(valid_data['a'], valid_data['b'])
spearman_corr, spearman_p = spearmanr(valid_data['a'], valid_data['b'])

print("=== תוצאות המבחנים הסטטיסטיים ===")
print(f"Pearson correlation:  r = {pearson_corr:.4f}, p-value = {pearson_p:.4e}")
print(f"Spearman correlation: rho = {spearman_corr:.4f}, p-value = {spearman_p:.4e}")

plt.figure(figsize=(8, 6))

sns.regplot(
    data=valid_data,
    x='b',
    y='a',
    scatter_kws={'alpha': 0.5, 'color': '#1f77b4'},
    line_kws={'color': '#d62728', 'linewidth': 2}
)

plt.title("Calculated Difference (a) vs. Recorded Weight Gain (b)", fontsize=13, pad=12)
plt.xlabel("Recorded Weight Gain: b = df['weight_gain']", fontsize=11)
plt.ylabel("Calculated Difference: a = Admission - Pre-pregnancy", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)

stats_text = (
    f"Pearson r = {pearson_corr:.3f} (p = {pearson_p:.1e})\n"
    f"Spearman ρ = {spearman_corr:.3f} (p = {spearman_p:.1e})"
)
plt.gca().text(
    0.05, 0.93, stats_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray')
)

plt.tight_layout()
plt.show()
"""





"""
d =df[df['weight_gain'] < -10]['weight_gain']

e = df['weight_pre_pregnancy'] - df['weight_at_admission'] > 10.0

a = df.loc[e, ['weight_at_admission', 'weight_pre_pregnancy', 'weight_gain']]
"""
# print(a)


"""
def check_missing_overlap(df, lst):
    results = []
    
 
    for col1, col2 in itertools.combinations(lst, 2):
        mask1 = df[col1].isna()
        mask2 = df[col2].isna()
        
        both_missing = (mask1 & mask2).sum()
        either_missing = (mask1 | mask2).sum()
        
        if either_missing > 0:
            overlap_pct = (both_missing / either_missing) * 100
        else:
            overlap_pct = 0.0
        
        results.append({
            'var1': col1,
            'var2': col2,
            'shared_missing_count': both_missing,
            'overlap_percentage': round(overlap_pct, 2)
        })
        
    return pd.DataFrame(results).sort_values(by='overlap_percentage', ascending=False)

    
lst = ["bmi_computed", 'weight_at_admission', 'weight_gain', 'weight_pre_pregnancy'] 
overlap_results = check_missing_overlap(df, lst)
print(overlap_results)
"""






"""


valid_data = df[['bmi_pre_pregnancy', 'bmi_computed']].dropna()

x = valid_data['bmi_pre_pregnancy']
y = valid_data['bmi_computed']

plt.figure(figsize=(9, 6))

sns.regplot(
    x=x, 
    y=y, 
    scatter_kws={'alpha': 0.5, 'color': '#1f77b4'}, 
    line_kws={'color': 'red', 'linewidth': 2}       
)

plt.title('Relationship between Pre-Pregnancy BMI and Computed BMI', fontsize=14)
plt.xlabel('BMI Pre-Pregnancy', fontsize=12)
plt.ylabel('BMI Computed', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


pearson_corr, pearson_p = pearsonr(x, y)

spearman_corr, spearman_p = spearmanr(x, y)

print("Statistical Analysis Results:")
print("-" * 40)
print(f"Pearson Correlation:  r   = {pearson_corr:.4f} | p-value = {pearson_p:.4e}")
print(f"Spearman Correlation: rho = {spearman_corr:.4f} | p-value = {spearman_p:.4e}")


"""









"""

crosstab_df = pd.crosstab(df[c.TARGET_VAR], df['meconium'])

crosstab_df.plot(kind='bar', stacked=False)

plt.title(f'Distribution of {"meconium"} vs {c.TARGET_VAR}')
plt.ylabel('Count')
plt.xlabel(c.TARGET_VAR)
plt.xticks(rotation=0)
plt.show()


"""



"""
clean_df = df
# _, clean_df = mu.split_cohort(df)

plt.figure(figsize=(10, 6))

sns.histplot(
    data=clean_df, 
    x='hemoglobin_min', 
    hue='fetal_presentation', 
    kde=True,          
    element='step',    
    stat='count',
    common_norm=False
)

plt.title('Distribution of hemoglobin_min by Fetal Presentation')
plt.xlabel('hemoglobin_min')
plt.ylabel('Count')
plt.show()

"""

"""
# c.TARGET_VAR
# _, df = mu.split_cohort(df)
crosstab_df = pd.crosstab(df[c.TARGET_VAR], df['antihypertensive_recorded'])

crosstab_df.plot(kind='bar', stacked=False)

plt.title(f'Distribution of {"antihypertensive_recorded"} vs {c.TARGET_VAR}')
plt.ylabel('Count')
plt.xlabel(c.TARGET_VAR)
plt.xticks(rotation=0)
plt.show()

"""



# oxytocin_recorded 
"""
_, clean_df = mu.split_cohort(df)
crosstab_df = pd.crosstab(clean_df[c.TARGET_VAR], clean_df['oxytocin_recorded'])

crosstab_df.plot(kind='bar', stacked=False)

plt.title(f'Distribution of {c.TARGET_VAR} vs oxytocin_recorded')
plt.ylabel('Count')
plt.xlabel(c.TARGET_VAR)
plt.xticks(rotation=0)
plt.show()



"""



"""

# min hb vs preeclampsia

df_hb = df[['hemoglobin_min',c.TARGET_VAR]].copy()

crosstab_df = pd.crosstab(df_hb['hemoglobin_min'], df_hb[c.TARGET_VAR])

crosstab_df.plot(kind='bar', stacked=False)

plt.title('Distribution of Induction vs Planned CS')
plt.ylabel('Count')
plt.xlabel('hemoglobin_min')
plt.xticks(rotation=0)
plt.show()
"""

"""
post_df = c.DF_FOR_MODEL.copy()
df = eu.apply_data_schema(post_df, c.gdm_schema)
parity_order = [1, 2, 3, 4, 5, 6]

df['parity'] = pd.Categorical(df['parity'],
                              categories=parity_order, ordered=True)

df = df[['gestational_htn', 'chronic_htn']]

crosstab_data = pd.crosstab(df['gestational_htn'], df['chronic_htn'])


fig, ax = plt.subplots(figsize=(8, 6))

crosstab_data.plot(kind='bar', ax=ax, width=0.35, color=['#1f77b4', '#ff7f0e'], edgecolor='black')

ax.set_title('Distribution of Gestational HTN by Chronic HTN Status', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Gestational Hypertension (0 = No, 1 = Yes)', fontsize=12, labelpad=10)
ax.set_ylabel('Count (Number of Patients)', fontsize=12, labelpad=10)

ax.legend(['Chronic HTN: No (0)', 'Chronic HTN: Yes (1)'], title='Chronic HTN', fontsize=10, title_fontsize=11)


ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

ax.set_axisbelow(True)
ax.yaxis.grid(True, color='gray', linestyle='--', alpha=0.5)

plt.show()
missing_rows = df[df['gestational_htn'].isna() | df['chronic_htn'].isna()]


print(missing_rows)

"""


# range date 
"""
df_r_date = df[['birth_date','admission_date']].copy()
min_bd = df_r_date['birth_date'].min()
max_bd = df_r_date['birth_date'].max()
min_ad = df_r_date['admission_date'].min()
max_ad = df_r_date['admission_date'].max()
print(f"bd: {min_bd} - {max_bd}")
print(f"ad {min_ad} - {max_ad}")
"""


# test the untested vars

"""
fdf = df[['induction' , 'was_planned_cs', 'glucose_any',  'glucose_max']].copy()

crosstab_df = pd.crosstab(fdf['glucose_any'], fdf['glucose_max'])

crosstab_df.plot(kind='bar', stacked=False)

plt.title('Distribution of Induction vs Planned CS')
plt.ylabel('Count')
plt.xlabel('glucose_any')
plt.xticks(rotation=0)
plt.show()
"""



# test dwarf
"""
df_100 = df.copy()

df_100 = df_100[df_100['patient_id'] == "9000168161"]
print(df_100)


"""

# low hight test
"""
seps_df = df[['patient_id', 'height_cm', 'weight_at_admission','weight_gain', 'weight_pre_pregnancy', 'bmi_computed', 'was_planned_cs','primary_cesarean' ]].copy()

seps_df = seps_df[seps_df['height_cm'] < 150]

print(seps_df)


"""
"""
# light
seps_df = df[['patient_id', 'height_cm', 'weight_at_admission','weight_gain', 'weight_pre_pregnancy', 'bmi_computed', 'was_planned_cs','primary_cesarean' ]].copy()

seps_df = seps_df[seps_df['weight_pre_pregnancy'] < 45]

print(seps_df)

"""

"""
seps_df = df[['patient_id', 'height_cm', 'weight_at_admission','weight_gain', 'weight_pre_pregnancy', 'bmi_computed', 'was_planned_cs','primary_cesarean' ]].copy()

seps_df = seps_df[seps_df['weight_gain'].abs() > 15]

print(seps_df)
"""

# chort birth steps
"""
filtered_df = pre_df[pre_df.duplicated(subset=['patient_id'], keep=False)][
    ['patient_id', 'parity', 'primary_cesarean', 'was_planned_cs']
]

condition = filtered_df.groupby('patient_id')['primary_cesarean'].transform('max') == 1
filtered_df = filtered_df[condition]

filtered_df = filtered_df.sort_values(by=['patient_id', 'parity'])

filtered_df.to_csv('filtered_patients_advanced.csv', index=False)

print(filtered_df.shape)
print(filtered_df.head())


filtered_df['primary_cesarean_sum'] = filtered_df.groupby('patient_id')['primary_cesarean'].transform('sum')

filtered_df.to_csv('filtered_patients_with_sum.csv', index=False)

print(filtered_df.head())
"""