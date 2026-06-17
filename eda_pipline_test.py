import eda_utils as eu
import config as c


# EDA pipeline test


pre_df = c.HOLY_DATA.copy()

df = eu.apply_data_schema(pre_df, c.gdm_schema)


cat_vars, _, _, _, _, _= eu.split_variables_by_type(df)





