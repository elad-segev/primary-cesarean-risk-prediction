import eda_utils as eu
import config as c

pre_df = c.HOLY_DATA.copy()

df = eu.apply_data_schema(pre_df, c.gdm_schema)


x, y, z = eu.visualize_outliers_and_proportions(df, )

print(x,y,z, sep="\n\n")



