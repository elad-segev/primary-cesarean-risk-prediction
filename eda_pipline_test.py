import eda_utils as eu
import config as c

df = c.HOLY_DATA.copy()

X = eu.apply_data_schema(df, c.gdm_schema)

"""print(eu.describe_numerical(X))
print('\n\n')
print(eu.describe_categorical(X))
print('\n\n')"""

eu.visualize_feature_distributions(X)




