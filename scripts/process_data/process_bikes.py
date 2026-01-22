################################################################################################

import os, pickle
import polars as pl

################################################################################################

# Paths
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, '..', 'data')
data_filename = 'bikes.xlsx'
data_file_path = os.path.join(data_path, data_filename)

################################################################################################

df = pl.read_excel(data_file_path)

################################################################################################

df = df.drop_nulls()

################################################################################################

response = 'cnt'
excluded_variables = ['casual', 'registered', 'dteday', 'yrmo']
predictors = [col for col in df.columns if col not in excluded_variables + [response]]

cat_predictors = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday']
quant_predictors = [col for col in predictors if col not in cat_predictors]
binary_predictors = [col for col in cat_predictors if len(df[col].unique()) == 2]
multiclass_predictors = [col for col in cat_predictors if col not in binary_predictors]

################################################################################################

p1 = len(quant_predictors)
p2 = len(binary_predictors)
p3 = len(multiclass_predictors)

################################################################################################

q25 = df[response].quantile(0.25)
q75 = df[response].quantile(0.75)

df = df.with_columns(pl.col(response).cut(
    breaks=[q25, q75],
    labels=[
        'low',
        'medium',
        'high'
    ],
    left_closed=True
))

################################################################################################

encoding = {}

encoding[response] = {
    'low': 0,
    'medium': 1,
    'high': 2         
}

for col in encoding: 
    df = df.with_columns(pl.col(col).replace_strict(encoding[col]).alias(col))

################################################################################################

X = df[quant_predictors + binary_predictors + multiclass_predictors]
y = df[response]

################################################################################################

n_clusters = len(y.unique())

################################################################################################

output = {
    'df': df, 
    'X': X, 
    'y': y, 
    'p1': p1, 
    'p2': p2, 
    'p3': p3,
    'n_clusters': n_clusters,
    'encoding': encoding,
    'quant_predictors': quant_predictors,
    'binary_predictors': binary_predictors,
    'multiclass_predictors': multiclass_predictors
}


output_file_name = "bikes_processed.pkl"
output_file_path = os.path.join(data_path, output_file_name)

with open(output_file_path, "wb") as f:
    pickle.dump(output, f)

print(f'Outputs saved at {output_file_path}')

################################################################################################