################################################################################################

import os, json
import polars as pl

################################################################################################

# Paths
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..')
raw_data_dir = os.path.join(project_path, 'data', 'raw_data')
processed_data_dir = os.path.join(project_path, 'data', 'processed_data')
os.makedirs(processed_data_dir, exist_ok=True)
data_filename = 'bikes.xlsx'
data_file_path = os.path.join(raw_data_dir, data_filename)

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

n_clusters = len(df[response].unique())

################################################################################################

metadata = {
    'p1': p1, 
    'p2': p2, 
    'p3': p3,
    'n_clusters': n_clusters,
    'encoding': encoding,
    'response': response,
    'quant_predictors': quant_predictors,
    'binary_predictors': binary_predictors,
    'multiclass_predictors': multiclass_predictors
}

################################################################################################

metadata_file_name = "metadata_bikes.json"
processed_data_file_name = "bikes_processed.parquet"
metadata_file_path = os.path.join(processed_data_dir, metadata_file_name)
processed_data_file_path = os.path.join(processed_data_dir, processed_data_file_name)

with open(metadata_file_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

df.write_parquet(processed_data_file_path)

print(f'✅ Outputs saved successfully at {processed_data_dir}')

################################################################################################