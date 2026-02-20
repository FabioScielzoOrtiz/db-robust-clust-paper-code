################################################################################################

import os, json
from ucimlrepo import fetch_ucirepo 
import polars as pl

################################################################################################

# Paths
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..', '..')
processed_data_dir = os.path.join(project_path, 'data', 'processed_data')
os.makedirs(processed_data_dir, exist_ok=True)

################################################################################################

# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 

# data (as pandas dataframes) 
X = pl.from_pandas(heart_disease.data.features)
y = pl.from_pandas(heart_disease.data.targets)

################################################################################################

# Remove missing values
df = pl.concat([X,y], how='horizontal').drop_nulls()
X = df[:, :(df.shape[1]-1)]

################################################################################################

# Encode response

response = 'num'
encoding = {response: {2: 1, 3: 1, 4: 1}}
df = df.with_columns(pl.col(response).replace(encoding[response]).alias(response))

################################################################################################

# Sort variables according to data type
len_unique_values = {}
for col in X.columns:
    len_unique_values[col] = len(X[col].unique())

quant_predictors = [col for col, len in len_unique_values.items() if len > 4] + ['ca']
cat_predictors = [col for col in X.columns if col not in quant_predictors]
binary_predictors = [col for col in cat_predictors if len_unique_values[col] == 2]
multiclass_predictors = [col for col in cat_predictors if col not in binary_predictors]

################################################################################################

df = df[quant_predictors + binary_predictors + multiclass_predictors + [response]]

################################################################################################

# Compute p1, p2, p3
p1 = len(quant_predictors)
p2 = len(binary_predictors)
p3 = len(multiclass_predictors)

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

metadata_file_name = "metadata_heart_disease.json"
processed_data_file_name = "heart_disease_processed.parquet"
metadata_file_path = os.path.join(processed_data_dir, metadata_file_name)
processed_data_file_path = os.path.join(processed_data_dir, processed_data_file_name)

with open(metadata_file_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

df.write_parquet(processed_data_file_path)

print(f'✅ Outputs saved successfully at {processed_data_dir}')

################################################################################################