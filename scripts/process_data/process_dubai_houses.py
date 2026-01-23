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
data_filename = 'dubai_houses.csv'
data_file_path = os.path.join(raw_data_dir, data_filename)

################################################################################################

df = pl.read_csv(data_file_path)

################################################################################################

response = 'quality'
quant_predictors = ['latitude', 'longitude', 'price', 'size_in_sqft', 'price_per_sqft']
binary_predictors = ['balcony', 'barbecue_area', 'private_pool', 'private_garden']
multiclass_predictors = ['no_of_bedrooms', 'no_of_bathrooms']

p1 = len(quant_predictors)
p2 = len(binary_predictors)
p3 = len(multiclass_predictors)

################################################################################################

encoding = {}

encoding[response] = { # 0: Low, 1: Medium-High-Ultra
    'Low': 0,
    'Medium': 1, 
    'High': 1, 
    'Ultra': 1
}

for col in binary_predictors:  
    unique_values_sorted = sorted(df[col].unique().to_list())
    new_values = list(range(0, len(unique_values_sorted)))
    encoding[col] = dict(zip(unique_values_sorted, new_values))

for col in encoding.keys(): 
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

metadata_file_name = "metadata_dubai_houses.json"
processed_data_file_name = "dubai_houses_processed.parquet"
metadata_file_path = os.path.join(processed_data_dir, metadata_file_name)
processed_data_file_path = os.path.join(processed_data_dir, processed_data_file_name)

with open(metadata_file_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

df.write_parquet(processed_data_file_path)

print(f'✅ Outputs saved successfully at {processed_data_dir}')

################################################################################################