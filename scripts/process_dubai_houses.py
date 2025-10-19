################################################################################################

import os, pickle
import polars as pl

################################################################################################

# Paths
current_path = os.getcwd()#os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, '..', 'data')
data_filename = 'dubai_houses.csv'
data_file_path = os.path.join(data_path, data_filename)

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


output_file_name = "dubai_houses_processed.pkl"
output_file_path = os.path.join(data_path, output_file_name)

with open(output_file_path, "wb") as f:
    pickle.dump(output, f)

print(f'Outputs saved at {output_file_path}')

################################################################################################