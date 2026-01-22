################################################################################################

import os, pickle
from ucimlrepo import fetch_ucirepo 
import polars as pl

################################################################################################

# Paths
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, '..', 'data')

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
y = pl.DataFrame(df[:, df.shape[1]-1])

################################################################################################

# Encode response
encoding = {'num': {2: 1, 3: 1, 4: 1}}
y = y.with_columns(pl.col('num').replace(encoding['num']).alias('num'))

################################################################################################

# Sort variables according to data type
len_unique_values = {}
for col in X.columns:
    len_unique_values[col] = len(X[col].unique())

quant_predictors = [col for col, len in len_unique_values.items() if len > 4] + ['ca']
cat_predictors = [col for col in X.columns if col not in quant_predictors]
binary_predictors = [col for col in cat_predictors if len_unique_values[col] == 2]
multiclass_predictors = [col for col in cat_predictors if col not in binary_predictors]

X = X[quant_predictors + binary_predictors + multiclass_predictors]

################################################################################################

# Compute p1, p2, p3
p1 = len(quant_predictors)
p2 = len(binary_predictors)
p3 = len(multiclass_predictors)

################################################################################################

n_clusters = len(y.unique())

################################################################################################

# Save outputs

output = {
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

output_file_name = "uci_heart_disease_processed.pkl"
output_file_path = os.path.join(data_path, output_file_name)

with open(output_file_path, "wb") as f:
    pickle.dump(output, f)

print(f'Outputs saved at {output_file_path}')

################################################################################################