################################################################################################

import os, pickle
import polars as pl
import pandas as pd
from aif360.sklearn.datasets import fetch_german

################################################################################################

# Paths
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, '..', 'data')

################################################################################################

# Fetch German Credit Data
X, y = fetch_german()
X = X.reset_index(drop=True)
y = pd.DataFrame(y.reset_index(drop=True))
df_raw = pd.concat([y, X], axis=1)
df = pl.from_pandas(df_raw)

################################################################################################

# Sort variables according to data type
ordinal_cols = ['checking_status', 'savings_status', 'employment']
quant_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype != pl.Categorical] + ordinal_cols
cat_cols = [col for col in df.columns if col not in quant_cols]
binary_cols = [col for col in cat_cols if len(df[col].unique()) == 2]
multiclass_cols = [col for col in cat_cols if col not in binary_cols]
df = df[quant_cols + binary_cols + multiclass_cols]

################################################################################################

# Encode categorical variables

encoding = {}
categorical_cols = [col for col in df.columns if df[col].dtype == pl.Categorical]
nominal_cols = [x for x in categorical_cols if x not in ordinal_cols]

# Encoding for nominal cols

for col in nominal_cols:        
    unique_values_sorted = sorted(df[col].unique().to_list())
    new_values = list(range(0, len(unique_values_sorted)))
    encoding[col] = dict(zip(unique_values_sorted, new_values))

# Encoding for ordinal cols

encoding['checking_status'] = {
    'no checking': 0, 
    '<0': 1,
    '0<=X<200': 2,
    '>=200': 3
}

encoding['savings_status'] = {
    'no known savings': 0,
    '<100': 1,
    '100<=X<500': 2,
    '500<=X<1000': 3,
    '>=1000': 4
}

encoding['employment'] = {
    'unemployed': 0,
    '<1': 1,
    '1<=X<4': 2,
    '4<=X<7': 3,
    '>=7': 4
}

# Encoding 
for col in categorical_cols: 
    df = df.with_columns(pl.col(col).replace_strict(encoding[col]).alias(col))

################################################################################################

# Split in Predictors and Response
response = 'credit-risk'
predictors = [col for col in df.columns if col != response]
X = df[predictors]
y = df[response]

################################################################################################

# Compute p1, p2, p3
quant_predictors = [col for col in predictors if col in quant_cols]
binary_predictors = [col for col in predictors if col in binary_cols]
multiclass_predictors = [col for col in predictors if col in multiclass_cols]
p1 = len(quant_predictors)
p2 = len(binary_predictors)
p3 = len(multiclass_predictors)

################################################################################################

n_clusters = len(y.unique())

################################################################################################

# Save outputs

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

output_file_name = "uci_german_credit_processed.pkl"
output_file_path = os.path.join(data_path, output_file_name)

with open(output_file_path, "wb") as f:
    pickle.dump(output, f)

print(f'Outputs saved at {output_file_path}')

################################################################################################