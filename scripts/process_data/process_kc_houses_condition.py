################################################################################################

import os, pickle
import polars as pl

################################################################################################

# Paths
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..')
processed_data_dir = os.path.join(project_path, 'data', 'processed_data')
data_filename = 'kc_houses.xlsx'
data_file_path = os.path.join(processed_data_dir, data_filename)

################################################################################################

df = pl.read_excel(data_file_path)

################################################################################################

response = 'condition'
excluded_variables = ['id', 'date', 'zipcode', 
                      'yr_renovated', # full of zeros
                      #'sqft_lot', 'sqft_lot15', 
                      'waterfront' # almost a constant variable
                    ]
predictors = [col for col in df.columns if col not in excluded_variables + [response]]
cat_variables = ['view', 'grade']

################################################################################################

encoding = {}

for col in cat_variables:  
    unique_values_sorted = sorted(df[col].unique().to_list())
    new_values = list(range(0, len(unique_values_sorted)))
    encoding[col] = dict(zip(unique_values_sorted, new_values))

################################################################################################

encoding[response] = { # low-medium: 0, medium-high: 1, high: 2
    1: 0,
    2: 0,
    3: 0,
    4: 1,
    5: 2
}

encoding['floors'] = { # 1, 2, 3
    1.5: 1,
    2.5: 2,
    3.5: 3
}


for col in encoding.keys(): 
    try:
        df = df.with_columns(pl.col(col).replace_strict(encoding[col]).alias(col))
    except:
        df = df.with_columns(pl.col(col).replace(encoding[col]).cast(pl.Int64).alias(col))

################################################################################################

quant_to_cat = ['floors']
cat_predictors = [col for col in cat_variables if col != response] + quant_to_cat
quant_predictors = [col for col in predictors if col not in cat_predictors]
binary_predictors = [col for col in cat_predictors if len(df[col].unique()) == 2]
multiclass_predictors = [col for col in cat_predictors if col not in binary_predictors]

################################################################################################

p1 = len(quant_predictors)
p2 = len(binary_predictors)
p3 = len(multiclass_predictors)

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

output_file_name = "kc_houses_processed.pkl"
output_file_path = os.path.join(processed_data_dir, output_file_name)

with open(output_file_path, "wb") as f:
    pickle.dump(output, f)

print(f'✅ Output saved successfully at {output_file_path}')


################################################################################################