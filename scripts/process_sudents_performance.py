################################################################################################

import os, pickle
from ucimlrepo import fetch_ucirepo 
import polars as pl

################################################################################################

# Paths
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, '..', 'data')

################################################################################################

# Fetch dataset 
student_performance = fetch_ucirepo(id=320) 
  
# data (as pandas dataframes) 
X = pl.from_pandas(student_performance.data.features)
y = pl.from_pandas(student_performance.data.targets)

################################################################################################

# Response categorization
y = y.with_columns(pl.col('G3') / 2)
y = y.with_columns(pl.col('G3').cut(
    breaks=[5, 6, 7, 9], 
    labels = ["suspenso", "suficiente", "bien", "notable", "sobresaliente"],
    left_closed=True
).alias('G3_cat')
)

################################################################################################

# Select response
selected_response = 'G3_cat'
y = y[selected_response]

################################################################################################

# Select predictors
selected_predictors = ["failures", "absences", "studytime", "schoolsup", "famsup", "Medu", "Fedu", "famrel", "goout", "Walc", "health"]
X = X[selected_predictors]

################################################################################################

# Encode categorical predictors 

encoding = {}
predictors_to_encode = [col for col in X.columns if X[col].dtype == pl.String] # all nominal
for col in predictors_to_encode:
    unique_values_sorted = sorted(X[col].unique().to_list())
    new_values = list(range(0, len(unique_values_sorted)))
    encoding[col] = dict(zip(unique_values_sorted, new_values))

for col in predictors_to_encode: 
    X = X.with_columns(pl.col(col).replace_strict(encoding[col]).alias(col))

################################################################################################

# Encode response
encoding[y.name] = {'suspenso': 0, 'suficiente': 1, 'bien': 2, 'notable': 3, 'sobresaliente': 4} 
y = y.replace_strict(encoding[y.name]).alias(y.name)

################################################################################################

# Sort variables according to data type

len_unique_values = {}
for col in X.columns:
    len_unique_values[col] = len(X[col].unique())

quant_predictors = ['absences', 'failures']
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

# Save outputs

output = {
    'X': X, 
    'y': y, 
    'p1': p1, 
    'p2': p2, 
    'p3': p3,
    'encoding': encoding,
    'quant_predictors': quant_predictors,
    'binary_predictors': binary_predictors,
    'multiclass_predictors': multiclass_predictors
}

output_file_name = "uci_students_performance_processed.pkl"
output_file_path = os.path.join(data_path, output_file_name)

with open(output_file_path, "wb") as f:
    pickle.dump(output, f)

print(f'Outputs saved at {output_file_path}')

################################################################################################