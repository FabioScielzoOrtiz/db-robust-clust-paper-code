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
df = pl.concat([X,y], how='horizontal')

################################################################################################

# Response categorization
'''
df = df.with_columns((pl.col('G3') / 2).alias('G3'))
df = df.with_columns(pl.col('G3').cut(
    breaks=[5, 6, 7, 9], 
    labels = ["suspenso", "suficiente", "bien", "notable", "sobresaliente"],
    left_closed=True
).alias('G3_cat'))
'''

################################################################################################

# Select response
response = 'studytime'

################################################################################################

# Select predictors
selected_predictors = ["failures", "absences", "schoolsup", "famsup", "Medu", "Fedu", "famrel", "goout", "Walc", "health",
                       "sex", "famsize", "Pstatus", "activities", "higher", "internet", "Mjob", "Fjob", "age", "freetime", "G3"]

################################################################################################

df = df[selected_predictors + [response]]

################################################################################################

# Sort variables according to data type

binary_predictors = ["schoolsup", "famsup", "sex", "famsize", "Pstatus", "activities", "higher", "internet"]
multiclass_predictors = ["failures", "Medu", "Fedu", "famrel", "goout", "Walc", "health", "freetime", "Mjob", "Fjob"]
cat_predictors = binary_predictors + multiclass_predictors
quant_predictors = [col for col in df.columns if col not in cat_predictors and col != response]

# Encode categorical predictors 

encoding = {}
for col in cat_predictors:
    unique_values_sorted = sorted(df[col].unique().to_list())
    new_values = list(range(0, len(unique_values_sorted)))
    encoding[col] = dict(zip(unique_values_sorted, new_values))

for col in cat_predictors + [response]: 
    df = df.with_columns(pl.col(col).replace_strict(encoding[col]).alias(col))

################################################################################################

# Encode response
#encoding[y.name] = {'suspenso': 0, 'suficiente': 1, 'bien': 2, 'notable': 3, 'sobresaliente': 4} # 5 categories ('suspenso', 'suficiente', 'bien', 'notable', 'sobresaliente')
#encoding[y.name] = {'suspenso': 0, 'suficiente': 1, 'bien': 1, 'notable': 2, 'sobresaliente': 2} # 3 categories ('suspenso' (0), 'aprobado-bajo' (1), 'aprobado-alto' (2))
#encoding[y.name] = {'suspenso': 0, 'suficiente': 1, 'bien': 1, 'notable': 1, 'sobresaliente': 1} # 3 categories ('suspenso' (0), 'aprobado' (1))

#y = y.replace_strict(encoding[y.name]).alias(y.name)

################################################################################################

X = df[quant_predictors + binary_predictors + multiclass_predictors]
y = df[response]

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