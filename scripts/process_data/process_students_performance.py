################################################################################################

import os, json
from ucimlrepo import fetch_ucirepo 
import polars as pl

################################################################################################

# Paths
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..')
processed_data_dir = os.path.join(project_path, 'data', 'processed_data')
os.makedirs(processed_data_dir, exist_ok=True)

################################################################################################

# Fetch dataset 
student_performance = fetch_ucirepo(id=320) 
  
# data (as pandas dataframes) 
X = pl.from_pandas(student_performance.data.features)
y = pl.from_pandas(student_performance.data.targets)
df = pl.concat([X,y], how='horizontal')

################################################################################################

response = 'G3_cat'

# Response categorization
df = df.with_columns((pl.col('G3') / 2).alias('G3'))
df = df.with_columns(pl.col('G3').cut(
    breaks=[5, 6, 7, 9], 
    labels = ["suspenso", "suficiente", "bien", "notable", "sobresaliente"],
    left_closed=True
).alias(response))

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

encoding[response] = {'suspenso': 0, 'suficiente': 1, 'bien': 1, 'notable': 1, 'sobresaliente': 1} # 3 categories ('suspenso' (0), 'aprobado' (1))

for col in cat_predictors + [response]: 
    df = df.with_columns(pl.col(col).replace_strict(encoding[col]).alias(col))

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

metadata_file_name = "metadata_students_performance.json"
processed_data_file_name = "students_performance_processed.parquet"
metadata_file_path = os.path.join(processed_data_dir, metadata_file_name)
processed_data_file_path = os.path.join(processed_data_dir, processed_data_file_name)

with open(metadata_file_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

df.write_parquet(processed_data_file_path)

print(f'✅ Outputs saved successfully at {processed_data_dir}')

################################################################################################