################################################################################################

import os, json, argparse
import polars as pl

################################################################################################

# Configuración del parámetro de entrada
parser = argparse.ArgumentParser(description="Procesar datos de KC Houses.")
parser.add_argument('--mode', type=str, choices=['binary', 'ternary'], required=True,
                    help="Define si la variable de respuesta será 'binary' (2 clases) o 'ternary' (3 clases).")
args = parser.parse_args()
mode = args.mode

print(f"Iniciando procesamiento en modo: {mode.upper()}")

################################################################################################

# Paths
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..', '..')
raw_dat_dir = os.path.join(project_path, 'data', 'raw_data')
processed_data_dir = os.path.join(project_path, 'data', 'processed_data')
os.makedirs(processed_data_dir, exist_ok=True)
data_filename = 'kc_houses.xlsx'
data_file_path = os.path.join(raw_dat_dir, data_filename)

################################################################################################

df = pl.read_excel(data_file_path)

################################################################################################

response = 'price'
excluded_variables = ['id', 'date', 'zipcode', 
                      'yr_renovated', # full of zeros
                      #'sqft_lot', 'sqft_lot15', 
                      'waterfront' # almost a constant variable
                    ]
predictors = [col for col in df.columns if col not in excluded_variables + [response]]
cat_variables = ['view', 'grade']

################################################################################################

# Cuantiles necesarios
q10 = df[response].quantile(0.10)
q90 = df[response].quantile(0.90)

# Lógica condicional basada en el parámetro 'mode'
if mode == 'binary':
    breaks = [q90]
    labels = ['c1', 'c2']
    response_encoding = {'c1': 0, 'c2': 1}
else: # ternary
    breaks = [q10, q90]
    labels = ['c1', 'c2', 'c3']
    response_encoding = {'c1': 0, 'c2': 1, 'c3': 2}

# Aplicar los cortes dinámicamente
df = df.with_columns(pl.col(response).cut(
    breaks=breaks,
    labels=labels,
    left_closed=True
).alias(response))

################################################################################################

encoding = {}

for col in cat_variables:  
    unique_values_sorted = sorted(df[col].unique().to_list())
    new_values = list(range(0, len(unique_values_sorted)))
    encoding[col] = dict(zip(unique_values_sorted, new_values))

################################################################################################

# Aplicar el encoding de la variable de respuesta según el modo
encoding[response] = response_encoding

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

df = df[quant_predictors + binary_predictors + multiclass_predictors + [response]]

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

# Nombres de archivo dinámicos según el modo
metadata_file_name = f"metadata_kc_houses_{mode}.json"
processed_data_file_name = f"kc_houses_{mode}_processed.parquet"

metadata_file_path = os.path.join(processed_data_dir, metadata_file_name)
processed_data_file_path = os.path.join(processed_data_dir, processed_data_file_name)

with open(metadata_file_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

df.write_parquet(processed_data_file_path)

print(f'✅ Outputs ({mode}) saved successfully at {processed_data_dir}')

################################################################################################