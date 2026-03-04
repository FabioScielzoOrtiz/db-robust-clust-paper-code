########################################################################################################################################################################

import os, sys
import polars as pl
from sklearn.metrics import accuracy_score, balanced_accuracy_score

########################################################################################################################################################################

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
sys.path.append(project_path)

########################################################################################################################################################################

from config.config_simulations import SIMULATION_CONFIGS, REAL_DATASET_KEYS

########################################################################################################################################################################

data_dir = os.path.join(project_path, 'data', 'processed_data')
data_path = os.path.join(data_dir, 'datasets_structure.parquet') 
datasets_structure = pl.read_parquet(data_path)

########################################################################################################################################################################

EXPERIMENT_RANDOM_STATE = 123 
N_REALIZATIONS = 100
CHUNK_SIZE = 5
PROP_ERRORS_THRESHOLD = 0.30

########################################################################################################################################################################

CONFIG_EXPERIMENT = {

    # 'simulation_1': {
    #     **BASE_CONFIG, 
    #     'frac_sample_sizes': [0.0005, 0.005, 0.01, 0.05, 0.1, 0.2, 0.35],
    #     'n_clusters': 4
    # },

    # 'simulation_2': {
    #     **BASE_CONFIG,
    #     'n_clusters': 4
    # },

    # 'simulation_3': {
    #     **BASE_CONFIG,
    # },

    # 'simulation_4': {
    #     **BASE_CONFIG,
    #     'frac_sample_sizes': [0.0005, 0.005, 0.01, 0.05, 0.08]
    # },

    # 'simulation_5': {
    #     **BASE_CONFIG,
    # },

    # 'simulation_6': {
    #     **BASE_CONFIG,

    # },

    # 'simulation_7': {
    #     **BASE_CONFIG,
    # },

    k: {
        'frac_sample_sizes': [0.005, 0.1, 0.2, 0.3, 0.4]
    }
    
    for k in list(SIMULATION_CONFIGS.keys()) + REAL_DATASET_KEYS

}

########################################################################################################################################################################

for data_id, config in CONFIG_EXPERIMENT.items():
    row = datasets_structure.filter(pl.col('data_id') == data_id)
    if not row.is_empty():
        config.update({
            'p1': row['n_quant'][0],
            'p2': row['n_binary'][0],
            'p3': row['n_multiclass'][0],
            'n_clusters': row['n_clusters'][0],
            'alpha': 0.1 if 'outliers' in data_id else 0.05,
            'score_metric': accuracy_score if row['is_balanced'][0] else balanced_accuracy_score
        })

########################################################################################################################################################################