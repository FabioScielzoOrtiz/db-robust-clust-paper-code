########################################################################################################################################################################

import os
import polars as pl
from sklearn.metrics import accuracy_score, balanced_accuracy_score

########################################################################################################################################################################

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
data_dir = os.path.join(project_path, 'data', 'processed_data')
data_path = os.path.join(data_dir, 'datasets_structure.parquet') 
datasets_structure = pl.read_parquet(data_path)

########################################################################################################################################################################

EXPERIMENT_RANDOM_STATE = 123 
N_REALIZATIONS = 100
CHUNK_SIZE = 5
MAX_DURATION_MINS = 15 # TimeOut Threshold
PROP_ERRORS_THRESHOLD = 0.30

########################################################################################################################################################################

CONFIG_EXPERIMENT = {

    # 'simulation_testing': {
    #     **BASE_CONFIG, 
    #     'frac_sample_size_sample_dist_clust': 0.1,
    #     'frac_sample_size_fold_sample_dist_clust': 0.1,
    #     'n_splits': 5,
    #     'n_clusters': 4
    # },

    # 'simulation_5': {
    #     **BASE_CONFIG,
    #     'frac_sample_size_sample_dist_clust': 0.005,
    #     'frac_sample_size_fold_sample_dist_clust': 0.005,
    #     'n_splits': 10,
    # },

    # 'simulation_6': {
    #     **BASE_CONFIG,
    #     'frac_sample_size_sample_dist_clust': 0.005,
    #     'frac_sample_size_fold_sample_dist_clust': 0.005,
    #     'n_splits': 10,

    # },

    # 'simulation_7': {
    #     **BASE_CONFIG,
    #     'frac_sample_size_sample_dist_clust': 0.005,
    #     'frac_sample_size_fold_sample_dist_clust': 0.005,
    #     'n_splits': 20,
    # },

    'simulation_base': {
        'frac_sample_size_sample_dist_clust': 0.1, # verify with exp 2a
        'frac_sample_size_fold_sample_dist_clust': 0.2, # verify with exp 3a
        'n_splits': 5 # verify with exp 3a
    },

    'simulation_size_1': {
        'frac_sample_size_sample_dist_clust': 0.1, # verify with exp 2a
        'frac_sample_size_fold_sample_dist_clust': 0.2, # verify with exp 3a
        'n_splits': 5, # verify with exp 3a
    },

    # 'simulation_size_2': {
    #     'frac_sample_size_sample_dist_clust': 0.05,
    #     'frac_sample_size_fold_sample_dist_clust': 0.1,
    #     'n_splits': 10,
    # },

    'simulation_dim_1': {
        'frac_sample_size_sample_dist_clust': 0.1,
        'frac_sample_size_fold_sample_dist_clust': 0.2,
        'n_splits': 5,
    },

    'simulation_num_clusters_1': {
        'frac_sample_size_sample_dist_clust': 0.1,
        'frac_sample_size_fold_sample_dist_clust': 0.2,
        'n_splits': 5,
    },

    'simulation_separation_1': {
        'frac_sample_size_sample_dist_clust': 0.2,
        'frac_sample_size_fold_sample_dist_clust': 0.2,
        'n_splits': 5,
    },

    'simulation_corr_1': {
        'frac_sample_size_sample_dist_clust': 0.1,
        'frac_sample_size_fold_sample_dist_clust': 0.2,
        'n_splits': 5,
    },

    'dubai_houses': {
        'frac_sample_size_sample_dist_clust': 0.1,
        'frac_sample_size_fold_sample_dist_clust': 0.9,
        'n_splits': 10,
    },

    'heart_disease': {
        'frac_sample_size_sample_dist_clust': 0.5,
        'frac_sample_size_fold_sample_dist_clust': 0.7,
        'n_splits': 5
    },

    'kc_houses': {
        'frac_sample_size_sample_dist_clust': 0.01,
        'frac_sample_size_fold_sample_dist_clust': 0.6,
        'n_splits': 10,
    },

}

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

'''
NOT_FEASIBLE_METHODS = {
    k: ['SpectralClustering', 'Dipinit']
    for k in ['simulation_size_1', 'kc_houses']
}

NOT_FEASIBLE_METHODS.update({
    k: ['SpectralClustering', 'KMedoids-pam', 'Diana', 'Birch', 'Dipinit', 'AgglomerativeClustering']
    for k in [f'simulation_size_{i}' for i in range(2, 4 + 1)]
})

NOT_FEASIBLE_METHODS.update({
    k: [] for k in ['dubai_houses', 'heart_disease']
})
'''

########################################################################################################################################################################

RANDOM_STATE_MDS = {
    'simulation_base': 73704,
    'simulation_size_1': 78328,
    'simulation_dim_1': 35084,
    'simulation_num_clusters_1': 3431,
    'simulation_separation_1': 4173,
    'heart_disease': 6863,
    'dubai_houses': 6863,
    'kc_houses': 6802
}

########################################################################################################################################################################

ADDITIONAL_METHODS_MDS = {
    'simulation_base': ['KMeans', 'KMedoids-pam'],
    'simulation_size_1': ['KMeans', 'CLARA'],
    'simulation_dim_1': ['KMeans', 'CLARA'],
    'simulation_num_clusters_1': ['KMeans', 'CLARA'],
    'simulation_separation_1': ['KMeans', 'CLARA'],
    'heart_disease': ['MiniBatchKMeans', 'CLARA'],
    'dubai_houses': ['MiniBatchKMeans', 'CLARA'],
    'kc_houses': ['MiniBatchKMeans', 'CLARA']
}

########################################################################################################################################################################