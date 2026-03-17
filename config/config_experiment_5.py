########################################################################################################################################################################

import os, sys
import polars as pl
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import warnings
warnings.filterwarnings("ignore")

########################################################################################################################################################################

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..')
sys.path.append(project_path)

########################################################################################################################################################################

from config.config_simulations import SIMULATION_CONFIGS, REAL_DATASET_KEYS

########################################################################################################################################################################

data_dir = os.path.join(project_path, 'data', 'processed_data')
data_path = os.path.join(data_dir, 'datasets_structure.parquet') 
datasets_structure = None
if os.path.exists(data_path):
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

    k: {
        'frac_sample_size_sample_dist_clust': 0.1,
        'frac_sample_size_fold_sample_dist_clust': 0.3,
        'n_splits': 5
    }
    
    for k in list(SIMULATION_CONFIGS.keys()) + REAL_DATASET_KEYS

}

CONFIG_EXPERIMENT.update({
    'heart_disease': {
        'frac_sample_size_sample_dist_clust': 0.5,
        'frac_sample_size_fold_sample_dist_clust': 0.7,
        'n_splits': 3
    }
})

########################################################################################################################################################################

if datasets_structure is not None:
    
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

SCENARIOS = {

    'simulation_base': 'base',
    
    'simulation_size_1': 'big_data',
    'simulation_size_2': 'big_data',

    'simulation_dim_1': 'high_dimensionality',
    'simulation_dim_2': 'high_dimensionality',

    'simulation_num_clusters_1': 'high_num_clusters',
    'simulation_num_clusters_2': 'high_num_clusters',

    'simulation_separation_1': 'low_separation',
    'simulation_separation_2': 'low_separation',

    'simulation_base': 'high_separation',
    'simulation_separation_3': 'high_separation',

    'simulation_corr_1': 'high_correlation',
    'simulation_corr_2': 'high_correlation',
    
    'simulation_outliers_1': 'disperse_outliers',
    'simulation_outliers_2': 'disperse_outliers',

    'simulation_outliers_3': 'grouped_outliers_convex',
    'simulation_outliers_4': 'grouped_outliers_convex',    

    'simulation_outliers_5': 'grouped_outliers_non_convex',
    'simulation_outliers_6': 'grouped_outliers_non_convex',    

    'simulation_imbalance_1': 'imbalance_clusters',
    'simulation_imbalance_2': 'imbalance_clusters',

    'simulation_sphericity_1': 'non_convexity',
    'simulation_sphericity_2': 'non_convexity',
    'simulation_sphericity_3': 'non_convexity',

    'simulation_size_outliers_1': 'big_data_with_disperse_outliers',
    'simulation_size_outliers_2': 'big_data_with_grouped_outliers_non_convex',
    'simulation_size_outliers_3': 'big_data_with_grouped_outliers_convex',

    'simulation_sphericity_outliers_1': 'non_convexity_with_disperse_outliers',
    'simulation_sphericity_outliers_2': 'non_convexity_with_grouped_outliers_non_convex',
    'simulation_sphericity_outliers_3': 'non_convexity_with_grouped_outliers_convex',

    'simulation_imbalance_outliers_1': 'imbalance_clusters_with_disperse_outliers',
    'simulation_imbalance_outliers_2': 'imbalance_clusters_with_grouped_outliers_non_convex',
    'simulation_imbalance_outliers_3': 'imbalance_clusters_with_grouped_outliers_convex',

    'simulation_sphericity_imbalance_1': 'non_convexity_with_imbalance_clusters',
    #'simulation_sphericity_imbalance_2': 'non_convexity_with_imbalance_clusters',
}

########################################################################################################################################################################

DIMENSIONS = {

    'size': ['simulation_base', 'simulation_size_1', 'simulation_size_2'],

    'dimensionality': ['simulation_base', 'simulation_dim_3', 'simulation_dim_1', 'simulation_dim_2'],

    'num_clusters': ['simulation_base','simulation_num_clusters_1', 'simulation_num_clusters_2'],
    
    'separation': [
        'simulation_separation_1', 'simulation_separation_2', 'simulation_base', 'simulation_separation_3'],

    'correlation': ['simulation_base', 'simulation_corr_1', 'simulation_corr_2'],

    'disperse_outliers': ['simulation_base', 'simulation_outliers_1', 'simulation_outliers_2'],

    'grouped_outliers_convex': ['simulation_base', 'simulation_outliers_3', 'simulation_outliers_4'],

    'grouped_outliers_non_convex': ['simulation_base', 'simulation_outliers_5', 'simulation_outliers_6'],

    'imbalance': ['simulation_base', 'simulation_imbalance_2', 'simulation_imbalance_1'],

    'sphericity': ['simulation_base', 'simulation_sphericity_1', 'simulation_sphericity_2', 'simulation_sphericity_3'],

    'size_disperse_outliers': ['simulation_base', 'simulation_outliers_2', 'simulation_size_outliers_1'],
    'size_grouped_outliers_non_convex': ['simulation_base', 'simulation_outliers_6', 'simulation_size_outliers_2'],
    'size_grouped_outliers_convex': ['simulation_base', 'simulation_outliers_3', 'simulation_size_outliers_3'],

    'sphericity_disperse_outliers': ['simulation_base', 'simulation_sphericity_3', 'simulation_sphericity_outliers_1'],
    'sphericity_grouped_outliers_non_convex': ['simulation_base', 'simulation_sphericity_3', 'simulation_sphericity_outliers_2'],
    'sphericity_grouped_outliers_convex': ['simulation_base', 'simulation_sphericity_3', 'simulation_sphericity_outliers_3'],

    'imbalance_disperse_outliers': ['simulation_base', 'simulation_imbalance_1', 'simulation_imbalance_outliers_1'],
    'imbalance_grouped_outliers_non_convex': ['simulation_base', 'simulation_imbalance_1', 'simulation_imbalance_outliers_2'],
    'imbalance_grouped_outliers_convex': ['simulation_base', 'simulation_imbalance_1', 'simulation_imbalance_outliers_3'],

    'sphericity_imbalance': ['simulation_base', 'simulation_imbalance_1', 'simulation_sphericity_imbalance_1'],

    'separation_outliers': ['simulation_base', 'simulation_separation_2', 'simulation_separation_outliers_1'],

    'separation_sphericity': ['simulation_base', 'simulation_separation_2', 'simulation_separation_sphericity_1']

}

########################################################################################################################################################################

DIMENSIONS_FORMATTED = {
    
    'size': {
        'simulation_base': 'Num rows 10000',
        'simulation_size_1': 'Num rows 35000',
        'simulation_size_2': 'Num rows 50000'
    },

    'dimensionality': {
        'simulation_base': 'Num cols 8',
        'simulation_dim_3': 'Num cols 25',
        'simulation_dim_1': 'Num cols 50',
        'simulation_dim_2': 'Num cols 100'
    },

    'num_clusters': {
        'simulation_base': 'Num clusters 3',
        'simulation_num_clusters_1': 'Num clusters 5',
        'simulation_num_clusters_2': 'Num clusters 7'
    },

    'separation': {
        'simulation_separation_1': 'Separation factor 0.2',
        'simulation_separation_2': 'Separation factor 0.5',
        'simulation_base': 'Separation factor 1',
        'simulation_separation_3': 'Separation factor 2'
    },

    'correlation': {
        'simulation_base': 'Prop high corr 0.297',
        'simulation_corr_1': 'Prop high corr 0.385',
        'simulation_corr_2': 'Prop high corr 0.592'
    },

    'disperse_outliers': {
        'simulation_base': 'Without outliers\n Mean prop outliers 0.002',
        'simulation_outliers_1': 'Disperse outliers\n Mean prop outliers 0.047',
        'simulation_outliers_2': 'Disperse outliers\n Mean prop outliers 0.071',
    },

    'grouped_outliers_convex': {
        'simulation_base': 'Without outliers\n Mean prop outliers 0.002',
        'simulation_outliers_3': 'Grouped convex outliers\n Mean prop outliers 0.044\n Outliers dispersion factor 1',
        'simulation_outliers_4': 'Grouped convex outliers\n Mean prop outliers 0.058\n Outliers dispersion factor 4',
    },   

    'grouped_outliers_non_convex': {
        'simulation_base': 'Without outliers\n Mean prop outliers 0.002',
        'simulation_outliers_5': 'Grouped non convex outliers\n Mean prop outliers 0.063\n Outliers anisotropy factor 4',
        'simulation_outliers_6': 'Grouped non convex outliers\n Mean prop outliers 0.073\n Outliers anisotropy factor 12',
    }, 

    'imbalance': {
        'simulation_base': 'Clusters perfect balance',
        'simulation_imbalance_2': 'Clusters props [0.3, 0.25, 0.45]\n Balance ratio 1.8',
        'simulation_imbalance_1': 'Clusters props [0.15, 0.2, 0.65]\n Balance ratio 4.3',
    },

    'sphericity': {
        'simulation_base': 'Anisotropy factor 1',
        'simulation_sphericity_1': 'Anisotropy factor 4',
        'simulation_sphericity_2': 'Anisotropy factor 6',
        'simulation_sphericity_3': 'Anisotropy factor 10'
    },

    'size_disperse_outliers': {
        'simulation_base': 'Num rows 10000\n Without outliers', 
        'simulation_outliers_2': 'Num rows 10000\n Disperse outliers\n Mean prop outliers 0.071',
        'simulation_size_outliers_1': 'Num rows 20000\n Disperse outliers\n Mean prop outliers 0.071'
    },

    'size_grouped_outliers_non_convex': {
        'simulation_base': 'Num rows 10000\n Without outliers', 
        'simulation_outliers_6': 'Num rows 10000\n Grouped non convex outliers\n Mean prop outliers 0.073',
        'simulation_size_outliers_2': 'Num rows 20000\n Grouped non convex outliers\n Mean prop outliers 0.073'
    },

    'size_grouped_outliers_convex': {
        'simulation_base': 'Num rows 10000\n Without outliers', 
        'simulation_outliers_3': 'Num rows 10000\n Grouped convex outliers\n Mean prop outliers 0.044',
        'simulation_size_outliers_3': 'Num rows 20000\n Grouped convex outliers\n Mean prop outliers 0.044'
    },

    'sphericity_disperse_outliers': {
        'simulation_base': 'Anisotropy factor 1\n Without outliers', 
        'simulation_sphericity_3': 'Anisotropy factor 10\n Without outliers',
        'simulation_sphericity_outliers_1': 'Anisotropy factor 10\n Disperse outliers\n Mean prop outliers 0.071'
    },

    'sphericity_grouped_outliers_non_convex': {
        'simulation_base': 'Anisotropy factor 1\n Without outliers', 
        'simulation_sphericity_3': 'Anisotropy factor 10\n Without outliers',
        'simulation_sphericity_outliers_2': 'Anisotropy factor 10\n Grouped non convex outliers\n Mean prop outliers 0.073'
    },

    'sphericity_grouped_outliers_convex': {
        'simulation_base': 'Anisotropy factor 1\n Without outliers', 
        'simulation_sphericity_3': 'Anisotropy factor 10\n Without outliers',
        'simulation_sphericity_outliers_3': 'Anisotropy factor 10\n Grouped convex outliers\n Mean prop outliers 0.044'
    },

    'imbalance_disperse_outliers': {
        'simulation_base': 'Clusters perfect balance\n Without outliers', 
        'simulation_imbalance_1': 'Clusters props [0.15, 0.2, 0.65]\n Balance ratio 4.3\n Without outliers',
        'simulation_imbalance_outliers_1': 'Clusters props [0.15, 0.2, 0.65]\n Balance ratio 4.3\n Disperse outliers\n Mean prop outliers 0.071'
    },

    'imbalance_grouped_outliers_non_convex': {
        'simulation_base': 'Clusters perfect balance\n Without outliers', 
        'simulation_imbalance_1': 'Clusters props [0.15, 0.2, 0.65]\n Balance ratio 4.3\n Without outliers',
        'simulation_imbalance_outliers_2': 'Clusters props [0.15, 0.2, 0.65]\n Balance ratio 4.3\n Grouped non convex outliers\n Mean prop outliers 0.073'
    },

    'imbalance_grouped_outliers_convex': {
        'simulation_base': 'Clusters perfect balance\n Without outliers', 
        'simulation_imbalance_1': 'Clusters props [0.15, 0.2, 0.65]\n Balance ratio 4.3\n Without outliers',
        'simulation_imbalance_outliers_3': 'Clusters props [0.15, 0.2, 0.65]\n Balance ratio 4.3\n Grouped convex outliers\n Mean prop outliers 0.044'
    },

    'sphericity_imbalance': {
        'simulation_base': 'Clusters perfect balance\n Anisotropy factor 1', 
        'simulation_imbalance_1': 'Clusters props [0.15, 0.2, 0.65]\n Balance ratio 4.3\n Anisotropy factor 1',
        'simulation_sphericity_imbalance_1': 'Clusters props [0.15, 0.2, 0.65]\n Balance ratio 4.3\n Anisotropy factor 10',
    },

    'separation_outliers': {
        'simulation_base': 'Separation factor 1\n Without outliers',
        'simulation_separation_2': 'Separation factor 0.5\n Without outliers',
        'simulation_separation_outliers_1': 'Separation factor 0.5\n Disperse outliers\n Mean prop outliers 0.071'
    },

    'separation_sphericity': {
        'simulation_base': 'Separation factor 1\n Anisotropy factor 1',
        'simulation_separation_2': 'Separation factor 0.5\n Anisotropy factor 1',
        'simulation_separation_sphericity_1': 'Separation factor 0.5\n Anisotropy factor 10'        
    }

}

########################################################################################################################################################################

# 1. Definición de las propuestas por dataset
PROPOSALS_REFERENCE_MODELS = {}
PROPOSALS_REFERENCE_MODELS = {
    'dubai_houses': [
        'FoldFastKmedoidsGGower-robust_mahalanobis_winsorized-jaccard-hamming',
        'FastKmedoidsGGower-robust_mahalanobis_winsorized-jaccard-hamming'
        ],
    'global': [
        'FoldFastKmedoidsGGower-robust_mahalanobis_winsorized-sokal-hamming',
        'FastKmedoidsGGower-robust_mahalanobis_winsorized-sokal-hamming',  
    ]
}

########################################################################################################################################################################

# 2. Definición de competidores
COMPETITORS_REFERENCE_MODELS = [
    'KMeans',
    'MiniBatchKMeans',
    'KMedoids-pam',
    'KMedoids-fastpam',
    'KMedoids-fasterpam',
    'KMedoids-fastermsc',
    'CLARA',
    'GaussianMixture',
    'AgglomerativeClustering',
    'SpectralCoclustering',
    'LDAKmeans'
]

########################################################################################################################################################################

# 3. Mapeo final de modelos por dataset
REFERENCE_MODELS = {k: PROPOSALS_REFERENCE_MODELS[k] + COMPETITORS_REFERENCE_MODELS for k in PROPOSALS_REFERENCE_MODELS.keys()}

########################################################################################################################################################################

# 4. Paleta de colores (Actualizada con las variantes Jaccard)
REFERENCE_MODELS_PALETTE = {
    # Variantes Sokal
    'FoldFastKmedoidsGGower-robust_mahalanobis_winsorized-sokal-hamming': "#e82727", 
    'FastKmedoidsGGower-robust_mahalanobis_winsorized-sokal-hamming': "#c51313",      
    # Variantes Jaccard (Mismos colores para mantener consistencia visual)
    'FoldFastKmedoidsGGower-robust_mahalanobis_winsorized-jaccard-hamming': "#e82727", 
    'FastKmedoidsGGower-robust_mahalanobis_winsorized-jaccard-hamming': "#c51313", 
    # Competidores
    'KMeans': "#2c2cf5",  
    'MiniBatchKMeans': "#6774d6",
    'LDAKmeans': "#8dbdf0",
    'CLARA': "#52C812", 
    'GaussianMixture': "#d38e25",      
    'AgglomerativeClustering': "#1A7C0C", 
    'SpectralCoclustering': "#26cce2" ,
    'KMedoids-pam': "#6909c4",
    'KMedoids-fastpam': "#8e39e4",
    'KMedoids-fasterpam': "#a162e1",
    'KMedoids-fastermsc': "#bf29d0",  
}

########################################################################################################################################################################

# 5. Nombres formateados (Corregido el bucle e incluidas las variantes Jaccard)
# Primero extraemos todos los modelos únicos del diccionario y la lista
ALL_UNIQUE_MODELS = set(COMPETITORS_REFERENCE_MODELS)
for models in PROPOSALS_REFERENCE_MODELS.values():
    ALL_UNIQUE_MODELS.update(models)

# Inicializamos el diccionario de nombres base correctamente
REFERENCE_MODELS_FORMATTED_NAMES = {m: m for m in ALL_UNIQUE_MODELS}

# Actualizamos con los nombres limpios para las visualizaciones
REFERENCE_MODELS_FORMATTED_NAMES.update({
    # Sokal
    'FoldFastKmedoidsGGower-robust_mahalanobis_winsorized-sokal-hamming': 'Fold Fast KMedoids',
    'FastKmedoidsGGower-robust_mahalanobis_winsorized-sokal-hamming': 'Fast KMedoids',
    # Jaccard
    'FoldFastKmedoidsGGower-robust_mahalanobis_winsorized-jaccard-hamming': 'Fold Fast KMedoids',
    'FastKmedoidsGGower-robust_mahalanobis_winsorized-jaccard-hamming': 'Fast KMedoids',
})

########################################################################################################################################################################

ADDITIONAL_METHODS_MDS = {
    k:  ['KMeans', 'GaussianMixture', 'KMedoids-pam', 'KMedoids-fasterpam', 'CLARA'] 
    for k in list(SIMULATION_CONFIGS.keys()) + REAL_DATASET_KEYS
}

########################################################################################################################################################################