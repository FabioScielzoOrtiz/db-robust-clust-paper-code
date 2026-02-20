from sklearn.metrics import accuracy_score, balanced_accuracy_score

EXPERIMENT_RANDOM_STATE = 123 
N_REALIZATIONS = 10 #TODO: 100 en produccion
CHUNK_SIZE = 5
PROP_ERRORS_THRESHOLD = 0.30

BASE_CONFIG = {
    'method': 'pam',
    'init': 'build',
    'max_iter': 100,
    'p1': 4,
    'p2': 2,
    'p3': 2,
    'd1': 'robust_mahalanobis',
    'd2': 'jaccard',
    'd3': 'hamming',
    'robust_method': 'trimmed',
    'alpha': 0.05,
    'epsilon': 0.05,
    'n_iters': 20,
    'VG_sample_size': 1000,
    'VG_n_samples': 5,
    # Valores por defecto para variables que cambian a veces
    'n_clusters': 3, 
    'frac_sample_sizes': [0.0005, 0.005, 0.01, 0.05, 0.1, 0.20],
    'score_metric': accuracy_score
}

CONFIG_EXPERIMENT = {

    'simulation_testing': {
        **BASE_CONFIG, 
        'frac_sample_sizes': [0.1, 0.2, 0.3, 0.4],
        'n_clusters': 4
    },

    'simulation_1': {
        **BASE_CONFIG, 
        'frac_sample_sizes': [0.0005, 0.005, 0.01, 0.05, 0.1, 0.2, 0.35],
        'n_clusters': 4
    },

    'simulation_2': {
        **BASE_CONFIG,
        'n_clusters': 4
    },

    'simulation_3': {
        **BASE_CONFIG,
    },

    'simulation_4': {
        **BASE_CONFIG,
        'frac_sample_sizes': [0.0005, 0.005, 0.01, 0.05, 0.08]
    },

    'simulation_5': {
        **BASE_CONFIG,
    },

    'simulation_6': {
        **BASE_CONFIG,

    },

    'simulation_7': {
        **BASE_CONFIG,
    },

    'dubai_houses': {
        **BASE_CONFIG,
        'frac_sample_sizes': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'n_clusters': None,
        'p1': None,
        'p2': None,
        'p3': None,
        'score_metric': balanced_accuracy_score
    },

    'heart_disease': {
        **BASE_CONFIG,
        'frac_sample_sizes': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'n_clusters': None,
        'p1': None,
        'p2': None,
        'p3': None
    },

    'kc_houses': {
        **BASE_CONFIG,
        'frac_sample_sizes': [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        'n_clusters': None,
        'p1': None,
        'p2': None,
        'p3': None,
        'score_metric': balanced_accuracy_score
    },

}