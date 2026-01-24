from sklearn.metrics import accuracy_score, balanced_accuracy_score
EXPERIMENT_RANDOM_STATE = 123 
N_REALIZATIONS = 10 #TODO: 100 en produccion
CHUNK_SIZE = 5

BASE_CONFIG = {
    'random_state': 123,
    'method': 'pam',
    'init': 'heuristic',
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
    'shuffle': True, 
    'kfold_random_state': 111,
    # Valores por defecto para variables que cambian a veces
    'n_clusters': 3, 
    'score_metric': accuracy_score
}

CONFIG_EXPERIMENT = {

    'simulation_testing': {
        **BASE_CONFIG, 
        'frac_sample_size': 0.1,
        'n_splits': 5,
        'n_clusters': 4
    },

    'simulation_1': {
        **BASE_CONFIG, 
        'frac_sample_size': 0.1,
        'n_splits': 5,
        'n_clusters': 4
    },

    'simulation_2': {
        **BASE_CONFIG,
        'frac_sample_size': 0.05,
        'n_splits': 10,
        'n_clusters': 4
    },

    'simulation_3': {
        **BASE_CONFIG,
        'frac_sample_size': 0.005,
        'n_splits': 10,
    },

    'simulation_4': {
        **BASE_CONFIG,
        'frac_sample_size': 0.005,
        'n_splits': 20,
    },

    'simulation_5': {
        **BASE_CONFIG,
        'frac_sample_size': 0.005,
        'n_splits': 10,
    },

    'simulation_6': {
        **BASE_CONFIG,
        'frac_sample_size': 0.005,
        'n_splits': 10,

    },

    'simulation_7': {
        **BASE_CONFIG,
        'frac_sample_size': 0.005,
        'n_splits': 20,
    },

    'dubai_houses': {
        **BASE_CONFIG,
        'frac_sample_size_fast_kmedoids': 0.1,
        'frac_sample_size_fold_fast_kmedoids': 0.9,
        'n_splits': 10,
        'n_clusters': None,
        'p1': None,
        'p2': None,
        'p3': None,
        'score_metric': balanced_accuracy_score
    },

    'heart_disease': {
        **BASE_CONFIG,
        'frac_sample_size_fast_kmedoids': 0.5,
        'frac_sample_size_fold_fast_kmedoids': 0.7,
        'n_splits': 5,
        'n_clusters': None,
        'p1': None,
        'p2': None,
        'p3': None
    },

    'kc_houses': {
        **BASE_CONFIG,
        'frac_sample_size_fast_kmedoids': 0.01,
        'frac_sample_size_fold_fast_kmedoids': 0.6,
        'n_splits': 10,
        'n_clusters': None,
        'p1': None,
        'p2': None,
        'p3': None,
        'score_metric': balanced_accuracy_score
    },

}