EXPERIMENT_RANDOM_STATE = 123 
N_REALIZATIONS = 10 #TODO: 100 en produccion
CHUNK_SIZE = 5
PROP_ERRORS_THRESHOLD = 0.30

CONFIG_EXPERIMENT = {
    'n_splits': 5, 
    'frac_sample_size': 0.1,
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
    'n_clusters': 4, 
    'n_samples_list': [5000, 10000, 20000, 35000]
}