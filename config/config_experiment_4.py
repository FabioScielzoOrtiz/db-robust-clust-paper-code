from sklearn.metrics import accuracy_score

EXPERIMENT_RANDOM_STATE = 123 
N_REALIZATIONS = 100
CHUNK_SIZE = 5
PROP_ERRORS_THRESHOLD = 0.30

CONFIG_EXPERIMENT = {
    'n_splits': 5, 
    'frac_sample_size': 0.1,
    'meta_frac_sample_size': 1,
    'random_state': 123,
    'method': 'pam',
    'init': 'build',
    'max_iter': 100,
    'p1': 4,
    'p2': 2,
    'p3': 2,
    'd1': 'robust_mahalanobis',
    'd2': 'sokal',
    'd3': 'hamming',
    'robust_method': 'winsorized',
    'alpha': 0.05,
    'shuffle': True, 
    'n_clusters': 4, 
    'score_metric': accuracy_score,
    'data_sizes': [1000, 3000] # [5000, 10000, 20000, 35000]
}