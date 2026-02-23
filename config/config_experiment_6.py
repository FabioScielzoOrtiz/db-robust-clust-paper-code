from sklearn.metrics import accuracy_score

EXPERIMENT_RANDOM_STATE = 123 
N_REALIZATIONS = 10
CHUNK_SIZE = 5

CONFIG_EXPERIMENT = {
    'base_method_random_state': EXPERIMENT_RANDOM_STATE,
    'n_splits': 5, 
    'frac_sample_size': 0.1,
    'meta_frac_sample_size': 1,
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
    'score_metric': accuracy_score
}