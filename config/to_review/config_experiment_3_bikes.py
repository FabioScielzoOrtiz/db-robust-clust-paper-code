import numpy as np

def get_config():
    
    config = {
        'n_splits': np.array([5, 10, 15, 20]),
        'frac_sample_sizes': np.array([0.2, 0.4, 0.6, 0.8, 1]), 
        'n_clusters': None,
        'method': 'pam',
        'init': 'heuristic',
        'max_iter': 100,
        'p1': None,
        'p2': None,
        'p3': None,
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
    }

    return config
