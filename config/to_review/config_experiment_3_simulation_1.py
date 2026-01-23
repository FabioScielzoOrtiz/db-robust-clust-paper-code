import numpy as np

def get_config():
    
    config = {
        'n_splits': np.array([5, 10, 20, 40]),
        'frac_sample_sizes': np.array([0.005, 0.05, 0.1, 0.25]), 
        'n_clusters': 4,
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
    }

    return config
