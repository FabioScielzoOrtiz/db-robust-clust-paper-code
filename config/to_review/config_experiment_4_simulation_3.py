import numpy as np


def get_config():
    
    config = {
        'random_state': 123,
        'n_splits': 10, 
        'frac_sample_size': 0.05,
        'n_clusters': 4,
        'method': 'pam',
        'init': 'heuristic',
        'max_iter': 100,
        'p1': 4,
        'p2': 2,
        'p3': 2,
        'alpha': 0.05,
        'epsilon': 0.05,
        'n_iters': 20,
        'VG_sample_size': 1000,
        'VG_n_samples': 5,
        'shuffle': True, 
        'kfold_random_state': 111,
    }

    return config
