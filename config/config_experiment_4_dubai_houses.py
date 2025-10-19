def get_config():
    
    config = {
        'random_state': 123,
        'n_splits': 10, 
        'frac_sample_size_fast_kmedoids': 0.1,
        'frac_sample_size_fold_fast_kmedoids': 0.9,
        'n_clusters': None,
        'method': 'pam',
        'init': 'heuristic',
        'max_iter': 100,
        'p1': None,
        'p2': None,
        'p3': None,
        'alpha': 0.05,
        'epsilon': 0.05,
        'n_iters': 20,
        'VG_sample_size': 1000,
        'VG_n_samples': 5,
        'shuffle': True, 
        'kfold_random_state': 111,
    }

    return config
