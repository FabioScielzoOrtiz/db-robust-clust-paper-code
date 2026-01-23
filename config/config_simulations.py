SIMULATION_CONFIGS = {
    'simulation_testing': {
        'n_samples': 5000,
        'centers': 4,
        'cluster_std': [2, 2, 2, 3],
        'outlier_configs': [
            {'col_name': 'X1', 'prop_above': 0.05, 'sigma': 2},
            {'col_name': 'X2', 'prop_below': 0.05, 'sigma': 2}
        ]
    },
    'simulation_1': {
        'n_samples': 35000,
        'centers': 4,
        'cluster_std': [2, 2, 2, 3],
        'outlier_configs': [
            {'col_name': 'X1', 'prop_above': 0.05, 'sigma': 2},
            {'col_name': 'X2', 'prop_below': 0.05, 'sigma': 2}
        ]
    },
    'simulation_2': {
        'n_samples': 100000,
        'centers': 4,
        'cluster_std': [2, 2, 2, 3],
        'outlier_configs': [
            {'col_name': 'X1', 'prop_above': 0.05, 'sigma': 2},
            {'col_name': 'X2', 'prop_below': 0.05, 'sigma': 2}
        ]
    },
    # ... simulation 3, 4, 5, 6 ...
    'simulation_7': {
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'custom_sampling': [60000, 90000, 150000],
        'outlier_configs': [
            {'col_name': 'X1', 'prop_above': 0.05, 'sigma': 2},
            {'col_name': 'X2', 'prop_below': 0.05, 'sigma': 2}
        ]
    }
}