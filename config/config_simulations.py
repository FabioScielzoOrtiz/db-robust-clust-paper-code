SIMULATION_CONFIGS = {

    'simulation_testing': {
        'n_samples': 9000,
        'centers': 4,
        'cluster_std': [2, 2, 2, 3],
        'n_features': 8,
        'outlier_configs': [
            {'col_name': 'X1', 'prop_above': 0.05, 'sigma': 2},
            {'col_name': 'X2', 'prop_below': 0.05, 'sigma': 2}
        ]
    },

    'simulation_1': {
        'n_samples': 35000,
        'centers': 4,
        'cluster_std': [2, 2, 2, 3],
        'n_features': 8,
        'outlier_configs': [
            {'col_name': 'X1', 'prop_above': 0.05, 'sigma': 2},
            {'col_name': 'X2', 'prop_below': 0.05, 'sigma': 2}
        ]
    },

    'simulation_2': {
        'n_samples': 100000,
        'centers': 4,
        'cluster_std': [2, 2, 2, 3],
        'n_features': 8,
        'outlier_configs': [
            {'col_name': 'X1', 'prop_above': 0.05, 'sigma': 2},
            {'col_name': 'X2', 'prop_below': 0.05, 'sigma': 2}
        ]
    },

    'simulation_3': {
        'n_samples': 300000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'outlier_configs': [
            {'col_name': 'X1', 'prop_above': 0.05, 'sigma': 2},
            {'col_name': 'X2', 'prop_below': 0.05, 'sigma': 2}
        ]
    },

    'simulation_4': {
        'n_samples': 1000000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'outlier_configs': [
            {'col_name': 'X1', 'prop_above': 0.05, 'sigma': 2},
            {'col_name': 'X2', 'prop_below': 0.05, 'sigma': 2}
        ]
    },

    'simulation_5': {
        'n_samples': 300000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'outlier_configs': [
            {'col_name': 'X1', 'prop_above': 0.085, 'sigma': 2},
            {'col_name': 'X2', 'prop_below': 0.10, 'sigma': 2},
            {'col_name': 'X3', 'prop_below': 0.06, 'sigma': 2}
        ]
    },

    'simulation_6': {
        'n_samples': 300000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'outlier_configs': None
    },
    
    'simulation_7': {
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'custom_sampling': [60000, 90000, 150000],
        'outlier_configs': [
            {'col_name': 'X1', 'prop_above': 0.05, 'sigma': 2},
            {'col_name': 'X2', 'prop_below': 0.05, 'sigma': 2}
        ]
    }

}