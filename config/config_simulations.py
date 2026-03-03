SIMULATION_CONFIGS = {

    # 'simulation_testing': {
    #     'n_samples': 5000,
    #     'centers': 4,
    #     'cluster_std': [2, 2, 2, 3],
    #     'n_features': 8,
    #     'outlier_configs': [
    #         {'col_name': 'X1', 'prop_above': 0.05, 'sigma': 2},
    #         {'col_name': 'X2', 'prop_below': 0.05, 'sigma': 2}
    #     ]
    # },

    # 'simulation_1': {
    #     'n_samples': 35000,
    #     'centers': 4,
    #     'cluster_std': [2, 2, 2, 3],
    #     'n_features': 8,
    #     'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
    #     'outlier_configs': [
    #         {'col_name': 'X1', 'prop_above': 0.05, 'sigma': 2},
    #         {'col_name': 'X2', 'prop_below': 0.05, 'sigma': 2}
    #     ]
    # },

    # 'simulation_2': {
    #     'n_samples': 100000,
    #     'centers': 4,
    #     'cluster_std': [2, 2, 2, 3],
    #     'n_features': 8,
    #     'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
    #     'outlier_configs': [
    #         {'col_name': 'X1', 'prop_above': 0.05, 'sigma': 2},
    #         {'col_name': 'X2', 'prop_below': 0.05, 'sigma': 2}
    #     ],
    # },

    # 'simulation_3': {
    #     'n_samples': 300000,
    #     'centers': 3,
    #     'cluster_std': [2, 2, 3],
    #     'n_features': 8,
    #     'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
    #     'outlier_configs': [
    #         {'col_name': 'X1', 'prop_above': 0.05, 'sigma': 2},
    #         {'col_name': 'X2', 'prop_below': 0.05, 'sigma': 2}
    #     ],
    # },

    # 'simulation_4': {
    #     'n_samples': 1000000,
    #     'centers': 3,
    #     'cluster_std': [2, 2, 3],
    #     'n_features': 8,
    #     'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
    #     'outlier_configs': [
    #         {'col_name': 'X1', 'prop_above': 0.05, 'sigma': 2},
    #         {'col_name': 'X2', 'prop_below': 0.05, 'sigma': 2}
    #     ],
    # },

    # 'simulation_5': {
    #     'n_samples': 300000,
    #     'centers': 3,
    #     'cluster_std': [2, 2, 3],
    #     'n_features': 8,
    #     'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
    #     'outlier_configs': [
    #         {'col_name': 'X1', 'prop_above': 0.085, 'sigma': 2},
    #         {'col_name': 'X2', 'prop_below': 0.10, 'sigma': 2},
    #         {'col_name': 'X3', 'prop_below': 0.06, 'sigma': 2}
    #     ],
    # },

    # 'simulation_6': {
    #     'n_samples': 300000,
    #     'centers': 3,
    #     'cluster_std': [2, 2, 3],
    #     'n_features': 8,
    #     'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
    #     'outlier_configs': None,
    # },
    
    # 'simulation_7': {
    #     'n_samples': None,
    #     'centers': 3,
    #     'cluster_std': [2, 2, 3],
    #     'n_features': 8,
    #     'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
    #     'cluster_proportions': [0.2, 0.3, 0.5],
    #     'outlier_configs': [
    #         {'col_name': 'X1', 'prop_above': 0.05, 'sigma': 2},
    #         {'col_name': 'X2', 'prop_below': 0.05, 'sigma': 2}
    #     ],
    # },

#######################################################################################################################
    
    # BASE 

    'simulation_base': {
        'n_samples': 10000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
    },

#######################################################################################################################

   # SIZE 

    'simulation_size_1': {
        'n_samples': 35000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
    },

    'simulation_size_2': {
        'n_samples': 100000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
    },

#######################################################################################################################
    
    # DIMENSION 

    'simulation_dim_1': { # dim increase, adding useful info (not noise) --> increase clusters separation
        'n_samples': 10000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 50, 
        'feature_types': {'n_binary': 12, 'n_multiclass': 13, 'n_bins_multiclass': 4},
    },


    'simulation_dim_2': { # dim increase, adding useful info (not noise) --> increase clusters separation
        'n_samples': 10000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 100, 
        'feature_types': {'n_binary': 25, 'n_multiclass': 25, 'n_bins_multiclass': 4},
    },

#######################################################################################################################
    
    # NUMBER CLUSTERS 

    'simulation_num_clusters_1': {
        'n_samples': 10000,
        'centers': 5,
        'cluster_std': 2,
        'n_features': 8,
        'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
    },

    'simulation_num_clusters_2': {
        'n_samples': 10000,
        'centers': 10,
        'cluster_std': 2,
        'n_features': 8,
        'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
    },

#######################################################################################################################

   # SEPARATION 

    'simulation_separation_1': {
        'n_samples': 10000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
        'separation_factor': 0.2
    },

    'simulation_separation_2': {
        'n_samples': 10000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
        'separation_factor': 0.5
    },

    'simulation_separation_3': {
        'n_samples': 10000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
        'separation_factor': 0.7
    },


    'simulation_separation_4': {
        'n_samples': 10000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
        'separation_factor': 2
    },

#######################################################################################################################

   # CORRELATION / REDUNDANCY 

    'simulation_corr_1': {
        'n_samples': 10000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
        'n_redundant': 1
    },

    'simulation_corr_2': {
        'n_samples': 10000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
        'n_redundant': 4
    },

#######################################################################################################################

   # OUTLIERS 

    'simulation_outliers_1': {
        'n_samples': 10000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
        'outlier_configs': [
            {'col_name': 'X1', 'prop_above': 0.1, 'sigma': 2},
            {'col_name': 'X2', 'prop_below': 0.1, 'sigma': 2}
        ]
    },

    'simulation_outliers_2': {
        'n_samples': 10000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
        'outlier_configs': [
            {'col_name': 'X1', 'prop_above': 0.1, 'sigma': 5},
            {'col_name': 'X2', 'prop_below': 0.1, 'sigma': 2},
            {'col_name': 'X3', 'prop_below': 0.1, 'sigma': 2}
        ]
    },

    'simulation_outliers_3': {
        'n_samples': 10000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
        'outlier_configs': None,
        'grouped_outliers_config': {
        'prop_outliers': 0.1, 'n_groups': 3, 'distance': 45
        },
    },

    'simulation_outliers_4': {
        'n_samples': 10000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
        'outlier_configs': None,
        'grouped_outliers_config': {
        'prop_outliers': 0.15, 'n_groups': 3, 'distance': 70
        },
    },

#######################################################################################################################

    # IMBALANCE

    'simulation_imbalance_1': {
        'n_samples': 10000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
        'cluster_proportions': [0.15, 0.2, 0.65],

    },

    'simulation_imbalance_2': {
        'n_samples': 10000,
        'centers': 2,
        'cluster_std': [2, 3],
        'n_features': 8,
        'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
        'cluster_proportions': [0.25, 0.75],
    },    


#######################################################################################################################

    # SPHERICITY 

    'simulation_sphericity_1': {
        'n_samples': 10000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
        'anisotropy_factor': 4
    },

    'simulation_sphericity_2': {
        'n_samples': 10000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
        'anisotropy_factor': 6
    },

    'simulation_sphericity_3': {
        'n_samples': 10000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
        'anisotropy_factor': 10
    },

#######################################################################################################################

    'simulation_geometry_1': {
        'n_samples': 10000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
        'geometry': 'moons'
    },

    'simulation_geometry_2': {
        'n_samples': 10000,
        'centers': 3,
        'cluster_std': [2, 2, 3],
        'n_features': 8,
        'feature_types': {'n_binary': 2, 'n_multiclass': 2, 'n_bins_multiclass': 4},
        'geometry': 'circles'
    },

#######################################################################################################################

}