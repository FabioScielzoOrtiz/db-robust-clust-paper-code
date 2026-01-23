########################################################################################################################################################################

import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from db_robust_clust.data import outlier_contamination


########################################################################################################################################################################

def process_simulated_data(X):
    
    # numpy array to pandas df
    X = pd.DataFrame(X)     

    # Column names 
    X.columns = [f"X{i}" for i in range(1, X.shape[1]+1)]

    # quant variables converted to binary and multi-class
    X['X5'] = pd.cut(X['X5'], bins=[X['X5'].min()-1, X['X5'].mean(), X['X5'].max()+1], labels=False)
    X['X6'] = pd.cut(X['X6'], bins=[X['X6'].min()-1, X['X6'].mean(), X['X6'].max()+1], labels=False)
    X['X7'] = pd.cut(X['X7'], bins=[X['X7'].min()-1, X['X7'].quantile(0.25), X['X7'].quantile(0.50), X['X7'].quantile(0.75), X['X7'].max()+1], labels=False)
    X['X8'] = pd.cut(X['X8'], bins=[X['X8'].min()-1, X['X8'].quantile(0.25), X['X8'].quantile(0.50), X['X8'].quantile(0.75), X['X8'].max()+1], labels=False)   

    return X

########################################################################################################################################################################

def generate_simulation(
    random_state,
    n_samples,
    centers,
    cluster_std,
    outlier_configs = None, # Lista de dicts para configurar outliers
    custom_sampling = False, # Para la lógica de simulation_7
    return_outlier_idx=False
):
    # Lógica custom de muestreo (Simulation 7)
    if custom_sampling:
        X, y = make_blobs(n_samples=450000, centers=centers, cluster_std=cluster_std, n_features=8, random_state=random_state)
        idx = {}
        for size, cluster in zip(custom_sampling, range(centers)):
             # Nota: corrección de seguridad en random choice
            available_indices = np.where(y == cluster)[0]
            idx[cluster] = np.random.choice(available_indices, size=size, replace=False)
        X = np.concatenate([X[idx[c]] for c in range(centers)])
        y = np.concatenate([y[idx[c]] for c in range(centers)])
    else:
        # Lógica estándar
        X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, n_features=8, random_state=random_state)

    X = process_simulated_data(X)

    # Gestión dinámica de outliers
    outlier_indices = []
    if outlier_configs:
        for cfg in outlier_configs:
            # cfg es un dict: {'col_name': 'X1', 'prop_above': 0.05, ...}
            X, idx_out = outlier_contamination(X, random_state=random_state, **cfg)
            outlier_indices.append(idx_out)

    if return_outlier_idx and outlier_indices:
        all_outliers = np.unique(np.concatenate(outlier_indices))
        return X, y, all_outliers
    
    return X, y

########################################################################################################################################################################

'''
def get_simulation_testing(random_state=123, n_samples=5000, return_outlier_idx=False):
        
    # Data simulation
    X, y = make_blobs(n_samples=n_samples, centers=4, cluster_std=[2,2,2,3], n_features=8, random_state=random_state)

    # Process simulated data
    X = process_simulated_data(X)

    # Outlier contamination
    X, outliers_idx_X1 = outlier_contamination(X, col_name='X1', prop_above=0.05, sigma=2, random_state=random_state)
    X, outliers_idx_X2 = outlier_contamination(X, col_name='X2', prop_below=0.05, sigma=2, random_state=random_state)

    if return_outlier_idx:
        outliers_idx = outliers_idx_X1.copy() if np.array_equal(outliers_idx_X1, outliers_idx_X2) else np.unique(np.concatenate([outliers_idx_X1, outliers_idx_X2]))
        return X, y, outliers_idx
    else:
        return X, y

########################################################################################################################################################################

def get_simulation_1(random_state, return_outlier_idx=False):
        
    # Data simulation
    X, y = make_blobs(n_samples=35000, centers=4, cluster_std=[2,2,2,3], n_features=8, random_state=random_state)

    # Process simulated data
    X = process_simulated_data(X)

    # Outlier contamination
    X, outliers_idx_X1 = outlier_contamination(X, col_name='X1', prop_above=0.05, sigma=2, random_state=random_state)
    X, outliers_idx_X2 = outlier_contamination(X, col_name='X2', prop_below=0.05, sigma=2, random_state=random_state)

    if return_outlier_idx:
        outliers_idx = outliers_idx_X1.copy() if np.array_equal(outliers_idx_X1, outliers_idx_X2) else np.unique(np.concatenate([outliers_idx_X1, outliers_idx_X2]))
        return X, y, outliers_idx
    else:
        return X, y

########################################################################################################################################################################

def get_simulation_2(random_state, return_outlier_idx=False):
        
    # Data simulation
    X, y = make_blobs(n_samples=100000, centers=4, cluster_std=[2,2,2,3], n_features=8, random_state=random_state)

    # Process simulated data
    X = process_simulated_data(X)

    # Outlier contamination
    X, outliers_idx_X1 = outlier_contamination(X, col_name='X1', prop_above=0.05, sigma=2, random_state=random_state)
    X, outliers_idx_X2 = outlier_contamination(X, col_name='X2', prop_below=0.05, sigma=2, random_state=random_state)

    if return_outlier_idx:
        outliers_idx = outliers_idx_X1.copy() if np.array_equal(outliers_idx_X1, outliers_idx_X2) else np.unique(np.concatenate([outliers_idx_X1, outliers_idx_X2]))
        return X, y, outliers_idx
    else:
        return X, y

########################################################################################################################################################################

def get_simulation_3(random_state, return_outlier_idx=False):
        
    # Data simulation
    X, y = make_blobs(n_samples=300000, centers=3, cluster_std=[2,2,3], n_features=8, random_state=random_state)

    # Process simulated data
    X = process_simulated_data(X)

    # Outlier contamination
    X, outliers_idx_X1 = outlier_contamination(X, col_name='X1', prop_above=0.05, sigma=2, random_state=random_state)
    X, outliers_idx_X2 = outlier_contamination(X, col_name='X2', prop_below=0.05, sigma=2, random_state=random_state)

    if return_outlier_idx:
        outliers_idx = outliers_idx_X1.copy() if np.array_equal(outliers_idx_X1, outliers_idx_X2) else np.unique(np.concatenate([outliers_idx_X1, outliers_idx_X2]))
        return X, y, outliers_idx
    else:
        return X, y
    
########################################################################################################################################################################

def get_simulation_4(random_state, return_outlier_idx=False):
        
    # Data simulation
    X, y = make_blobs(n_samples=1000000, centers=3, cluster_std=[2,2,3], n_features=8, random_state=random_state)

    # Process simulated data
    X = process_simulated_data(X)

    # Outlier contamination
    X, outliers_idx_X1 = outlier_contamination(X, col_name='X1', prop_above=0.05, sigma=2, random_state=random_state)
    X, outliers_idx_X2 = outlier_contamination(X, col_name='X2', prop_below=0.05, sigma=2, random_state=random_state)

    if return_outlier_idx:
        outliers_idx = outliers_idx_X1.copy() if np.array_equal(outliers_idx_X1, outliers_idx_X2) else np.unique(np.concatenate([outliers_idx_X1, outliers_idx_X2]))
        return X, y, outliers_idx
    else:
        return X, y

########################################################################################################################################################################

def get_simulation_5(random_state, return_outlier_idx=False):
        
    # Data simulation
    X, y = make_blobs(n_samples=300000, centers=3, cluster_std=[2,2,3], n_features=8, random_state=random_state)

    # Process simulated data
    X = process_simulated_data(X)

    # Outlier contamination
    X, outliers_idx_X1 = outlier_contamination(X, col_name='X1', prop_above=0.085, sigma=2, random_state=random_state)
    X, outliers_idx_X2 = outlier_contamination(X, col_name='X2', prop_below=0.10, sigma=2, random_state=random_state)
    X, outliers_idx_X3 = outlier_contamination(X, col_name='X4', prop_below=0.06, sigma=2, random_state=random_state)

    if return_outlier_idx:
        outliers_idx = outliers_idx_X1.copy() if np.array_equal(outliers_idx_X1, outliers_idx_X2, outliers_idx_X3) else np.unique(np.concatenate([outliers_idx_X1, outliers_idx_X2, outliers_idx_X3]))
        return X, y, outliers_idx
    else:
        return X, y
    
########################################################################################################################################################################

def get_simulation_6(random_state):
        
    # Data simulation
    X, y = make_blobs(n_samples=300000, centers=3, cluster_std=[2,2,3], n_features=8, random_state=random_state)

    # Process simulated data
    X = process_simulated_data(X)

    return X, y
    
########################################################################################################################################################################

def get_simulation_7(random_state, return_outlier_idx=False):
           
    centers = 3 

    X, y = make_blobs(n_samples=450000, centers=centers, cluster_std=[2,2,3], n_features=8, random_state=random_state)
 
    # Seleccionar aleatoriamente puntos de cada cluster
    idx = {}
    n_samples_list = [60000, 90000, 150000]
    for size, cluster in zip(n_samples_list, range(centers)):
        idx[cluster] = np.random.choice(np.where(y == cluster)[0], size=size, replace=False)

    # Reconstruir X e y con los tamaños deseados
    X = np.concatenate([X[idx[c]] for c in range(centers)])
    y = np.concatenate([y[idx[c]] for c in range(centers)])

    # Process simulated data
    X = process_simulated_data(X)

    # Outlier contamination
    X, outliers_idx_X1 = outlier_contamination(X, col_name='X1', prop_above=0.05, sigma=2, random_state=random_state)
    X, outliers_idx_X2 = outlier_contamination(X, col_name='X2', prop_below=0.05, sigma=2, random_state=random_state)

    if return_outlier_idx:
        outliers_idx = outliers_idx_X1.copy() if np.array_equal(outliers_idx_X1, outliers_idx_X2) else np.unique(np.concatenate([outliers_idx_X1, outliers_idx_X2]))
        return X, y, outliers_idx
    else:
        return X, y

'''

########################################################################################################################################################################    

    



