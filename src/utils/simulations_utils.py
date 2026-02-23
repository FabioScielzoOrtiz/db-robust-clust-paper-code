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
    n_features,
    outlier_configs=None, 
    custom_sampling=False, 
    return_outlier_idx=False,
    separation_factor=1.0  
):
    # Lógica para aplicar la separabilidad
    # Si centers es un array de coordenadas, multiplicamos las distancias desde el origen.
    # Si es un int (ej. 3 clústeres), escalamos la caja de generación por defecto de sklearn (-10, 10).
    if isinstance(centers, (list, np.ndarray)):
        adjusted_centers = np.array(centers) * separation_factor
        c_box = (-10.0, 10.0)
        n_clusters = len(centers)
    else:
        adjusted_centers = centers
        c_box = (-10.0 * separation_factor, 10.0 * separation_factor)
        n_clusters = centers

    # Lógica custom de muestreo (Simulation 7)
    if custom_sampling:
        X, y = make_blobs(n_samples=450000, centers=adjusted_centers, center_box=c_box, 
                          cluster_std=cluster_std, n_features=n_features, random_state=random_state)
        idx = {}
        for size, cluster in zip(custom_sampling, range(n_clusters)):
            available_indices = np.where(y == cluster)[0]
            idx[cluster] = np.random.choice(available_indices, size=size, replace=False)
        X = np.concatenate([X[idx[c]] for c in range(n_clusters)])
        y = np.concatenate([y[idx[c]] for c in range(n_clusters)])
    else:
        # Lógica estándar
        X, y = make_blobs(n_samples=n_samples, centers=adjusted_centers, center_box=c_box, 
                          cluster_std=cluster_std, n_features=n_features, random_state=random_state)

    X = process_simulated_data(X)

    # Gestión dinámica de outliers
    outlier_indices = []
    if outlier_configs:
        for cfg in outlier_configs:
            X, idx_out = outlier_contamination(X, random_state=random_state, **cfg)
            outlier_indices.append(idx_out)

    if return_outlier_idx and outlier_indices:
        all_outliers = np.unique(np.concatenate(outlier_indices))
        return X, y, all_outliers
    
    return X, y

########################################################################################################################################################################    