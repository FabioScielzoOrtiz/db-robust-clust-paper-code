########################################################################################################################################################################

import pandas as pd
import polars as pl
import numpy as np
from sklearn.datasets import make_blobs

########################################################################################################################################################################

def generate_categorical_features(X, feature_types=None):

    X = pd.DataFrame(X)
    X.columns = [f"X{i}" for i in range(1, X.shape[1] + 1)]
    if feature_types is None:
        return X
      
    n_binary = feature_types.get('n_binary', 0)
    n_multiclass = feature_types.get('n_multiclass', 0)
    n_bins_multiclass = feature_types.get('n_bins_multiclass', 4)

    cols = list(X.columns)
    col_idx = len(cols) - 1

    for _ in range(n_multiclass):
        c = cols[col_idx]
        X[c] = pd.qcut(X[c], q=n_bins_multiclass, labels=False, duplicates='drop')
        col_idx -= 1

    for _ in range(n_binary):
        c = cols[col_idx]
        X[c] = pd.cut(X[c], bins=2, labels=False)
        col_idx -= 1

    return X

########################################################################################################################################################################

def outlier_contamination(X, col_name, prop_below=None, prop_above=None, sigma=2, random_state=123):
   
    X_new = X.copy()
    Q25 = X_new[col_name].quantile(0.25)
    Q75 = X_new[col_name].quantile(0.75)
    IQR = Q75 - Q25
    lower_bound = Q25 - 1.5 * IQR
    upper_bound = Q75 + 1.5 * IQR
    
    np.random.seed(random_state)
    outlier_idx = []
    n_total = len(X_new)
    available_idx = np.arange(n_total)
    
    if prop_below is not None and prop_below > 0:
        n_below = int(n_total * prop_below)
        idx_below = np.random.choice(available_idx, size=n_below, replace=False)
        X_new.loc[idx_below, col_name] = np.random.uniform(lower_bound - sigma*np.abs(lower_bound), lower_bound, size=n_below)
        outlier_idx.extend(idx_below)
        available_idx = np.setdiff1d(available_idx, idx_below) 
        
    if prop_above is not None and prop_above > 0:
        n_above = int(n_total * prop_above)
        idx_above = np.random.choice(available_idx, size=n_above, replace=False)
        X_new.loc[idx_above, col_name] = np.random.uniform(upper_bound, upper_bound + sigma*np.abs(upper_bound), size=n_above)
        outlier_idx.extend(idx_above)

    return X_new, np.array(outlier_idx)

########################################################################################################################################################################

def generate_simulation(
    n_samples=1000, centers=3, cluster_std=1.0, n_features=10, n_redundant=0,
    cluster_proportions=None, 
    anisotropy_factor=1.0, 
    feature_types=None, outlier_configs=None, grouped_outliers_config=None, 
    separation_factor=1.0, return_outlier_idx=False, random_state=42
):
    np.random.seed(random_state)
    
    # --- CONTROL DE REDUNDANCIA ---
    n_features_base = n_features - n_redundant
    if n_features_base <= 0:
        raise ValueError("n_redundant debe ser estrictamente menor que n_features")
    
    # 1. BALANCEO
    if cluster_proportions:
        sizes = [int(n_samples * p) for p in cluster_proportions]
        sizes[-1] += n_samples - sum(sizes) 
        n_samples_input = sizes 
    else:
        n_samples_input = n_samples

    # 2. SEPARABILIDAD Y CENTROS
    if isinstance(centers, (list, np.ndarray)):
        adjusted_centers = np.array(centers) * separation_factor
        c_box = (-10.0, 10.0)
    else:
        adjusted_centers = None if cluster_proportions else centers
        c_box = (-10.0 * separation_factor, 10.0 * separation_factor)

    # 3. BASE (Esferas perfectas en el subespacio base)
    X_arr, y = make_blobs(n_samples=n_samples_input, centers=adjusted_centers, center_box=c_box, 
                          cluster_std=cluster_std, n_features=n_features_base, random_state=random_state)

    # 4. ESFERICIDAD / ANISOTROPÍA (Deformación controlada sin mover los centros)
    if anisotropy_factor > 1.0:
        rng_aniso = np.random.RandomState(random_state)
        # Generar rotación ortogonal aleatoria (Q)
        random_mat = rng_aniso.randn(n_features_base, n_features_base)
        Q, _ = np.linalg.qr(random_mat)
        
        # Matriz de escalado (S): estira solo a lo largo de la primera dimensión rotada
        S = np.eye(n_features_base)
        S[0, 0] = anisotropy_factor
        
        # Transformación final (Rotar + Estirar)
        T = np.dot(Q, S)
        
        # Aplicar solo a la varianza interna de cada clúster, anclando los centros
        X_aniso = np.zeros_like(X_arr)
        for cluster_id in np.unique(y):
            mask = (y == cluster_id)
            centroid = X_arr[mask].mean(axis=0)
            X_centered = X_arr[mask] - centroid
            X_aniso[mask] = np.dot(X_centered, T) + centroid
        X_arr = X_aniso

    # 5. OUTLIERS AGRUPADOS (Nuevos clústeres satélite vinculados a la etiqueta original)
    outlier_indices = []
    if grouped_outliers_config:
        n_out = grouped_outliers_config.get('n_outliers', 50)
        n_groups = grouped_outliers_config.get('n_groups', 2)
        dist = grouped_outliers_config.get('distance', 50.0)
        
        n_out = min(n_out, len(X_arr))
        rng_out = np.random.RandomState(random_state + 1)
        
        out_sizes = [n_out // n_groups] * n_groups
        out_sizes[-1] += n_out - sum(out_sizes)
        
        # Generamos las nubes de puntos de los outliers
        out_centers = rng_out.uniform(-dist, dist, size=(n_groups, n_features_base))
        X_out_all, y_out_all = make_blobs(n_samples=out_sizes, centers=out_centers, 
                                          cluster_std=cluster_std, random_state=random_state + 2)
        
        if anisotropy_factor > 1.0:
            centroid_out = X_out_all.mean(axis=0)
            X_out_all = np.dot((X_out_all - centroid_out), T) + centroid_out
            
        unique_clusters = np.unique(y)
        
        # Asignamos cada grupo de outliers a un clúster original
        for g in range(n_groups):
            size_g = out_sizes[g]
            
            # Elegimos el clúster objetivo (distribuyéndolos equitativamente)
            target_cluster = unique_clusters[g % len(unique_clusters)]
            
            # Buscamos índices disponibles SOLO de ese clúster objetivo
            available_idx = np.where(y == target_cluster)[0]
            size_g = min(size_g, len(available_idx))
            
            # Reemplazamos
            idx_to_replace = rng_out.choice(available_idx, size=size_g, replace=False)
            X_group = X_out_all[y_out_all == g][:size_g]
            
            X_arr[idx_to_replace] = X_group
            outlier_indices.append(idx_to_replace)

    # 5.5 MULTICOLINEALIDAD (Variables redundantes inyectadas sobre la base geométrica)
    if n_redundant > 0:
        rng_red = np.random.RandomState(random_state + 2)
        # Matriz de pesos aleatorios (-1 a 1)
        B = 2 * rng_red.rand(n_features_base, n_redundant) - 1
        X_redundant = np.dot(X_arr, B)
        # Añadir ruido para no hacer colinealidad matemáticamente perfecta
        X_redundant += rng_red.normal(scale=0.1, size=X_redundant.shape)
        # Unir variables base con redundantes (Total de columnas volverá a ser n_features)
        X_arr = np.hstack([X_arr, X_redundant])

    # 6. VARIABLES CAT/BIN
    if feature_types is not None:
        X = generate_categorical_features(X_arr, feature_types)
    else:
        X = X_arr

    # 7. OUTLIERS DISPERSOS
    if outlier_configs:
        for cfg in outlier_configs:
            X, idx_out = outlier_contamination(X, random_state=random_state, **cfg)
            outlier_indices.append(idx_out)

    if return_outlier_idx:
        all_outliers = np.unique(np.concatenate(outlier_indices)) if outlier_indices else np.array([])
        return X, y, all_outliers
        
    return X, y

########################################################################################################################################################################
'''
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
'''
########################################################################################################################################################################    