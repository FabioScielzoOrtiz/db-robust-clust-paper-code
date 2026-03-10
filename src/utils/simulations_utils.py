########################################################################################################################################################################

import pandas as pd
import polars as pl
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles

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
'''
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
'''

########################################################################################################################################################################

def inject_outliers(X, y, config, random_state=123):
    """
    Inyecta hasta tres tipos de outliers en un dataset X, y.
    Devuelve las matrices modificadas y una lista con los índices alterados.
    """
    X_out = X.copy()
    y_out = y.copy()
    n_total, n_features = X_out.shape
    all_outlier_indices = []
    
    # Extraemos las configuraciones (si no existen, serán None)
    grouped_config = config.get('grouped', None)
    disperse_configs = config.get('dispersed', None)

    # -------------------------------------------------------------------------
    # TIPO 1: OUTLIERS AGRUPADOS (Satélites)
    # -------------------------------------------------------------------------
    if grouped_config:
        rng = np.random.RandomState(random_state + 1)
        n_groups = grouped_config.get('n_groups', 2)
        group_props = grouped_config.get('group_proportions', None)
        dist = grouped_config.get('distance', 50.0)
        geom = grouped_config.get('geometry', 'convex')
        dispersion_factor = grouped_config.get('dispersion_factor', 1.0)
        
        # Calcular tamaños
        if group_props:
            out_sizes = [int(n_total * p) for p in group_props]
            n_groups = len(out_sizes)
        else:
            prop = grouped_config.get('prop_outliers', 0.05)
            n_out = int(n_total * prop)
            out_sizes = [n_out // n_groups] * n_groups
            if n_groups > 0 and sum(out_sizes) < n_out:
                out_sizes[-1] += n_out - sum(out_sizes)
                
        # Estimar el ruido base directamente de los clústeres originales
        unique_clusters = np.unique(y)
        base_noise = np.mean([np.std(X_out[y == c]) for c in unique_clusters])
        
        # Aplicar el factor de dispersión (o usar el 'noise' si el usuario lo fuerza)
        noise_out = grouped_config.get('noise', base_noise * dispersion_factor)
        
        for g in range(n_groups):
            size_g = out_sizes[g]
            if size_g <= 0: continue
                
            target_cluster = unique_clusters[g % len(unique_clusters)]
            
            # Buscar candidatos válidos
            available_idx = np.where(y == target_cluster)[0]
            available_idx = np.setdiff1d(available_idx, all_outlier_indices)
            
            size_g = min(size_g, len(available_idx))
            if size_g <= 0: continue
                
            replace_idx = rng.choice(available_idx, size=size_g, replace=False)
            
            # Generar geometría
            if geom == 'moons' and size_g > 1:
                X_g, _ = make_moons(n_samples=size_g, noise=noise_out, random_state=random_state+g)
                X_g = X_g * (dist / 5.0) 
                if n_features > 2:
                    X_g = np.hstack([X_g, rng.normal(0, noise_out, size=(size_g, n_features - 2))])
            elif geom == 'circles' and size_g > 1:
                X_g, _ = make_circles(n_samples=size_g, noise=noise_out, random_state=random_state+g)
                X_g = X_g * (dist / 5.0)
                if n_features > 2:
                    X_g = np.hstack([X_g, rng.normal(0, noise_out, size=(size_g, n_features - 2))])
            else:
                X_g, _ = make_blobs(n_samples=size_g, centers=[np.zeros(n_features)], 
                                    cluster_std=noise_out, random_state=random_state+g)
                if geom == 'anisotropic':
                    rand_mat = rng.randn(n_features, n_features)
                    Q_g, _ = np.linalg.qr(rand_mat)
                    S_g = np.eye(n_features)
                    S_g[0, 0] = grouped_config.get('anisotropy_factor', 4.0)
                    X_g = np.dot(X_g, np.dot(Q_g, S_g))
            
            # (Falta tu código de posicionar los X_g y guardarlos, ¡asumo que lo tienes debajo!)
            
            # Posicionar como satélite
            centroid = np.mean(X_out[y == target_cluster], axis=0)
            direction = rng.randn(n_features)
            direction /= np.linalg.norm(direction)
            X_g += centroid + direction * dist
            
            X_out[replace_idx] = X_g
            all_outlier_indices.extend(replace_idx)

    # -------------------------------------------------------------------------
    # TIPO 2: OUTLIERS DE SUBESPACIO (Fallos en variables específicas)
    # -------------------------------------------------------------------------
    if disperse_configs:
        rng = np.random.RandomState(random_state + 2)
        
        for f_config in disperse_configs:
            prop_below = f_config.get('prop_below', 0.0)
            prop_above = f_config.get('prop_above', 0.0)
            col_idx = f_config.get('col_idx', 0)
            sigma = f_config.get('sigma', 2.0)
            
            col_data = X_out[:, col_idx]
            q25, q75 = np.percentile(col_data, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            available_idx = np.setdiff1d(np.arange(n_total), all_outlier_indices)
            
            if prop_below > 0:
                n_below = min(int(n_total * prop_below), len(available_idx))
                if n_below > 0:
                    idx_below = rng.choice(available_idx, size=n_below, replace=False)
                    X_out[idx_below, col_idx] = rng.uniform(
                        lower_bound - sigma*np.abs(lower_bound), lower_bound, size=n_below)
                    all_outlier_indices.extend(idx_below)
                    available_idx = np.setdiff1d(available_idx, idx_below)
                    
            if prop_above > 0:
                n_above = min(int(n_total * prop_above), len(available_idx))
                if n_above > 0:
                    idx_above = rng.choice(available_idx, size=n_above, replace=False)
                    X_out[idx_above, col_idx] = rng.uniform(
                        upper_bound, upper_bound + sigma*np.abs(upper_bound), size=n_above)
                    all_outlier_indices.extend(idx_above)

    return X_out, y_out, all_outlier_indices

########################################################################################################################################################################

def generate_simulation(
    n_samples=1000, centers=3, cluster_std=1.0, n_features=10, n_redundant=0,
    cluster_proportions=None, geometry='convex',
    anisotropy_factor=1.0, 
    feature_types=None, 
    outliers_config=None,
    separation_factor=1.0, return_outlier_idx=False, random_state=123
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
   
    # 3. BASE (Geometrías)
    if geometry in ['moons', 'circles']:
        noise = cluster_std[0] if isinstance(cluster_std, list) else cluster_std

        if geometry == 'moons':
            X_base, y = make_moons(n_samples=n_samples_input, noise=noise, random_state=random_state)
        else:
            X_base, y = make_circles(n_samples=n_samples_input, noise=noise, random_state=random_state)
        
        X_base = X_base * separation_factor * 5 

        if n_features_base > 2:
            extra_dims = np.random.normal(0, noise, size=(X_base.shape[0], n_features_base - 2))
            X_arr = np.hstack([X_base, extra_dims])
        else:
            X_arr = X_base
    else:
        X_arr, y = make_blobs(n_samples=n_samples_input, centers=adjusted_centers, center_box=c_box, 
                              cluster_std=cluster_std, n_features=n_features_base, random_state=random_state)

    # 4. ESFERICIDAD / ANISOTROPÍA
    if anisotropy_factor > 1.0:
        rng_aniso = np.random.RandomState(random_state)
        random_mat = rng_aniso.randn(n_features_base, n_features_base)
        Q, _ = np.linalg.qr(random_mat)
        
        S = np.eye(n_features_base)
        S[0, 0] = anisotropy_factor
        
        T = np.dot(Q, S)
        
        X_aniso = np.zeros_like(X_arr)
        for cluster_id in np.unique(y):
            mask = (y == cluster_id)
            centroid = X_arr[mask].mean(axis=0)
            X_centered = X_arr[mask] - centroid
            X_aniso[mask] = np.dot(X_centered, T) + centroid
        X_arr = X_aniso

    # 5. MULTICOLINEALIDAD (Inyección directa de correlación)
    if n_redundant > 0:
        rng_red = np.random.RandomState(random_state + 2)
        X_redundant = np.zeros((X_arr.shape[0], n_redundant))
        
        for i in range(n_redundant):
            # 1. Elegimos una variable base al azar para "clonar"
            target_col = rng_red.choice(n_features_base)
            
            # 2. Calculamos su dispersión para que el ruido sea proporcional
            base_col = X_arr[:, target_col]
            col_std = np.std(base_col) + 1e-6
            
            # 3. Generamos ruido gaussiano
            noise = rng_red.normal(0, col_std, size=X_arr.shape[0])
            
            # 4. Creamos la variable redundante (ej: 80% original + 20% ruido)
            # Esto garantiza una correlación altísima casi de 1 a 1 con la variable elegida
            noise_fraction = 0.02  # Puedes subir esto si quieres que la correlación no sea TAN obvia
            X_redundant[:, i] = base_col + (noise * noise_fraction)
            
        X_arr = np.hstack([X_arr, X_redundant])  

    # =========================================================================
    # 6. INYECCIÓN DE OUTLIERS
    # =========================================================================
    # Pasamos directamente el diccionario completo a nuestra función modular
    if outliers_config:
        X_arr, y, all_outlier_indices = inject_outliers(
            X=X_arr, 
            y=y, 
            config=outliers_config, 
            random_state=random_state
        )
    else:
        all_outlier_indices = []

    # 7. VARIABLES CAT/BIN
    if feature_types is not None:
        X = generate_categorical_features(X_arr, feature_types) # Asegúrate de tener esto definido
    else:
        X = X_arr

    # 8. RETORNO
    if return_outlier_idx:
        all_outliers = np.unique(all_outlier_indices) if all_outlier_indices else np.array([])
        return X, y, all_outliers
        
    return X, y

########################################################################################################################################################################