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
    feature_configs = config.get('feature_specific', None)
    scattered_config = config.get('scattered', None)

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
    if feature_configs:
        rng = np.random.RandomState(random_state + 2)
        
        for f_config in feature_configs:
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

    # -------------------------------------------------------------------------
    # TIPO 3: OUTLIERS DISPERSOS (Ruido uniforme multivariante)
    # -------------------------------------------------------------------------
    if scattered_config:
        rng = np.random.RandomState(random_state + 3)
        prop = scattered_config.get('prop_outliers', 0.05)
        
        n_scattered = int(n_total * prop)
        available_idx = np.setdiff1d(np.arange(n_total), all_outlier_indices)
        n_scattered = min(n_scattered, len(available_idx))
        
        if n_scattered > 0:
            replace_idx = rng.choice(available_idx, size=n_scattered, replace=False)
            
            min_vals = np.min(X_out, axis=0)
            max_vals = np.max(X_out, axis=0)
            margin = (max_vals - min_vals) * 0.2
            
            noise_matrix = rng.uniform(low=min_vals - margin, high=max_vals + margin, 
                                       size=(n_scattered, n_features))
            
            X_out[replace_idx] = noise_matrix
            all_outlier_indices.extend(replace_idx)

    # -------------------------------------------------------------------------
    # TIPO 4: OUTLIERS MULTIVARIANTES LOCALIZADOS (Halos alrededor de clústeres)
    # -------------------------------------------------------------------------
    if 'cluster_localized' in config:
        c_config = config['cluster_localized']
        prop_outliers = c_config.get('prop_outliers', 0.05)
        # Ahora el scale_factor es un empuje extra MÁS ALLÁ del borde del clúster
        scale_factor = c_config.get('scale_factor', 1.5) 
        
        n_outliers = int(n_total * prop_outliers)
        available_idx = np.setdiff1d(np.arange(n_total), all_outlier_indices)
        
        if n_outliers > 0 and len(available_idx) > 0:
            n_outliers = min(n_outliers, len(available_idx))
            rng_loc = np.random.RandomState(random_state + 4)
            idx_outliers = rng_loc.choice(available_idx, size=n_outliers, replace=False)
            
            unique_clusters = np.unique(y)
            cluster_stats = {}
            for c in unique_clusters:
                c_mask = (y == c)
                cluster_points = X_out[c_mask]
                centroid = cluster_points.mean(axis=0)
                
                # NUEVO: Calculamos el punto más lejano de este clúster (el "borde")
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                max_radius = np.max(distances)
                
                cluster_stats[c] = {
                    'centroid': centroid,
                    'max_radius': max_radius
                }
            
            for idx in idx_outliers:
                c_label = y[idx]
                stats = cluster_stats[c_label]
                
                # Dirección aleatoria
                direction = rng_loc.randn(n_features)
                direction /= np.linalg.norm(direction) 
                
                # NUEVO: Posicionamos el punto en el BORDE del clúster + un empuje extra
                # scale_factor = 1.2 significa un 20% más lejos del punto más extremo del clúster
                push_distance = stats['max_radius'] * scale_factor
                X_out[idx] = stats['centroid'] + (direction * push_distance)
                
            all_outlier_indices.extend(idx_outliers)

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


'''
def generate_simulation(
    n_samples=1000, centers=3, cluster_std=1.0, n_features=10, n_redundant=0,
    cluster_proportions=None, geometry='convex',
    anisotropy_factor=1.0, 
    feature_types=None, outlier_configs=None, grouped_outliers_config=None, 
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
   
    # 3. BASE (Modificado para geometrías no convexas)
    if geometry in ['moons', 'circles']:
        
        noise = cluster_std[0] if isinstance(cluster_std, list) else cluster_std

        if geometry == 'moons':
            # Generamos la base en 2D
            X_base, y = make_moons(
                n_samples=n_samples_input, 
                noise=noise, 
                random_state=random_state
            )
        else:
            X_base, y = make_circles(
                n_samples=n_samples_input, 
                noise=noise, 
                random_state=random_state
            )
        
        # Escalamos la base según tu separation_factor
        X_base = X_base * separation_factor * 5 

        # Rellenamos el resto de dimensiones base (n_features_base) con ruido
        # Esto evita que el clustering sea "demasiado fácil" por tener dimensiones vacías
        if n_features_base > 2:
            extra_dims = np.random.normal(0, noise, size=(X_base.shape[0], n_features_base - 2))
            X_arr = np.hstack([X_base, extra_dims])
        else:
            X_arr = X_base
    else:
        # Tu lógica original de make_blobs
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
        prop_outliers = grouped_outliers_config.get('prop_outliers', 0.05)
        n_groups = grouped_outliers_config.get('n_groups', 2)
        dist = grouped_outliers_config.get('distance', 50.0)
        
        # Calculamos el número absoluto de outliers
        n_out = int(len(X_arr) * prop_outliers)
        
        # Seguridad: por si se introduce una proporción > 1 por error
        n_out = min(n_out, len(X_arr))
        
        rng_out = np.random.RandomState(random_state + 1)
        
        # Pequeña validación por si la proporción es tan baja que da 0 outliers
        if n_out > 0:
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
        #X_redundant += rng_red.normal(scale=0.02, size=X_redundant.shape)
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
'''
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