########################################################################################################################################################################

import time
import logging
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines 
import seaborn as sns
from db_robust_clust.models import FastKmedoidsGGower, FoldFastKmedoidsGGower
from db_robust_clust.metrics import adjusted_score
from sklearn.metrics import adjusted_rand_score
from collections import defaultdict
from db_robust_clust.models import FastKmedoidsGGower, FoldFastKmedoidsGGower
from sklearn_extra.cluster import KMedoids, CLARA
from sklearn.cluster import (KMeans, AgglomerativeClustering,
                             SpectralClustering, SpectralBiclustering, SpectralCoclustering, Birch, 
                             BisectingKMeans, MiniBatchKMeans)
from sklearn.mixture import GaussianMixture
from clustpy.partition import SubKmeans, LDAKmeans, DipInit
from clustpy.hierarchical import Diana
from func_timeout import func_timeout, FunctionTimedOut 

########################################################################################################################################################################

'''
# TODO
def make_experiment_1():

    sample_sizes = [10000, 20000, 30000, 40000, 50000, 60000, 75000]

    times = {}

    for n in sample_sizes:
        print(n)

        X, Y = make_blobs(n_samples=n, centers=3, cluster_std=[2,2,3], n_features=8, random_state=123)
        X = pd.DataFrame(X)      
        X.columns = [f"X{i}" for i in range(1, X.shape[1]+1)]

        # Se convierten dos variables cuantitativas a binarias, y otras dos a multiclase, discretizandolas.
        X['X5'] = pd.cut(X['X5'], bins=[X['X5'].min()-1, X['X5'].mean(), X['X5'].max()+1], labels=False)
        X['X6'] = pd.cut(X['X6'], bins=[X['X6'].min()-1, X['X6'].mean(), X['X6'].max()+1], labels=False)
        X['X7'] = pd.cut(X['X7'], bins=[X['X7'].min()-1, X['X7'].quantile(0.25), X['X7'].quantile(0.50), X['X7'].quantile(0.75), X['X7'].max()+1], labels=False)
        X['X8'] = pd.cut(X['X8'], bins=[X['X8'].min()-1, X['X8'].quantile(0.25), X['X8'].quantile(0.50), X['X8'].quantile(0.75), X['X8'].max()+1], labels=False)   

        try:
            start_time = time.time()
            D_euclidean = Euclidean_dist_matrix(X)
            kmedoids = KMedoids(n_clusters=3, metric='precomputed', method='pam', init='heuristic', max_iter=150, random_state=123)
            kmedoids.fit(D_euclidean)
            end_time = time.time()
            times[n] = end_time - start_time
        except:
            times[n] = 'not feasible'

    with open(r'../../results/kmedoids_slow/kmedoids_slow_times.pkl', 'wb') as file:
        pickle.dump(times, file)   
'''

########################################################################################################################################################################

def make_experiment_2(X, y, frac_sample_sizes, n_clusters, method, init, max_iter, random_state, 
                      p1, p2, p3, d1, d2, d3, robust_method, alpha, epsilon, n_iters, 
                      VG_sample_size, VG_n_samples, score_metric): 

    # Logger local para tener contexto
    logger = logging.getLogger(__name__)

    results = {
        'time': {}, 
        'adj_accuracy': {}, 
        'ARI': {}, 
    }
    
    # Log inicial informativo
    logger.info(f"🚀 Starting Experiment 1 | Seed: {random_state} | N_Sample_Sizes: {len(frac_sample_sizes)}")

    for frac_sample_size in frac_sample_sizes:
        
        # 1. Configuración del modelo
        # Usamos debug para configuración detallada, info para progreso
        logger.info(f"  >> Processing frac_sample_size: {frac_sample_size:.2f}")
        
        try:
            fast_kmedoids = FastKmedoidsGGower(
                n_clusters=n_clusters, 
                method=method, 
                init=init, 
                max_iter=max_iter, 
                random_state=random_state,
                frac_sample_size=frac_sample_size, 
                p1=p1, p2=p2, p3=p3, 
                d1=d1, d2=d2, d3=d3, 
                robust_method=robust_method, 
                alpha=alpha, 
                epsilon=epsilon, 
                n_iters=n_iters, 
                VG_sample_size=VG_sample_size, 
                VG_n_samples=VG_n_samples
            )
            
            # 2. Medición de Tiempo y Ajuste
            start_time = time.time()
            fast_kmedoids.fit(X=X) 
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            results['time'][frac_sample_size] = elapsed_time

            # 3. Validación de Clusters (Sanity Check)
            unique_labels = np.unique(fast_kmedoids.labels_)
            n_found_clusters = len(unique_labels)
            
            if n_found_clusters != n_clusters:
                logger.warning(f"     ⚠️ Cluster Mismatch: Expected {n_clusters}, found {n_found_clusters} unique labels.")

            # 4. Cálculo de Métricas
            acc, adj_labels = adjusted_score(y_pred=fast_kmedoids.labels_, y_true=y, metric=score_metric)
            ari = adjusted_rand_score(labels_pred=adj_labels, labels_true=y)

            results['adj_accuracy'][frac_sample_size] = acc
            results['ARI'][frac_sample_size] = ari

            # 5. Log de Resultados Inmediato (Feedback instantáneo)
            logger.info(f"     ✅ Finished in {elapsed_time:.2f}s | ARI: {ari:.2f} | Acc: {acc:.2f}")

        except Exception as e:
            logger.error(f"     ❌ Error fitting model for frac {frac_sample_size}: {e}")
            for k in results.keys():
                results[k][frac_sample_size] = None
            results['status'][frac_sample_size] = f"Error: {str(e)}"

    return results

########################################################################################################################################################################

def get_pivoted_results(results, iterable):
    
    i = list(results.keys())[0] # any key of results would be valid
    pivoted_results = {k1: {k2: {} for k2 in iterable} for k1 in results[i].keys()}
    for k1 in results[i].keys():
        for k2 in iterable:
            pivoted_results[k1][k2] = [results[k][k1][k2] for k in results.keys()]    

    return pivoted_results

########################################################################################################################################################################

def get_avg_results(results, pivoted_results, iterable):
    
    i = list(results.keys())[0] # any key of results would be valid

    avg_results = {k: {k: {} for k in iterable} for k in results[i].keys()}

    for k1 in avg_results.keys():
        for k2 in iterable:
            if isinstance(pivoted_results[k1][k2][0], np.ndarray):
                avg_results[k1][k2] = pivoted_results[k1][k2]
            else:
                avg_results[k1][k2] = np.mean(pivoted_results[k1][k2])

    return avg_results

########################################################################################################################################################################

def make_experiment_3(X, y, n_splits, frac_sample_sizes, n_clusters, method, init, max_iter, random_state, 
                      p1, p2, p3, d1, d2, d3, robust_method, alpha, epsilon, n_iters, shuffle, kfold_random_state,
                      VG_sample_size, VG_n_samples, score_metric):

    results = {
        'time': {k: {} for k in n_splits},
        'adj_accuracy': {k: {} for k in n_splits},
        'ARI': {k: {} for k in n_splits}
    }
           
    for split in n_splits:
        print('n_splits:', split)
        for frac_sample_size in frac_sample_sizes:
            print('frac_sample_size:', frac_sample_size)

            try:
            
                fold_fast_kmedoids = FoldFastKmedoidsGGower(                                            
                    n_clusters=n_clusters, 
                    method=method, 
                    init=init, 
                    max_iter=max_iter, 
                    random_state=random_state,
                    frac_sample_size=frac_sample_size, 
                    p1=p1, 
                    p2=p2, 
                    p3=p3, 
                    d1=d1, 
                    d2=d2, 
                    d3=d3, 
                    robust_method=robust_method, 
                    alpha=alpha, 
                    epsilon=epsilon, 
                    n_iters=n_iters, 
                    VG_sample_size=VG_sample_size, 
                    VG_n_samples=VG_n_samples,
                    n_splits=split, 
                    shuffle=shuffle, 
                    kfold_random_state=kfold_random_state,
                )
                
                start_time = time.time()
                fold_fast_kmedoids.fit(X=X) 
                end_time = time.time()
                results['time'][split][frac_sample_size] = end_time - start_time
                results['adj_accuracy'][split][frac_sample_size], adj_labels = adjusted_score(y_pred=fold_fast_kmedoids.labels_, y_true=y, metric=score_metric)
                results['ARI'][split][frac_sample_size] = adjusted_rand_score(labels_pred=adj_labels, labels_true=y)           
            
            except Exception as e:
                logger.error(f"     ❌ Error fitting model for split {split} and frac {frac_sample_size}: {e}")
                for k in results.keys():
                    results[k][split][frac_sample_size] = None
                results['status'][split][frac_sample_size] = f"Error: {str(e)}"

    return results


########################################################################################################################################################################

import time
import numpy as np

def make_experiment_4(X, y, models, score_metric, 
                      max_duration_mins=10 
                      ):  
    
    model_names = list(models.keys())

    results = {
        'time': {k: {} for k in model_names}, 
        'adj_accuracy': {k: {} for k in model_names}, 
        'ARI': {k: {} for k in model_names},
        'labels': {k: {} for k in model_names},
        'adj_labels': {k: {} for k in model_names},
        'status': {k: {} for k in model_names}
    }
    
    if not isinstance(X, np.ndarray):
        X = X.to_numpy()

    for model_name, model in models.items():
        print(f'Running model: {model_name}')
        
        try:
            start_time = time.time()
            
            # --- CAMBIO PRINCIPAL ---
            # Ejecutamos fit con un límite de tiempo.
            # Si tarda más de 'max_duration', lanza una excepción 'FunctionTimedOut'
            max_duration_secs = max_duration_mins * 60
            func_timeout(max_duration_secs, model.fit, args=(X,))
            # ------------------------

            end_time = time.time()

            results['time'][model_name] = end_time - start_time
            
            # Lógica de asignación de labels
            if model_name == 'GaussianMixture':
                results['labels'][model_name] = model.predict(X)
            elif 'Spectral' in model_name and model_name != 'SpectralClustering':
                results['labels'][model_name] = model.row_labels_
            else:
                results['labels'][model_name] = model.labels_
            
            results['adj_accuracy'][model_name], results['adj_labels'][model_name] = adjusted_score(y_pred=results['labels'][model_name] , y_true=y, metric=score_metric)
            results['ARI'][model_name] = adjusted_rand_score(labels_pred=results['adj_labels'][model_name], labels_true=y)

            results['status'][model_name] = 'OK'
            
        except FunctionTimedOut:
            # Capturamos específicamente el Timeout para imprimir un mensaje claro
            print(f"   ⏳ [TIMEOUT] {model_name} excedió el límite de {max_duration_mins} minutos.")
            # Forzamos que se ejecute la lógica de rellenar con None
            for k in results.keys():
                results[k][model_name] = None
          
            results['status'][model_name] = f"Error: Timeout {round(max_duration_mins, 2)} mins"
        
        except Exception as e:
            # Captura cualquier otro error (memoria, convergencia, etc.)
            print(f"   ❌ [ERROR] {model_name}: {e}")
            for k in results.keys():
                results[k][model_name] = None
           
            results['status'][model_name] = f"Error: {str(e)}"

    return results

########################################################################################################################################################################

def get_pivoted_results_two_iterables(results, iterable1, iterable2):
    """
    Reorganiza los resultados experimentales en un diccionario anidado para dos variables independientes.

    Parameters
    ----------
    results : dict
        Diccionario de resultados indexado por claves experimentales (e.g. repeticiones).
        Cada valor es un diccionario de métricas que a su vez contiene resultados por (var1, var2).
    iterable1 : iterable
        Primer conjunto de valores de configuración (por ejemplo, distintos valores de p1).
    iterable2 : iterable
        Segundo conjunto de valores de configuración (por ejemplo, distintos valores de d1).

    Returns
    -------
    dict
        Diccionario con estructura {métrica: {var1: {var2: [valores]}}}
    """
    i = list(results.keys())[0]  # cualquier clave vale
    pivoted_results = {
        k1: {
            v1: {v2: [] for v2 in iterable2} for v1 in iterable1
        } for k1 in results[i].keys()
    }

    for k in results.keys():  # cada repetición o experimento
        for k1 in results[k].keys():  # cada métrica
            for v1 in iterable1:
                for v2 in iterable2:
                    pivoted_results[k1][v1][v2].append(results[k][k1][v1][v2])

    return pivoted_results

########################################################################################################################################################################

def get_avg_results_two_iterables(results, pivoted_results, iterable1, iterable2):
    """
    Calcula el promedio de los resultados reorganizados por dos variables independientes.

    Parameters
    ----------
    results : dict
        Diccionario original de resultados por repetición.
    pivoted_results : dict
        Diccionario generado por get_pivoted_results_two_iterables.
    iterable1 : iterable
        Primer conjunto de valores de configuración.
    iterable2 : iterable
        Segundo conjunto de valores de configuración.

    Returns
    -------
    dict
        Diccionario con estructura {métrica: {var1: {var2: promedio}}}
    """
    i = list(results.keys())[0]
    avg_results = {
        k: {
            v1: {v2: None for v2 in iterable2} for v1 in iterable1
        } for k in results[i].keys()
    }

    for k1 in avg_results.keys():
        for v1 in iterable1:
            for v2 in iterable2:
                avg_results[k1][v1][v2] = np.mean(pivoted_results[k1][v1][v2])

    return avg_results

########################################################################################################################################################################

def avg_results_to_dfs(avg_results, column_1, column_2):
    dfs = {}
    for key, subdict in avg_results.items():
        rows = [(k1, k2, v) for k1, inner in subdict.items() for k2, v in inner.items()]
        dfs[key] = pd.DataFrame(rows, columns=[column_1, column_2, key])
    return dfs

########################################################################################################################################################################

def get_ggower_distances_names(quant_distances_names, binary_distances_names, multiclass_distances_names, robust_method):

    combinations_names = []
    for d1 in quant_distances_names:
        for d2 in binary_distances_names:
            for d3 in multiclass_distances_names:
                if 'robust' not in d1:
                    combinations_names.append(f'{d1}-{d2}-{d3}') 
                else:
                    for r in robust_method:
                        combinations_names.append(f'{d1}_{r}-{d2}-{d3}')

    return combinations_names

########################################################################################################################################################################

def split_list_in_chunks(list, chunk_size):
    return [list[i : i + chunk_size] for i in range(0, len(list), chunk_size)]

########################################################################################################################################################################

'''
# TODO
def plot_experiment_2_results():
    
    times_values = np.array(list(times.values()))
    times_values = times_values / 60 # to minutes
    fig, ax = plt.subplots(figsize=(7,4))
    ax = sns.lineplot(x=sample_sizes, y=times_values, color='blue', marker='o', markersize=6)
    ax.set_ylabel('Time (min)', size=11)
    ax.set_xlabel('Data Size', size=11)
    plt.xticks(fontsize=10, rotation=0)
    plt.yticks(fontsize=10)
    ax.set_xticks(sample_sizes)
    ax.set_yticks(np.round(np.linspace(np.min(times_values), np.max(times_values), 9), 0))
    plt.title("Euclidean $k$-Medoids - Time vs Data Size - Simulated Data", fontsize=12.5, weight='bold')
    plt.tight_layout()
    plt.show()

    file_name = '../../results/kmedoids_slow/kmedoids_slow_times'
    fig.savefig(file_name + '.jpg', format='jpg', dpi=500)
'''

########################################################################################################################################################################

def get_clustering_models(experiment_config, random_state, ggower_distances_names):

        models= {

            'KMeans': KMeans(
                n_clusters=experiment_config['n_clusters'],
                max_iter=experiment_config['max_iter'],
                init='k-means++', 
                n_init='auto',
                random_state = random_state,
                ),

            'CLARA': CLARA(
                n_clusters=experiment_config['n_clusters'], 
                metric='euclidean',
                #random_state = random_state # has not random_state parameter
                ),

            'Diana': Diana(
                n_clusters=experiment_config['n_clusters'],
                #random_state = random_state # has not random_state parameter
                ),

            'LDAKmeans': LDAKmeans(
                n_clusters=experiment_config['n_clusters'], 
                #random_state = random_state # has not random_state parameter
                ),

            'SubKmeans': SubKmeans(
                n_clusters=experiment_config['n_clusters'],
                #random_state = random_state # has not random_state parameter
                ),

            'DipInit': DipInit(
                n_clusters=experiment_config['n_clusters'],
                random_state = random_state
                ),

            'GaussianMixture': GaussianMixture(
                n_components=experiment_config['n_clusters'],
                random_state = random_state
                ),

            'AgglomerativeClustering': AgglomerativeClustering(
                n_clusters=experiment_config['n_clusters'],
                #random_state = random_state # has not random_state parameter
                ),

            'SpectralClustering': SpectralClustering(
                n_clusters=experiment_config['n_clusters'],
                random_state = random_state
                ),

            'SpectralBiclustering': SpectralBiclustering(
                n_clusters=experiment_config['n_clusters'],
                random_state = random_state
                ),

            'SpectralCoclustering': SpectralCoclustering(
                n_clusters=experiment_config['n_clusters'],
                random_state = random_state
                ),

            'Birch': Birch(
                n_clusters=experiment_config['n_clusters'],
                #random_state = random_state # has not random_state parameter
                ),

            'BisectingKMeans': BisectingKMeans(
                n_clusters=experiment_config['n_clusters'],
                random_state = random_state
                ),

            'MiniBatchKMeans': MiniBatchKMeans(
                n_clusters=experiment_config['n_clusters'],
                random_state = random_state
                ),
            
            'KMedoids-euclidean': KMedoids(
                n_clusters=experiment_config['n_clusters'], 
                method=experiment_config['method'], 
                init=experiment_config['init'], 
                max_iter=experiment_config['max_iter'], 
                metric='euclidean',
                random_state = random_state
                ),
            
        }

        for d in ggower_distances_names:

            d1, d2, d3 = d.split('-')

            if 'robust' in d1:
                r = d1.split('_')[-1]
                d1 = '_'.join(d1.split('_')[:2])
                
            models[f'FastKmedoidsGGower-{d1}_{r}-{d2}-{d3}'] = FastKmedoidsGGower(
                    n_clusters=experiment_config['n_clusters'], 
                    method=experiment_config['method'], 
                    init=experiment_config['init'], 
                    max_iter=experiment_config['max_iter'], 
                    frac_sample_size=experiment_config['frac_sample_size'], 
                    p1=experiment_config['p1'], 
                    p2=experiment_config['p2'], 
                    p3=experiment_config['p3'], 
                    d1=d1, 
                    d2=d2, 
                    d3=d3, 
                    robust_method=r, 
                    alpha=experiment_config['alpha'], 
                    epsilon=experiment_config['epsilon'], 
                    n_iters=experiment_config['n_iters'],
                    VG_sample_size=experiment_config['VG_sample_size'], 
                    VG_n_samples=experiment_config['VG_n_samples'],
                    random_state = random_state
                ) 

            models[f'FoldFastKmedoidsGGower-{d1}_{r}-{d2}-{d3}'] = FoldFastKmedoidsGGower(
                    n_clusters=experiment_config['n_clusters'], 
                    method=experiment_config['method'], 
                    init=experiment_config['init'], 
                    max_iter=experiment_config['max_iter'], 
                    frac_sample_size=experiment_config['frac_sample_size'], 
                    p1=experiment_config['p1'], 
                    p2=experiment_config['p2'], 
                    p3=experiment_config['p3'], 
                    d1=d1, 
                    d2=d2, 
                    d3=d3, 
                    robust_method=r, 
                    alpha=experiment_config['alpha'], 
                    epsilon=experiment_config['epsilon'], 
                    n_iters=experiment_config['n_iters'],
                    VG_sample_size=experiment_config['VG_sample_size'], 
                    VG_n_samples=experiment_config['VG_n_samples'],
                    n_splits=experiment_config['n_splits'], 
                    shuffle=experiment_config['shuffle'], 
                    kfold_random_state=experiment_config['kfold_random_state'],
                    random_state = random_state
                ) 
            
        return models

########################################################################################################################################################################

def plot_experiment_1_results():
    pass

########################################################################################################################################################################

def plot_experiment_2_results(df, data_name, num_realizations, save_path):

    best_row = df.sort("adj_accuracy", descending=True).row(0, named=True)

    best_frac = best_row["frac_sample_size"]
    best_acc = best_row["adj_accuracy"]
    best_ari = best_row["ARI"]
    best_time = best_row["time"]

    # Preparamos los datos para ejes X (convertidos a porcentaje) e Y
    x_data_pct = df["frac_sample_size"] * 100
    y_acc = df["adj_accuracy"]
    y_ari = df["ARI"]
    y_time = df["time"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes = axes.flatten()

    # --- SUBPLOT 1: Adjusted Accuracy ---
    sns.lineplot(
        x=[best_frac * 100], y=[best_acc], 
        color='red', marker='o', markersize=10, ax=axes[0], 
        #label='Best Acc.'
    )
    sns.lineplot(
        x=x_data_pct, y=y_acc, 
        color='blue', marker='o', markersize=5, ax=axes[0]
    )
    axes[0].set_title('Adj. Accuracy vs. Sample Size', size=12, weight='bold')
    axes[0].set_ylabel('Adj. Accuracy', size=11)
    #axes[0].legend(fontsize=9) # Opcional: para mostrar la label del punto rojo

    # --- SUBPLOT 2: ARI ---
    sns.lineplot(
        x=[best_frac * 100], y=[best_ari], 
        color='red', marker='o', markersize=10, ax=axes[1]
    )
    sns.lineplot(
        x=x_data_pct, y=y_ari, 
        color='blue', marker='o', markersize=5, ax=axes[1]
    )
    axes[1].set_title('ARI vs. Sample Size', size=12, weight='bold')
    axes[1].set_ylabel('ARI', size=11)

    # --- SUBPLOT 3: Time ---
    sns.lineplot(
        x=[best_frac * 100], y=[best_time], 
        color='red', marker='o', markersize=10, ax=axes[2]
    )
    sns.lineplot(
        x=x_data_pct, y=y_time, 
        color='blue', marker='o', markersize=5, ax=axes[2]
    )
    axes[2].set_title('Time vs. Sample Size', size=12, weight='bold')
    axes[2].set_ylabel('Time (secs)', size=11)

    # --- CONFIGURACIÓN COMÚN Y GUARDADO ---

    for ax in axes:
        ax.set_xlabel('Sample Size Parameter (%)', size=11)
        # Grid opcional para mejor legibilidad
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.subplots_adjust(top=0.83, hspace=0.5, wspace=0.23)

    # Título global dinámico (usando variables si las tienes)
    plt.suptitle(
        f'Accuracy, ARI and Time vs. Sample Size Parameter\n{data_name.replace('_', ' ').capitalize()} - Realizations: {num_realizations}', 
        fontsize=13, y=1.02, weight='bold', color='black', alpha=1
    )

    # Guardado
    fig.savefig(save_path, format='png', dpi=300, bbox_inches="tight", pad_inches=0.2)

    plt.show()

########################################################################################################################################################################

def plot_experiment_3_results(df, data_name, num_realizations, save_path):
    """
    Genera gráficos usando solo Polars.
    El punto rojo en TODOS los gráficos corresponde a la configuración 
    que obtuvo la mejor 'adj_accuracy'.
    """

    # 1. Obtener la MEJOR combinación basada SOLO en Accuracy
    # Ordenamos descendente por accuracy y tomamos la primera fila
    best_row = df.sort("adj_accuracy", descending=True).row(0, named=True)

    # Extraemos los valores de esa fila ganadora
    best_frac_pct = best_row['frac_sample_size'] * 100
    
    # Estos son los valores Y del punto rojo para cada gráfica
    best_values = {
        'adj_accuracy': best_row['adj_accuracy'],
        'ARI': best_row['ARI'],
        'time': best_row['time']
    }

    # 2. Configuración de Ejes y Métricas
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharex=True)
    
    metrics_config = [
        (0, 'adj_accuracy', 'Adj. Accuracy'),
        (1, 'ARI', 'ARI'),
        (2, 'time', 'Time (secs)')
    ]

    # 3. Obtener lista de folds únicos (usando sintaxis Polars)
    # unique() devuelve una Serie, sort() la ordena
    folds = df["n_splits"].unique().sort()

    # --- BUCLE DE PLOTEO ---
    for ax_idx, col_name, title_prefix in metrics_config:
        ax = axes[ax_idx]
        
        # A. Iterar sobre cada Fold para pintar las líneas azules
        for k in folds:
            # Filtrado estilo Polars
            subset = df.filter(pl.col("n_splits") == k)
            
            # Es importante ordenar por X para que la línea se dibuje bien
            subset = subset.sort("frac_sample_size")
            
            # Plotting: Polars requiere .to_list() para pasar los datos a matplotlib
            ax.plot(
                (subset['frac_sample_size'] * 100).to_list(), 
                subset[col_name].to_list(), 
                marker='o', markersize=5, 
                label=f"{k}-Fold"
            )

        # B. Pintar el PUNTO ROJO (Basado en la mejor Accuracy)
        # Usamos el valor correspondiente de 'best_values' para el eje Y actual
        ax.plot(
            [best_frac_pct],           # X: El porcentaje de la mejor fila
            [best_values[col_name]],   # Y: El valor de la métrica actual de la mejor fila
            marker='o', markersize=7, color='red', 
            zorder=10, label='Best Acc.' if ax_idx == 0 else ""
        )

        # C. Estética
        ax.set_title(f"{title_prefix} vs\nNumber of Folds and Sample Size", fontsize=11, fontweight='bold')
        ax.set_xlabel("Sample Size Parameter (%)", size=11)
        ax.set_ylabel(title_prefix, size=11)
        ax.grid(True, linestyle='--', alpha=0.5)

    # --- FINALIZACIÓN ---
    fig.suptitle(
        f"Results (Highlighted: Best Accuracy Configuration)\n{data_name.replace('_', ' ').capitalize()} - Realizations: {num_realizations}", 
        fontsize=13, fontweight='bold', y=0.98
    )

    # Leyenda (tomada del primer gráfico)
    handles, labels = axes[0].get_legend_handles_labels()
    # Filtramos duplicados en la leyenda por si acaso
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=len(folds)+1, fontsize=10, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.08, 1, 0.93])

    fig.savefig(save_path, format='png', dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.show()
    
########################################################################################################################################################################

def plot_experiment_4_results(df, data_name, num_realizations, save_path, 
                              our_methods_1=None, our_methods_2=None, 
                              other_methods=None, not_feasible_methods=None):
    """
    Genera gráficos de barras comparando modelos (Accuracy, ARI, Time).
    Requiere un DataFrame de Polars con columnas: 'model', 'adj_accuracy', 'ARI', 'time'.
    Calcula internamente medias y desviaciones estándar.
    """
    
    # Inicializar listas vacías si no se pasan
    our_methods_1 = our_methods_1 or []
    our_methods_2 = our_methods_2 or []
    other_methods = other_methods or []
    not_feasible_methods = not_feasible_methods or []

    # 2. Extraer vectores para plotear (convertir a numpy/list)
    model_names = df["model_name"].to_numpy()
    
    # Medias
    avg_adj_accuracy = df["mean_adj_accuracy"].fill_null(0).to_numpy()
    avg_ari = df["mean_ari"].fill_null(0).to_numpy()
    avg_time = df["mean_time"].fill_null(0).to_numpy()
    
    # Desviaciones 
    std_adj_acc = df["std_adj_accuracy"].fill_null(0).to_numpy()
    std_ari = df["std_ari"].fill_null(0).to_numpy()
    std_time = df["std_time"].fill_null(0).to_numpy()

    # Posiciones en el eje y acorde a las posiciones dadas en df
    y_pos = np.arange(len(model_names))

    # 3. Configuración de Subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 11))
    axes = axes.flatten()

    # Definir métricas para iterar
    metrics = [
        (0, avg_adj_accuracy, std_adj_acc, 'Adj. Accuracy'),
        (1, avg_ari, std_ari, 'Adj. Rand Index'),
        (2, avg_time, std_time, 'Time (secs)')
    ]

    # --- BUCLE DE PLOTEO ---
    for ax_idx, means, stds, xlabel in metrics:
        ax = axes[ax_idx]
        
        # A. Barras principales (Azules)
        sns.barplot(x=means, y=model_names, color='blue', width=0.5, alpha=0.9, ax=ax)
        
        # B. Resaltar la mejor barra (Roja - Índice 0)
        # Nota: Como el DF ya está ordenado por Acc, el primero siempre es el mejor en Acc.
        # Si quieres que en Time el rojo sea el más rápido, habría que cambiar la lógica, 
        # pero sigo tu código original donde el rojo es el mismo modelo (el mejor en Acc).
        sns.barplot(x=[means[0]], y=[model_names[0]], color='red', width=0.5, alpha=0.9, ax=ax)

        # C. Barras de error
        ax.errorbar(
            x=means,
            y=y_pos,
            xerr=stds,
            fmt='none',
            ecolor='black',
            elinewidth=1,
            capsize=3.5,
            alpha=1
        )
        
        # D. Etiquetas y Títulos
        ax.set_xlabel(xlabel, size=14)
        ax.tick_params(axis='x', labelsize=12)
        
        if ax_idx == 0:
            ax.set_ylabel('Clustering Methods', size=14)
            ax.set_title(f'Clustering Methods vs.\n{xlabel}', size=13, weight='bold')
            ax.tick_params(axis='y', labelsize=12)
        else:
            ax.set_title(f'Clustering Methods vs.\nARI' if ax_idx==1 else 'Clustering Methods vs.\nTime', size=13, weight='bold')
            ax.set_yticklabels([]) # Ocultar nombres en gráficas 2 y 3 para limpieza

    # 4. Colorear etiquetas del Eje Y (Solo en el primer gráfico)
    # Recorremos las labels generadas y cambiamos color según el grupo
    for label in axes[0].get_yticklabels():
        label.set_weight('bold')
        txt = label.get_text()
        if txt in our_methods_1:
            label.set_color('darkviolet') 
        elif txt in our_methods_2:
            label.set_color('green') 
        elif txt in other_methods:
            label.set_color('black') 
        elif txt in not_feasible_methods:
            label.set_color('red') 

    # 5. Crear Leyenda Personalizada
    legend_elements = [
        mlines.Line2D([0], [0], color='darkviolet', lw=6, label='Fast $k$-Medoids'),
        mlines.Line2D([0], [0], color='green', lw=6, label='$q$-Fold Fast $k$-Medoids'),
        mlines.Line2D([0], [0], color='black', lw=6, label='Other clustering methods'),
        mlines.Line2D([0], [0], color='red', lw=6, label='Not feasible clustering methods')
    ]

    # Añadir leyenda al primer gráfico (movida hacia la derecha para centrar globalmente abajo)
    # Ajuste del bbox_to_anchor para que quede centrada respecto a la figura global
    axes[0].legend(
        handles=legend_elements, 
        loc='upper center', 
        bbox_to_anchor=(1.7, -0.12), # Ajusta esto si se descuadra
        ncol=len(legend_elements), 
        fontsize=12,
        frameon=False
    )

    # 6. Título Global y Guardado
    plt.suptitle(
        f"Clustering Model Comparison \n{data_name.replace('_', ' ').capitalize()} - Realizations: {num_realizations}", 
        fontsize=16, fontweight='bold', y=0.98
    )

    # Ajuste final
    # plt.tight_layout() # A veces pelea con suptitle y legends externos, cuidado.
    
    fig.savefig(save_path, format='png', dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.show()

########################################################################################################################################################################