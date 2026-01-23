########################################################################################################################################################################

import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from db_robust_clust.models import FastKmedoidsGGower, FoldFastKmedoidsGGower
from db_robust_clust.metrics import adjusted_score
from sklearn.metrics import accuracy_score, adjusted_rand_score
from collections import defaultdict

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
                      VG_sample_size, VG_n_samples, metric): 

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
            acc, adj_labels = adjusted_score(y_pred=fast_kmedoids.labels_, y_true=y, metric=metric)
            ari = adjusted_rand_score(labels_pred=adj_labels, labels_true=y)

            results['adj_accuracy'][frac_sample_size] = acc
            results['ARI'][frac_sample_size] = ari

            # 5. Log de Resultados Inmediato (Feedback instantáneo)
            logger.info(f"     ✅ Finished in {elapsed_time:.2f}s | ARI: {ari:.2f} | Acc: {acc:.2f}")

        except Exception as e:
            logger.error(f"     ❌ Error fitting model for frac {frac_sample_size}: {e}")
            results['time'][frac_sample_size] = np.nan
            results['adj_accuracy'][frac_sample_size] = np.nan
            results['ARI'][frac_sample_size] = np.nan

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
                      VG_sample_size, VG_n_samples, metric=accuracy_score):

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
                results['adj_accuracy'][split][frac_sample_size], adj_labels = adjusted_score(y_pred=fold_fast_kmedoids.labels_, y_true=y, metric=metric)
                results['ARI'][split][frac_sample_size] = adjusted_rand_score(labels_pred=adj_labels, labels_true=y)           
            except Exception as e:
                print('Exception:', e)

    return results


########################################################################################################################################################################

def make_experiment_4(X, y, models, metric=accuracy_score):
    
    model_names = list(models.keys())

    results = {
        'time': {k: {} for k in model_names}, 
        'adj_accuracy': {k: {} for k in model_names}, 
        'ARI': {k: {} for k in model_names},
        'labels': {k: {} for k in model_names},
        'adj_labels': {k: {} for k in model_names}
    }

    X_np = X.to_numpy()

    for model_name, model in models.items():
        print(model_name)

        start_time = time.time()
        if model_name in ['SubKmeans', 'DipInit']:
            model.fit(X_np)
        else:
            model.fit(X)
        end_time = time.time()
        
        results['time'][model_name] = end_time - start_time
        if model_name == 'GaussianMixture':
            results['labels'][model_name] = model.predict(X)
        elif 'Spectral' in model_name and model_name != 'SpectralClustering':
            results['labels'][model_name] = model.row_labels_
        else:
            results['labels'][model_name] = model.labels_
        print('len y_pred:', len(np.unique(results['labels'][model_name])))
        results['adj_accuracy'][model_name], results['adj_labels'][model_name] = adjusted_score(y_pred=results['labels'][model_name] , y_true=y, metric=metric)
        results['ARI'][model_name] = adjusted_rand_score(labels_pred=results['adj_labels'][model_name], labels_true=y)


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

def get_GGower_distances_names(quant_distances_names, binary_distances_names, multiclass_distances_names, robust_method):

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

def plot_experiment_2_results(
        best_frac, best_acc, best_ari, best_time, 
        x_data_pct, y_acc, y_ari, y_time,
        data_name, num_realizations, save_path
    ):

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