########################################################################################################################################################################

import time
import numpy as np
import pandas as pd
from FastKmedoids.models import FastKmedoidsGGower, FoldFastKmedoidsGGower
from FastKmedoids.metrics import adjusted_accuracy
from sklearn.metrics import adjusted_rand_score
from simulations_utils import get_simulation_1
from collections import defaultdict
def nested_dict():
    return defaultdict(nested_dict)

########################################################################################################################################################################

def make_experiment_1(X, y, frac_sample_sizes, n_clusters, method, init, max_iter, random_state, 
                      p1, p2, p3, d1, d2, d3, robust_method, alpha, epsilon, n_iters, 
                      VG_sample_size, VG_n_samples):

    results = {
        'time': {}, 
        'adj_accuracy': {}, 
    }
    
    for frac_size in frac_sample_sizes:
        print('frac_size:', frac_size)
        
        try:

            fast_kmedoids = FastKmedoidsGGower(
                n_clusters=n_clusters, 
                method=method, 
                init=init, 
                max_iter=max_iter, 
                random_state=random_state,
                frac_sample_size=frac_size, 
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
                VG_n_samples=VG_n_samples
            )
            
            start_time = time.time()
            fast_kmedoids.fit(X=X) 
            end_time = time.time()
            results['time'][frac_size] = end_time - start_time
            results['adj_accuracy'][frac_size], _ = adjusted_accuracy(y_pred=fast_kmedoids.labels_, y_true=y)
        
        except Exception as e:
            print(e)

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

def make_experiment_2(n_sample_list, models, random_state):
    
    results = {
        'time': {k: {} for k in n_sample_list}, 
        'adj_accuracy': {k: {} for k in n_sample_list}, 
    }

    for model_name, model in models.items():
        print(model_name)

        for n_samples in n_sample_list:
            print(n_samples)
            
            X, y = get_simulation_1(n_samples=n_samples, random_state=random_state)

            try:
                start_time = time.time()
                model.fit(X)
                end_time = time.time()
                results['time'][n_samples][model_name] = end_time - start_time
                results['adj_accuracy'][n_samples][model_name], _ = adjusted_accuracy(y_pred=model.labels_ , y_true=y)
            except Exception as e:
                print(f'Exception: {e}')

    return results

########################################################################################################################################################################

def make_experiment_3(X, y, n_splits, frac_sample_sizes, n_clusters, method, init, max_iter, random_state, 
                      p1, p2, p3, d1, d2, d3, robust_method, alpha, epsilon, n_iters, shuffle, kfold_random_state,
                      VG_sample_size, VG_n_samples):

    results = {
        'time': {k: {} for k in n_splits},
        'adj_accuracy': {k: {} for k in n_splits}
    }
           
    for split in n_splits:
        print('n_splits:', split)
        for frac_size in frac_sample_sizes:
            print('frac_size:', frac_size)

            try:
            
                fold_fast_kmedoids = FoldFastKmedoidsGGower(                                            
                    n_clusters=n_clusters, 
                    method=method, 
                    init=init, 
                    max_iter=max_iter, 
                    random_state=random_state,
                    frac_sample_size=frac_size, 
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
                results['time'][split][frac_size] = end_time - start_time
                results['adj_accuracy'][split][frac_size], _ = adjusted_accuracy(y_pred=fold_fast_kmedoids.labels_, y_true=y)
            
            except Exception as e:
                print('Exception:', e)

    return results

########################################################################################################################################################################

def make_experiment_4(X, y, models):
    
    model_names = list(models.keys())

    results = {
        'time': {k: {} for k in model_names}, 
        'adj_accuracy': {k: {} for k in model_names}, 
        'ARI': {k: {} for k in model_names},
        'labels': {k: {} for k in model_names},
        'adj_labels': {k: {} for k in model_names},
        'not_feasible_models': []
    }

    for model_name, model in models.items():
        print(model_name)
        try:
            start_time = time.time()
            model.fit(X)
            end_time = time.time()
            results['time'][model_name] = end_time - start_time
            results['labels'][model_name] = model.labels_
            results['adj_accuracy'][model_name], results['adj_labels'][model_name] = adjusted_accuracy(y_pred=results['labels'][model_name] , y_true=y)
            results['ARI'][model_name] = adjusted_rand_score(labels_pred=results['adj_labels'][model_name], labels_true=y)
        except Exception as e:
            print(f'Exception: {e}')
            results['not_feasible_models'].append(model_name)

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
