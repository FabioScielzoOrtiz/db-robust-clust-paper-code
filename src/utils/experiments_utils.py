########################################################################################################################################################################

import time
import logging
import numpy as np
from db_robust_clust.models import SampleDistClustering, FoldSampleDistClustering
from db_robust_clust.metrics import adjusted_score
from sklearn.metrics import adjusted_rand_score
from sklearn_extra.cluster import KMedoids, CLARA
from sklearn.cluster import (KMeans, AgglomerativeClustering,
                             SpectralClustering, SpectralBiclustering, SpectralCoclustering, Birch, 
                             BisectingKMeans, MiniBatchKMeans)
from sklearn.mixture import GaussianMixture
from clustpy.partition import SubKmeans, LDAKmeans, DipInit
from clustpy.hierarchical import Diana
from func_timeout import func_timeout, FunctionTimedOut 

import os, sys
project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
from src.utils.simulations_utils import generate_simulation

########################################################################################################################################################################

def make_experiment_1(data_sizes, random_state, centers, cluster_std, n_features, outlier_configs,
                      n_clusters, metric, method, init, max_iter):

    logger = logging.getLogger(__name__)
    
    results = {}

    for n_samples in data_sizes:

        logger.info(f'⏳ Running for data size {n_samples}...')

        X, y = generate_simulation(
                random_state = random_state,
                n_samples = n_samples,
                centers=centers,
                cluster_std=cluster_std,
                n_features=n_features,
                outlier_configs = outlier_configs,  
                custom_sampling = False,  
                return_outlier_idx=False
            )
        try:
            start_time = time.time()
            model = KMedoids(
                n_clusters=n_clusters, 
                metric=metric, 
                method=method, 
                init=init, 
                max_iter=max_iter, 
                random_state=random_state)
            model.fit(X)
            end_time = time.time()
            results[n] = end_time - start_time
        except Exception as e:
            results[n] = e
    
    return results


########################################################################################################################################################################

def make_experiment_2(X, y, frac_sample_sizes, n_clusters, method, init, max_iter, random_state, 
                      p1, p2, p3, d1, d2, d3, robust_method, alpha, score_metric): 

    # Logger local para tener contexto
    logger = logging.getLogger(__name__)

    results = {
        'time': {}, 
        'adj_accuracy': {}, 
        'ARI': {}, 
    }
    
    # Log inicial informativo
    logger.info(f"🚀 Starting Experiment 1 | Seed: {random_state} | N_Sample_Sizes: {len(frac_sample_sizes)}")

    clustering_base_method = KMedoids(
                n_clusters=n_clusters, 
                metric='precomputed', 
                method=method, 
                init=init, 
                max_iter=max_iter, 
                random_state=random_state
    )
    
    for frac_sample_size in frac_sample_sizes:
        
        # 1. Configuración del modelo
        # Usamos debug para configuración detallada, info para progreso
        logger.info(f"  >> Processing frac_sample_size: {frac_sample_size:.2f}")
        
        try:
            
            sample_dist_clust = SampleDistClustering(
                clustering_method = clustering_base_method,
                metric = 'ggower',
                frac_sample_size=frac_sample_size,
                random_state=random_state,
                stratify=False,
                p1=p1, p2=p2, p3=p3,
                d1=d1, d2=d2, d3=d3, 
                robust_method=robust_method, 
                alpha=alpha
            )
            
            # 2. Medición de Tiempo y Ajuste
            start_time = time.time()
            sample_dist_clust.fit(X=X) 
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
            
                fold_fast_kmedoids = FoldSampleDistClustering(                                            
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

def make_experiment_4(data_sizes, centers, cluster_std, n_features, outlier_configs, random_state, models, score_metric):
    
    logger = logging.getLogger(__name__)

    results = {
        'time': {k: {} for k in data_sizes}, 
        'adj_accuracy': {k: {} for k in data_sizes}, 
        'ARI': {k: {} for k in data_sizes}, 
    }

    for model_name, model in models.items():
        
        logger.info(f'⚙️ Running for model {model_name}...')

        for n_samples in data_sizes:

            logger.info(f'⏳ Running for data size {n_samples}...')

            X, y = generate_simulation(
                    random_state = random_state,
                    n_samples = n_samples,
                    centers=centers,
                    cluster_std=cluster_std,
                    n_features=n_features,
                    outlier_configs = outlier_configs,  
                    custom_sampling = False,  
                    return_outlier_idx=False
                )
            
            start_time = time.time()
            model.fit(X)
            end_time = time.time()
            results['time'][n_samples][model_name] = end_time - start_time
            results['adj_accuracy'][n_samples][model_name], adj_labels = adjusted_score(y_pred=model.labels_, y_true=y, metric=score_metric)
            results['ARI'][n_samples][model_name] = adjusted_rand_score(labels_pred=adj_labels, labels_true=y)

    return results

########################################################################################################################################################################

def make_experiment_5(X, y, models, score_metric, 
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

def get_mixed_distances_names(quant_distances_names, binary_distances_names, multiclass_distances_names, robust_method):

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

def get_clustering_models_experiment_4(experiment_config, random_state):
        
    clustering_base_method = KMedoids(
                n_clusters=experiment_config['n_clusters'], 
                metric='precomputed', 
                method=experiment_config['method'], 
                init=experiment_config['init'],
                max_iter=experiment_config['max_iter'],
                random_state=experiment_config['random_state'],
    )

    models = {

        'KMedoids-euclidean': KMedoids(
            n_clusters=experiment_config['n_clusters'], 
            method=experiment_config['method'], 
            init=experiment_config['init'], 
            max_iter=experiment_config['max_iter'], 
            metric='euclidean',
            random_state = random_state
            ),

        'FastKmedoidsGGower-robust_mahalanobis_winsorized-sokal-hamming': SampleDistClustering(
                clustering_method = clustering_base_method,
                metric = 'ggower',
                frac_sample_size=experiment_config['frac_sample_size'],
                random_state=random_state,
                stratify=False,
                p1=experiment_config['p1'], 
                p2=experiment_config['p2'], 
                p3=experiment_config['p3'], 
                d1=experiment_config['d1'], 
                d2=experiment_config['d2'], 
                d3=experiment_config['d3'], 
                robust_method=experiment_config['robust_method'], 
                alpha=experiment_config['alpha'], 
            ),

        'FoldFastKmedoidsGGower-robust_mahalanobis_winsorized-sokal-hamming': FoldSampleDistClustering(
                clustering_method = clustering_base_method,
                metric = 'ggower',
                n_splits=experiment_config['n_splits'],
                shuffle=experiment_config['shuffle'],
                frac_sample_size=experiment_config['frac_sample_size'],
                meta_frac_sample_size=experiment_config['meta_frac_sample_size'],
                random_state=random_state,
                stratify=False,
                p1=experiment_config['p1'], 
                p2=experiment_config['p2'], 
                p3=experiment_config['p3'], 
                d1=experiment_config['d1'], 
                d2=experiment_config['d2'], 
                d3=experiment_config['d3'], 
                robust_method=experiment_config['robust_method'], 
                alpha=experiment_config['alpha'], 
            ) 
    }
        
    return models

########################################################################################################################################################################

def get_clustering_models_experiment_5(experiment_config, random_state, mixed_distances_names):

    clustering_base_method = KMedoids(
                n_clusters=experiment_config['n_clusters'], 
                metric='precomputed', 
                method=experiment_config['method'], 
                init=experiment_config['init'],
                max_iter=experiment_config['max_iter'],
                random_state=experiment_config['random_state'],
    )

    models = {

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

        #'SpectralClustering': SpectralClustering(
        #    n_clusters=experiment_config['n_clusters'],
        #    random_state = random_state
        #    ),

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

    for d in mixed_distances_names:

        d1, d2, d3 = d.split('-')

        if 'robust' in d1:
            r = d1.split('_')[-1]
            d1 = '_'.join(d1.split('_')[:2])
            
        models[f'FastKmedoidsGGower-{d1}_{r}-{d2}-{d3}'] = SampleDistClustering(
                clustering_method = clustering_base_method,
                metric = 'ggower',
                frac_sample_size=experiment_config['frac_sample_size'],
                random_state=random_state,
                stratify=False,
                p1=experiment_config['p1'], 
                p2=experiment_config['p2'], 
                p3=experiment_config['p3'], 
                d1=experiment_config['d1'], 
                d2=experiment_config['d2'], 
                d3=experiment_config['d3'], 
                robust_method=experiment_config['robust_method'], 
                alpha=experiment_config['alpha'], 
            )

        models[f'FoldFastKmedoidsGGower-{d1}_{r}-{d2}-{d3}'] = FoldSampleDistClustering(
                clustering_method = clustering_base_method,
                metric = 'ggower',
                n_splits=experiment_config['n_splits'],
                shuffle=experiment_config['shuffle'],
                frac_sample_size=experiment_config['frac_sample_size'],
                meta_frac_sample_size=experiment_config['meta_frac_sample_size'],
                random_state=random_state,
                stratify=False,
                p1=experiment_config['p1'], 
                p2=experiment_config['p2'], 
                p3=experiment_config['p3'], 
                d1=experiment_config['d1'], 
                d2=experiment_config['d2'], 
                d3=experiment_config['d3'], 
                robust_method=experiment_config['robust_method'], 
                alpha=experiment_config['alpha'], 
            ) 
        
    return models

########################################################################################################################################################################