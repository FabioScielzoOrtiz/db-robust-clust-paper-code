########################################################################################################################################################################

import os
import pickle
import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines 
from sklearn.manifold import MDS
from sklearn.utils import check_array
from robust_mixed_dist.mixed import generalized_gower_dist_matrix

########################################################################################################################################################################

def process_experiment_2_results(results_path, prop_errors_threshold):

    if not os.path.exists(results_path):
        print("❌ Error: El archivo no existe. Revisa el DATA_ID o la ruta.")
    else:
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        print(f"✅ Archivo cargado correctamente. Tipo de objeto: {type(results)}")
        print(f"📊 Número de realizaciones (seeds) capturadas: {len(results)}")

    rows = []
    for seed, metrics in results.items():   
        # Asumimos que todas las métricas tienen las mismas claves (frac_sample_sizes)
        frac_sample_sizes = metrics['ARI'].keys() 
        
        for frac in frac_sample_sizes:
            row = {
                'random_state': seed,
                'frac_sample_size': frac,
                'time': metrics['time'].get(frac),
                'adj_accuracy': metrics['adj_accuracy'].get(frac),
                'ARI': metrics['ARI'].get(frac),
                'status': metrics['status'].get(frac) if 'status' in metrics else 'OK'
            }
            rows.append(row)

    df = pl.DataFrame(rows)

    df = df.with_columns(
            pl.when(
                pl.col('status').str.contains('Error')
            ).then(
                True
            ).otherwise(
                False
            ).alias('status_error')
        )

    # 2. Preparación de datos (Polars)
    df = df.with_columns((pl.col("frac_sample_size") * 100).alias("sample_size_pct"))
    
    # Agrupamos y calculamos todo (medias y desviaciones estándar) internamente
    df_avg = df.group_by("sample_size_pct").agg([
        pl.col("adj_accuracy").mean().alias("mean_acc"),
        pl.col("adj_accuracy").std().alias("std_acc"),
        pl.col("ARI").mean().alias("mean_ari"),
        pl.col("ARI").std().alias("std_ari"),
        pl.col("time").mean().alias("mean_time"),
        pl.col("time").std().alias("std_time"),
        pl.col('status_error').mean().alias('prop_status_error')
    ]).sort("sample_size_pct")

    df_avg = df_avg.filter(pl.col('prop_status_error') < prop_errors_threshold)

    return df, df_avg

########################################################################################################################################################################

def process_experiment_3_results(results_path, prop_errors_threshold):

    if not os.path.exists(results_path):
        print("❌ Error: El archivo no existe. Revisa el DATA_ID o la ruta.")
    else:
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        print(f"✅ Archivo cargado correctamente. Tipo de objeto: {type(results)}")
        print(f"📊 Número de realizaciones (seeds) capturadas: {len(results)}")

    rows = []
    for seed, metrics in results.items():   
        # Asumimos que todas las métricas tienen las mismas claves (frac_sample_sizes)
        n_splits_arr = metrics['ARI'].keys() 
        frac_sample_size_arr = metrics['ARI'][list(n_splits_arr)[0]].keys()
        
        for n_splits in n_splits_arr:
            for frac in frac_sample_size_arr:        
                row = {
                    'random_state': seed,
                    'n_splits': n_splits,
                    'frac_sample_size': frac,
                    'time': metrics['time'].get(n_splits).get(frac),
                    'adj_accuracy': metrics['adj_accuracy'].get(n_splits).get(frac),
                    'ARI': metrics['ARI'].get(n_splits).get(frac),
                    'status': metrics['status'].get(n_splits).get(frac) if 'status' in metrics else 'OK',
                }
                rows.append(row)

    df = pl.DataFrame(rows)

    df = df.with_columns(
            pl.when(
                pl.col('status').str.contains('Error')
            ).then(
                True
            ).otherwise(
                False
            ).alias('status_error')
        )

    df = df.with_columns((pl.col("frac_sample_size") * 100).alias("sample_size_pct"))

    df_avg = df.group_by(["n_splits", "sample_size_pct"]).agg([
        pl.col("adj_accuracy").mean().alias("mean_acc"),
        pl.col("adj_accuracy").std().alias("std_acc"),
        pl.col("ARI").mean().alias("mean_ari"),
        pl.col("ARI").std().alias("std_ari"),
        pl.col("time").mean().alias("mean_time"),
        pl.col("time").std().alias("std_time"),
        pl.col("status_error").mean().alias("prop_status_error")
    ]).sort(["n_splits", "sample_size_pct"])

    df_avg = df_avg.filter(pl.col('prop_status_error') < prop_errors_threshold)

    return df, df_avg

########################################################################################################################################################################

def process_experiment_4_results(results_path):

    if not os.path.exists(results_path):
        print("❌ Error: El archivo no existe. Revisa el DATA_ID o la ruta.")
    else:
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        print(f"✅ Archivo cargado correctamente. Tipo de objeto: {type(results)}")
        print(f"📊 Número de realizaciones (seeds) capturadas: {len(results)}")

    records = []
    for seed, metrics_dict in results.items():
        for metric, size_dict in metrics_dict.items():
            for n_samples, model_dict in size_dict.items():
                for model_name, value in model_dict.items():
                    records.append({
                        'seed': seed,
                        'n_samples': n_samples,
                        'model_name': model_name,
                        'metric': metric,
                        'value': value
                    })

    df_long = pl.DataFrame(records)

    df = df_long.pivot(
        index=['seed', 'n_samples', 'model_name'], 
        on='metric', 
        values='value'
    )

    df_avg = df.group_by(["model_name", "n_samples"]).agg([
            pl.col("adj_accuracy").mean().alias("mean_acc"),
            pl.col("adj_accuracy").std().alias("std_acc"),
            pl.col("ARI").mean().alias("mean_ari"),
            pl.col("ARI").std().alias("std_ari"),
            pl.col("time").mean().alias("mean_time"),
            pl.col("time").std().alias("std_time")
        ]).sort(["model_name", "n_samples"])
    
    return df, df_avg

########################################################################################################################################################################

def process_experiment_5_results(results_path, not_feasible_methods_to_add, prop_errors_threshold):   

    if not os.path.exists(results_path):
        print("❌ Error: El archivo no existe. Revisa el DATA_ID o la ruta.")
    else:
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        print(f"✅ Archivo cargado correctamente. Tipo de objeto: {type(results)}")
        print(f"📊 Número de realizaciones (seeds) capturadas: {len(results)}")

    rows = []
    for seed, metrics in results.items():   
        
        model_names_arr = metrics['ARI'].keys() 

        for model_name in model_names_arr:        
            row = {
                'random_state': seed,
                'model_name': model_name,
                'time': metrics['time'].get(model_name),
                'adj_accuracy': metrics['adj_accuracy'].get(model_name),
                'ARI': metrics['ARI'].get(model_name),
                'status': metrics['status'].get(model_name) if 'status' in metrics else 'OK'
            }
            rows.append(row)

        df = pl.DataFrame(rows)

        df = df.with_columns(
        pl.when(
            pl.col('status').str.contains('Error')
        ).then(
            True
        ).otherwise(
            False
        ).alias('status_error')
        )

        df_avg = (
        df.group_by(['model_name'])
        .agg(
            [pl.mean(c).alias(f'mean_{c}'.lower()) for c in ['ARI', 'adj_accuracy', 'time']] +
            [pl.std(c).alias(f'std_{c}'.lower()) for c in ['ARI', 'adj_accuracy', 'time']] +
            [pl.mean('status_error').alias('prop_status_error')]
        )
        .sort(['mean_adj_accuracy'], descending=True, nulls_last=True)
        )

        not_feasible_methods = df_avg.filter(pl.col('prop_status_error') >= prop_errors_threshold)['model_name'].unique().to_list() + not_feasible_methods_to_add
             
        rows_to_add = []
        for m in not_feasible_methods: 
            if m not in df_avg['model_name'].unique():
                rows_to_add.append({k: None if k != 'model_name' else m for k in df_avg.columns})
        df_avg = pl.concat([df_avg, pl.DataFrame(rows_to_add)], how='vertical')

    return df, df_avg, not_feasible_methods, results

########################################################################################################################################################################

def fast_mds(sample_size, X, d1, d2, d3, robust_method, random_state, config_experiment):

    X = check_array(X)

    np.random.seed(random_state)
    sample_idx = np.random.choice(range(X.shape[0]), sample_size)

    D = generalized_gower_dist_matrix(
            X=X[sample_idx,:], 
            p1=config_experiment['p1'], 
            p2=config_experiment['p2'], 
            p3=config_experiment['p3'], 
            d1=d1, d2=d2, d3=d3, 
            robust_method=robust_method, 
            alpha=config_experiment['alpha'], 
        )
    
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=random_state) 

    X_mds = mds.fit_transform(D)

    return X_mds, sample_idx

########################################################################################################################################################################

def plot_experiment_1_results(time_results, data_sizes, save_path):

    times_values = np.array(list(time_results.values()))
    times_values = times_values / 60 # to minutes
    fig, ax = plt.subplots(figsize=(7,4))
    ax = sns.lineplot(x=data_sizes, y=times_values, color='blue', marker='o', markersize=6)
    ax.set_ylabel('Time (min)', size=11)
    ax.set_xlabel('Data Size', size=11)
    plt.xticks(fontsize=10, rotation=0)
    plt.yticks(fontsize=10)
    ax.set_xticks(data_sizes)
    ax.set_yticks(np.round(np.linspace(np.min(times_values), np.max(times_values), 9), 0))
    plt.title("KMedoids (Euclidean) - Time and Data Size", fontsize=12.5, weight='bold')
    plt.tight_layout()
    # Guardado
    fig.savefig(save_path, format='png', dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.show()

########################################################################################################################################################################

def plot_experiment_2_results(df, df_avg, data_name, num_realizations, save_path, 
                              error_style='fill', ylim_acc=None, ylim_ari=None, ylim_time=None):
    """
    Genera los gráficos del experimento 2 partiendo del DataFrame crudo.
    
    Parámetros:
    - df: DataFrame crudo de Polars (con las realizaciones sin agregar).
    - df_avg: DataFrame con las medias y desviaciones estándar calculadas.
    - error_style: 'fill' (área sombreada), 'bar' (barras de error), 'boxplot' (cajas) o None (solo línea).
    - ylim_...: Tuplas (min, max) opcionales para fijar los ejes Y.
    """
    
    # 1. Validación del parámetro de estilo
    valid_styles = ['fill', 'bar', 'boxplot', None]
    if error_style not in valid_styles:
        raise ValueError(f"El parámetro error_style debe ser uno de: {valid_styles}")

    # Extraemos el mejor resultado
    best_row = df_avg.sort("mean_acc", descending=True).row(0, named=True)
    
    # Coordenadas X (Numéricas para fill/bar/None, categóricas para boxplot)
    x_vals = df_avg.get_column("sample_size_pct")
    x_indices = range(len(x_vals)) 
    best_x_val = best_row["sample_size_pct"]
    best_x_idx = x_vals.to_list().index(best_x_val)

    # 3. Diccionario de configuración por métrica
    metrics = {
        'Adj. Accuracy': {
            'raw_col': 'adj_accuracy', 'mean_col': 'mean_acc', 'std_col': 'std_acc',
            'title': 'Adj. Accuracy vs. Sample Size', 'ylim': ylim_acc
        },
        'ARI': {
            'raw_col': 'ARI', 'mean_col': 'mean_ari', 'std_col': 'std_ari',
            'title': 'ARI vs. Sample Size', 'ylim': ylim_ari
        },
        'Time (secs)': {
            'raw_col': 'time', 'mean_col': 'mean_time', 'std_col': 'std_time',
            'title': 'Time vs. Sample Size', 'ylim': ylim_time
        }
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes = axes.flatten()

    # 4. Bucle principal de trazado
    for ax, (ylabel, data) in zip(axes, metrics.items()):
        # Extraemos las series de Polars para cálculos rápidos
        y_mean = df_avg.get_column(data['mean_col'])
        y_std = df_avg.get_column(data['std_col'])
        best_y = best_row[data['mean_col']]

        if error_style == 'boxplot':
            # Seaborn usa un eje X categórico (0, 1, 2...)
            sns.boxplot(data=df, x="sample_size_pct", y=data['raw_col'], color="whitesmoke", ax=ax)
            ax.plot(x_indices, y_mean.to_list(), color='blue', marker='o', markersize=5, linestyle='-', linewidth=1.5)
            ax.plot(best_x_idx, best_y, color='red', marker='o', markersize=8, linestyle='None')
            ax.grid(True, linestyle='--', alpha=0.5, axis='y') # Grid solo horizontal para boxplots
            
        else:
            # Matplotlib usa el eje X numérico real (10, 20, 30...)
            # Si es 'fill' o None, dibujamos la línea principal
            if error_style in ['fill', None]:
                ax.plot(x_vals.to_list(), y_mean.to_list(), color='blue', marker='o', markersize=5)
                # Solo añadimos el sombreado si es 'fill'
                if error_style == 'fill':
                    ax.fill_between(x_vals.to_list(), (y_mean - y_std).to_list(), (y_mean + y_std).to_list(), color='blue', alpha=0.2)
            
            # Si es 'bar', usamos errorbar que ya incluye la línea
            elif error_style == 'bar':
                ax.errorbar(x_vals.to_list(), y_mean.to_list(), yerr=y_std.to_list(), color='blue', marker='o', markersize=5, capsize=4)
            
            ax.plot(best_x_val, best_y, color='red', marker='o', markersize=8, linestyle='None')
            ax.grid(True, linestyle='--', alpha=0.5)

        # Configuraciones comunes
        ax.set_title(data['title'], size=12, weight='bold')
        ax.set_ylabel(ylabel, size=11)
        ax.set_xlabel('Sample Size Parameter (%)', size=11)
        
        if data['ylim'] is not None:
            ax.set_ylim(data['ylim'])

    # 5. Ajustes finales y guardado
    plt.subplots_adjust(top=0.83, hspace=0.3, wspace=0.25)
    
    # Subtítulo adaptado al estilo elegido
    formatted_data_name = data_name.replace('_', ' ').capitalize()
    
    plt.suptitle(
        f'Accuracy, ARI and Time vs. Sample Size Parameter\n{formatted_data_name} - Realizations: {num_realizations}', 
        fontsize=13, y=1.02, weight='bold', color='black', alpha=1
    )

    fig.savefig(save_path, format='png', dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.show()

########################################################################################################################################################################

def plot_experiment_3_results(df, df_avg, data_name, num_realizations, save_path, 
                              error_style='fill', ylim_acc=None, ylim_ari=None, ylim_time=None):
    """
    Genera los gráficos del experimento 3 alineando correctamente las líneas
    con los boxplots agrupados (dodged).
    """
    
    # 1. Validación de estilo
    valid_styles = ['fill', 'bar', 'boxplot', None]
    if error_style not in valid_styles:
        raise ValueError(f"El parámetro error_style debe ser uno de: {valid_styles}")

    # 3. Identificamos al GANADOR global
    best_row = df_avg.sort("mean_acc", descending=True).row(0, named=True)
    best_frac_pct = best_row['sample_size_pct']
    best_fold = best_row['n_splits']  # Necesitamos saber en qué fold ocurrió para el offset

    # 4. Configuración
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    axes = axes.flatten()

    metrics = {
        'Adj. Accuracy': {'raw_col': 'adj_accuracy', 'mean_col': 'mean_acc', 'std_col': 'std_acc', 'ylim': ylim_acc},
        'ARI': {'raw_col': 'ARI', 'mean_col': 'mean_ari', 'std_col': 'std_ari', 'ylim': ylim_ari},
        'Time (secs)': {'raw_col': 'time', 'mean_col': 'mean_time', 'std_col': 'std_time', 'ylim': ylim_time}
    }

    folds = df_avg["n_splits"].unique().sort().to_list()
    palette = sns.color_palette("tab10", n_colors=len(folds))
    df_pd = df.to_pandas()
    unique_fracs = sorted(df["sample_size_pct"].unique().to_list())

    # --- BUCLE PRINCIPAL DE DIBUJO ---
    for ax, (ylabel, data) in zip(axes, metrics.items()):
        best_y = best_row[data['mean_col']]

        if error_style == 'boxplot':
            sns.boxplot(data=df_pd, x="sample_size_pct", y=data['raw_col'], hue="n_splits", 
                        palette=palette, ax=ax, boxprops={'alpha': 0.6})
            
            # --- CÁLCULO DEL OFFSET (DODGE) DE SEABORN ---
            n_hues = len(folds)
            dodge_width = 0.8 / n_hues  # 0.8 es el ancho que usa Seaborn por defecto
            
            for i, k in enumerate(folds):
                subset = df_avg.filter(pl.col("n_splits") == k)
                y_mean = subset[data['mean_col']].to_list()
                
                # Desplazamos las X para que caigan en el centro de su caja
                offset = (i - (n_hues - 1) / 2) * dodge_width
                x_dodged = [x + offset for x in range(len(unique_fracs))]
                
                ax.plot(x_dodged, y_mean, marker='o', markersize=5, color=palette[i], linewidth=1.5)
            
            # Punto rojo: También necesita el offset correspondiente a su fold ganador
            best_x_idx = unique_fracs.index(best_frac_pct)
            best_hue_idx = folds.index(best_fold)
            best_offset = (best_hue_idx - (n_hues - 1) / 2) * dodge_width
            
            ax.plot(best_x_idx + best_offset, best_y, marker='o', color='red', markersize=7, zorder=10, linestyle='None')
            
            if ax.get_legend() is not None: ax.get_legend().remove()

        else:
            # Modos numéricos (fill, bar, None) no tienen "dodge", se dibujan sobre la X real
            for i, k in enumerate(folds):
                subset = df_avg.filter(pl.col("n_splits") == k)
                x_vals = subset["sample_size_pct"].to_list()
                y_mean = subset[data['mean_col']].to_list()
                y_std = subset[data['std_col']].to_list()
                color = palette[i]

                if error_style in ['fill', None]:
                    ax.plot(x_vals, y_mean, marker='o', markersize=5, color=color, label=f"{k}-Fold")
                    if error_style == 'fill':
                        y_upper = (subset.get_column(data['mean_col']) + subset.get_column(data['std_col'])).to_list()
                        y_lower = (subset.get_column(data['mean_col']) - subset.get_column(data['std_col'])).to_list()
                        ax.fill_between(x_vals, y_lower, y_upper, color=color, alpha=0.15)
                        
                elif error_style == 'bar':
                    ax.errorbar(x_vals, y_mean, yerr=y_std, marker='o', markersize=5, color=color, capsize=4, label=f"{k}-Fold")

            ax.plot(best_frac_pct, best_y, marker='o', color='red', markersize=7, zorder=10, linestyle='None')

        # Estética
        ax.set_title(f"{ylabel.split()[0]} vs Number of Folds and Sample Size", fontsize=11, fontweight='bold')
        ax.set_xlabel("Sample Size Parameter (%)", size=11)
        ax.set_ylabel(ylabel, size=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        if data['ylim'] is not None:
            ax.set_ylim(data['ylim'])

    # --- LEYENDA GLOBAL Y AJUSTES FINALES ---
    handles = [plt.Rectangle((0,0),1,1, color=palette[i], alpha=0.8) for i in range(len(folds))]
    labels = [f"{k}-Fold" for k in folds]
    best_marker = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=7, label='Best Acc.')
    handles.append(best_marker)
    labels.append('Best Acc.')
    
    fig.legend(handles, labels, loc='lower center', ncol=len(folds)+1, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    plt.subplots_adjust(top=0.83, bottom=0.20, wspace=0.25)

    formatted_data_name = data_name.replace('_', ' ').capitalize()
    fig.suptitle(
        f'Accuracy, ARI and Time vs. Number of Folds and Sample Size Parameter\n{formatted_data_name} - Realizations: {num_realizations}',  
        fontsize=13, fontweight='bold', y=1.02
    )

    fig.savefig(save_path, format='png', dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.show()

########################################################################################################################################################################

def plot_experiment_4_results(df, df_avg, num_realizations, save_path, 
                              error_style='fill', ylim_acc=None, ylim_ari=None, ylim_time=None):
    """
    Genera los gráficos del experimento 4 (Data Size vs Model) 
    permitiendo múltiples estilos de error.
    """
    
    # 1. Validación de estilo
    valid_styles = ['fill', 'bar', 'boxplot', None]
    if error_style not in valid_styles:
        raise ValueError(f"El parámetro error_style debe ser uno de: {valid_styles}")

    # 2. Configurar metadatos de los modelos
    model_config = {
        'KMedoids-euclidean': {
            'color': 'green', 
            'label': 'Kmedoids-Euclidean'
        },
        'FastKmedoidsGGower-robust_mahalanobis_winsorized-sokal-hamming': {
            'color': 'blue', 
            'label': 'FastKmedoids-RobustGGower'
        },
        'FoldFastKmedoidsGGower-robust_mahalanobis_winsorized-sokal-hamming': {
            'color': 'red', 
            'label': 'FoldFastKmedoids-RobustGGower' 
        }
    }

    # Filtramos para usar solo los modelos que realmente están en el DataFrame
    # y mantenemos el orden del diccionario para consistencia visual
    unique_models = [m for m in model_config.keys() if m in df["model_name"].unique().to_list()]
    
    # 3. Preparación de datos agregados (Polars)

    # 4. Inicializar la figura y métricas
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    axes = axes.flatten()

    metrics = {
        'Adj. Accuracy': {'raw_col': 'adj_accuracy', 'mean_col': 'mean_acc', 'std_col': 'std_acc', 'ylim': ylim_acc, 'title': 'Adj. Accuracy vs Data Size'},
        'ARI': {'raw_col': 'ARI', 'mean_col': 'mean_ari', 'std_col': 'std_ari', 'ylim': ylim_ari, 'title': 'ARI vs Data Size'},
        'Time (secs)': {'raw_col': 'time', 'mean_col': 'mean_time', 'std_col': 'std_time', 'ylim': ylim_time, 'title': 'Time vs Data Size'}
    }

    unique_samples = sorted(df["n_samples"].unique().to_list())
    palette = {m: model_config[m]['color'] for m in unique_models}

    # 5. Generar los gráficos iterando sobre las métricas
    for ax, (ylabel, data) in zip(axes, metrics.items()):
        
        if error_style == 'boxplot':
            # Seaborn separa las cajas por modelo (hue) automáticamente
            sns.boxplot(data=df, x="n_samples", y=data['raw_col'], hue="model_name", 
                        palette=palette, hue_order=unique_models, ax=ax, boxprops={'alpha': 0.6})
            
            # Cálculo del offset (dodge) para centrar las líneas
            n_hues = len(unique_models)
            dodge_width = 0.8 / n_hues
            
            for i, model_name in enumerate(unique_models):
                subset = df_avg.filter(pl.col("model_name") == model_name)
                y_mean = subset[data['mean_col']].to_list()
                
                offset = (i - (n_hues - 1) / 2) * dodge_width
                x_dodged = [x + offset for x in range(len(unique_samples))]
                
                ax.plot(x_dodged, y_mean, marker='o', markersize=5, color=model_config[model_name]['color'], linewidth=1.5)
            
            if ax.get_legend() is not None: ax.get_legend().remove()

        else:
            # Modos numéricos continuos (fill, bar, None)
            for model_name in unique_models:
                config = model_config[model_name]
                subset = df_avg.filter(pl.col("model_name") == model_name)
                
                x_vals = subset["n_samples"].to_list()
                y_mean = subset[data['mean_col']].to_list()
                y_std = subset[data['std_col']].to_list()
                color = config['color']

                if error_style in ['fill', None]:
                    ax.plot(x_vals, y_mean, marker='o', markersize=5, color=color)
                    if error_style == 'fill':
                        y_upper = (subset.get_column(data['mean_col']) + subset.get_column(data['std_col'])).to_list()
                        y_lower = (subset.get_column(data['mean_col']) - subset.get_column(data['std_col'])).to_list()
                        ax.fill_between(x_vals, y_lower, y_upper, color=color, alpha=0.15)
                        
                elif error_style == 'bar':
                    ax.errorbar(x_vals, y_mean, yerr=y_std, marker='o', markersize=5, color=color, capsize=4)

        # Detalles de cada subplot
        ax.set_title(data['title'], size=13, weight='bold')
        ax.set_xlabel('Data Size', size=12)
        ax.set_ylabel(ylabel, size=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        if data['ylim'] is not None:
            ax.set_ylim(data['ylim'])

    # 6. Ajustes globales y leyenda unificada
    handles = [plt.Rectangle((0,0),1,1, color=model_config[m]['color'], alpha=0.8) for m in unique_models]
    labels = [model_config[m]['label'] for m in unique_models]
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=11)

    plt.subplots_adjust(top=0.82, bottom=0.20, wspace=0.25)
    
    plt.suptitle(
        f'K-Medoids vs Fast K-Medoids vs Fold Fast K-Medoids)\n'
        f'Num. Experiment Realizations = {num_realizations}', 
        fontsize=14, weight='bold', color='black', alpha=1, y=1.02
    )

    # 7. Guardar y mostrar
    fig.savefig(save_path, format='png', dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.show()

########################################################################################################################################################################

def plot_experiment_5_results(df_avg, data_name, num_realizations, save_path, 
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
    model_names = df_avg["model_name"].to_numpy()
    
    # Medias
    avg_adj_accuracy = df_avg["mean_adj_accuracy"].fill_null(0).to_numpy()
    avg_ari = df_avg["mean_ari"].fill_null(0).to_numpy()
    avg_time = df_avg["mean_time"].fill_null(0).to_numpy()
    
    # Desviaciones 
    std_adj_acc = df_avg["std_adj_accuracy"].fill_null(0).to_numpy()
    std_ari = df_avg["std_ari"].fill_null(0).to_numpy()
    std_time = df_avg["std_time"].fill_null(0).to_numpy()

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
            elinewidth=1.2,
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

