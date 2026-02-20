########################################################################################################################################################################

import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines 

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

def plot_experiment_4_results(df, num_realizations, save_path):
        
    # 2. Configurar metadatos de los modelos para no repetir código
    model_config = {
        'KMedoids-euclidean': {
            'color': 'green', 
            'label': '$k$-medoids'
        },
        'FastKmedoidsGGower-robust_mahalanobis_winsorized-sokal-hamming': {
            'color': 'blue', 
            'label': 'Robust Fast $k$-medoids'
        },
        'FoldFastKmedoidsGGower-robust_mahalanobis_winsorized-sokal-hamming': {
            'color': 'red', 
            'label': 'Robust Fold Fast $k$-medoids' 
        }
    }

    # 3. Inicializar la figura
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    axes = axes.flatten()

    metrics_to_plot = [
        ('adj_accuracy', 'Adj. Accuracy vs Data size', 'Adj. Accuracy', axes[0]),
        ('ARI', 'ARI vs Data size', 'ARI', axes[1]),
        ('time', 'Time vs Data Size', 'Time (secs)', axes[2])
    ]

    unique_models = df["model_name"].unique().to_list()

    # 4. Generar los gráficos iterando sobre las métricas y los modelos
    for metric_col, title, ylabel, ax in metrics_to_plot:
        for model_name, config in model_config.items():
            
            # Comprobamos si el modelo está en nuestra lista de modelos únicos
            if model_name in unique_models:
                
                # POLARS: Filtrado de filas usando pl.col()
                df_subset = df.filter(pl.col('model_name') == model_name)
                
                sns.lineplot(
                    data=df_subset, 
                    x='n_samples', 
                    y=metric_col, 
                    color=config['color'], 
                    marker='o', 
                    markersize=5, 
                    label=config['label'], 
                    ax=ax
                )
        
        # Detalles de cada subplot
        ax.set_title(title, size=13, weight='bold')
        ax.set_ylabel(ylabel, size=12)
        ax.set_xlabel('Data Size', size=12)
        ax.legend().set_visible(False) # Ocultamos la leyenda individual

    # 5. Ajustes globales y leyenda unificada
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)

    plt.subplots_adjust(top=0.83, hspace=0.5, wspace=0.23)
    plt.suptitle(
        '$k$-Medoids vs Fast $k$-Medoids vs Fold Fast $k$-Medoids - Accuracy, ARI and Time\n'
        f'Num. Experiment Realizations = {num_realizations}', 
        fontsize=14, y=1.1, weight='bold', color='black', alpha=1
    )

    # 6. Guardar y mostrar
    fig.savefig(save_path, format='png', dpi=300, bbox_inches="tight", pad_inches=0.2)

    plt.show()

########################################################################################################################################################################

def plot_experiment_5_results(df, data_name, num_realizations, save_path, 
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