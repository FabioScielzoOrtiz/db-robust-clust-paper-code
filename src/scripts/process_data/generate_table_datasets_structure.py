###########################################################################################
# --- IMPORTS ---

import os
import sys
import json
import logging
import random
import argparse # <-- NUEVO: Para procesar argumentos de línea de comandos
import numpy as np
import polars as pl
from collections import defaultdict

###########################################################################################
# --- LOGGING CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

###########################################################################################
# --- GLOBAL CONSTANTS ---

# Métricas que varían con la semilla y de las cuales queremos media y desviación estándar
METRICS_TO_AVERAGE = [
    'mean_prop_outliers_quant', 
    'silhouette_index',
    'prop_high_corr_quant', 
    'sphericity_quant', 
    'prop_redundancy_quant',
    'normalized_balance_entropy', 
    'imbalance_ratio'
]

###########################################################################################
# --- PATH CONFIGURATION ---

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(script_path, '..', '..', '..'))

output_dir = os.path.join(project_path, 'data', 'processed_data')
output_save_path = os.path.join(output_dir, 'datasets_structure.parquet') 
os.makedirs(output_dir, exist_ok=True)

sys.path.append(project_path)

###########################################################################################
# --- CUSTOM IMPORTS ---

from src.utils.simulations_utils import generate_simulation
from src.utils.experiments_exploration_utils import get_dataset_structure  
from config.config_simulations import SIMULATION_CONFIGS, REAL_DATASET_KEYS
from config.config_experiment_5 import EXPERIMENT_RANDOM_STATE, N_REALIZATIONS

###########################################################################################
# --- MAIN EXECUTION ---

def get_experiment_seeds():
    """Replica la lógica de generación de semillas del experimento 5."""
    random.seed(EXPERIMENT_RANDOM_STATE)
    # Generamos N_REALIZATIONS semillas únicas
    return random.sample(range(N_REALIZATIONS * 1000), N_REALIZATIONS)


def main(force=False): # <-- NUEVO: Parámetro force
    
    logging.info("▶️ STARTING MASTER DATASET STRUCTURE GENERATION")
    if force:
        logging.warning("⚠️ FORCE MODE ENABLED: All datasets will be processed from scratch and previous data will be overwritten.")
    
    simulation_names = list(SIMULATION_CONFIGS.keys())
    DATA_IDS = simulation_names + REAL_DATASET_KEYS
    
    # =====================================================================
    # NUEVO: LÓGICA DE PROCESAMIENTO INCREMENTAL
    # =====================================================================
    existing_df = None
    processed_ids = set()

    # Solo leemos el archivo existente si NO estamos en modo force
    if not force and os.path.exists(output_save_path):
        try:
            existing_df = pl.read_parquet(output_save_path)
            
            id_column_name = 'data_id'  
            if id_column_name in existing_df.columns:
                processed_ids = set(existing_df[id_column_name].to_list())
                logging.info(f"  > Found existing file with {len(processed_ids)} already processed datasets.")
            else:
                logging.warning(f"  > Could not find an ID column in existing Parquet. Will process all.")
        except Exception as e:
            logging.warning(f"  > Could not read existing file. Starting fresh. Error: {e}")

    # Filtramos los IDs para quedarnos solo con los que no están en el archivo
    PENDING_DATA_IDS = [d_id for d_id in DATA_IDS if d_id not in processed_ids] 
    
    if not PENDING_DATA_IDS:
        logging.info("  ✅ All datasets have already been processed. Nothing to do. Exiting.")
        return # Termina la ejecución limpiamente
    
    experiment_seeds = get_experiment_seeds()
    #experiment_seeds = experiment_seeds[:20]
    
    logging.info(f"  > Generated {len(experiment_seeds)} seeds for simulation averaging.")
    logging.info(f"  > Total datasets to process this run: {len(PENDING_DATA_IDS)} out of {len(DATA_IDS)}")
    
    df_structure_list = []

    for data_id in PENDING_DATA_IDS:
        logging.info(f"  ⚙️ Processing dataset: {data_id.upper()}")

        # =====================================================================
        # 1. FLUJO PARA SIMULACIONES (Con promedio de semillas)
        # =====================================================================
        if data_id in SIMULATION_CONFIGS:
            simulation_config = SIMULATION_CONFIGS.get(data_id)
            metrics_accumulator = defaultdict(list)
            
            # Iteramos sobre todas las semillas del experimento
            for seed in experiment_seeds:
                logging.info(f"    ➡️ Processing for random state: {seed}")
                try:
                    X, y = generate_simulation(
                        **simulation_config,
                        return_outlier_idx=False,
                        random_state=seed
                    )
                    
                    # Safe Polars conversion
                    if 'pandas' in str(type(X)).lower():
                        X_pl = pl.from_pandas(X)
                    else:
                        cols = [f"X{i+1}" for i in range(X.shape[1])]
                        X_pl = pl.DataFrame(X, schema=cols)
                        
                    y_np = np.array(y).flatten()

                    # Calcular estructura de esta semilla específica
                    structure = get_dataset_structure(
                        X_pl, y_np, data_id, None, None, None, simulation_config
                    )
                    
                    # Acumular métricas
                    for k, v in structure.items():
                        metrics_accumulator[k].append(v)
                        
                except Exception as e:
                    logging.error(f"  ❌ Error generating sim {data_id} for seed {seed}: {e}")
                    continue

            # Consolidar resultados promediando
            if metrics_accumulator:
                avg_structure = {}
                for key, values in metrics_accumulator.items():
                    valid_values = [v for v in values if v is not None]
                    if not valid_values:
                        avg_structure[key] = None
                        if key in METRICS_TO_AVERAGE:
                            avg_structure[f"{key}_std"] = None
                        continue
                    
                    first_val = valid_values[0]

                    # Si es una métrica continua, calculamos media y std
                    if key in METRICS_TO_AVERAGE:
                        avg_structure[key] = round(float(np.mean(valid_values)), 4)
                        avg_structure[f"{key}_std"] = round(float(np.std(valid_values)), 4)
                    else:
                        # Para variables de configuración o estáticas (id, n_rows, geometría...)
                        avg_structure[key] = first_val
                
                df_structure_list.append(pl.DataFrame([avg_structure]))
                logging.info(f"    ✅ Averaged structure (mean & std) extracted successfully.")
            else:
                logging.error(f"    ❌ No successful seeds processed for simulation {data_id}.")

        # =====================================================================
        # 2. FLUJO PARA DATASETS REALES (Sin promedio)
        # =====================================================================
        else:
            processed_data_dir = os.path.join(project_path, 'data', 'processed_data')
            metadata_path = os.path.join(processed_data_dir, f'metadata_{data_id}.json')
            processed_data_path = os.path.join(processed_data_dir, f'{data_id}_processed.parquet')
            
            if not os.path.exists(metadata_path) or not os.path.exists(processed_data_path):
                 logging.warning(f"  ⚠️ Data or metadata not found for real dataset {data_id}. Skipping...")
                 continue
            
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    
                data = pl.read_parquet(processed_data_path)
                quant_predictors = metadata.get('quant_predictors', [])
                binary_predictors = metadata.get('binary_predictors', [])
                multiclass_predictors = metadata.get('multiclass_predictors', [])
                
                all_cols = quant_predictors + binary_predictors + multiclass_predictors
                X = data.select(all_cols)
                y = data.select(metadata['response']).to_series().to_numpy()
                
                n_binary = len(binary_predictors)
                n_multiclass = len(multiclass_predictors)
                simulation_config = None
                
                # Extraer estructura
                data_structure = get_dataset_structure(
                    X, y, data_id, quant_predictors, n_binary, n_multiclass, simulation_config
                )

                # Inyectar columnas _std en 0.0 para mantener el schema igual que en las simulaciones
                for key in METRICS_TO_AVERAGE:
                    if key in data_structure:
                        data_structure[f"{key}_std"] = 0.0

                df_structure_list.append(pl.DataFrame([data_structure]))
                logging.info(f"  ✅ Real data structure extracted successfully.")
                
            except Exception as e:
                logging.error(f"  ❌ Error loading/calculating metrics for real dataset {data_id}: {e}")
                continue

    # =====================================================================
    # 3. JOIN AND SAVE FINAL TABLE
    # =====================================================================
    if df_structure_list:
        # Concatenamos los datos procesados en esta ejecución
        df_structure_new = pl.concat(df_structure_list, how='diagonal')
        
        # Si existían datos previos (y NO estamos en modo force), los unimos a los nuevos
        if existing_df is not None:
            df_structure_concat = pl.concat([existing_df, df_structure_new], how='diagonal')
            logging.info("  🔗 Appended new datasets to existing data.")
        else:
            df_structure_concat = df_structure_new

        logging.info(f"  📊 Final structure table shape: {df_structure_concat.shape}")

        try:
            df_structure_concat.write_parquet(output_save_path)
            logging.info(f"  💾 Final file saved: {output_save_path}")
        except Exception as e:
            logging.error(f"  ❌ Failed to save final file: {e}")
            sys.exit(1)
    else:
        logging.warning("  ⚠️ No new data was successfully processed in this run.")

    logging.info("✅ DATASET STRUCTURE PIPELINE FINISHED SUCCESSFULLY")

###########################################################################################

if __name__ == "__main__":
    
    # --- NUEVO: Configuración de argparse ---
    parser = argparse.ArgumentParser(description="Generate dataset structure and metrics.")
    parser.add_argument(
        '--force', 
        action='store_true', 
        help="Force the execution for all datasets, overwriting any existing parquet file."
    )
    
    args = parser.parse_args()
    
    # Pasamos el argumento force a la función main
    main(force=args.force)