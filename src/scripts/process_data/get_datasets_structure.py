###########################################################################################

# --- IMPORTS ---

import os
import sys
import json
import logging
import numpy as np
import polars as pl

###########################################################################################

# --- LOGGING CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

###########################################################################################

# --- GLOBAL CONSTANTS ---

# Define aquí los nombres exactos de tus datasets reales
REAL_DATASET_NAMES = [
    'dubai_houses', 
    'kc_houses',
    'heart_disease'
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
from config.config_simulations import SIMULATION_CONFIGS

###########################################################################################

# --- MAIN EXECUTION ---

def main():
    
    logging.info("▶️ STARTING MASTER DATASET STRUCTURE GENERATION")
    
    simulation_names = list(SIMULATION_CONFIGS.keys())
    DATA_IDS = simulation_names + REAL_DATASET_NAMES
    
    logging.info(f"  > Total datasets to process: {len(DATA_IDS)} ({len(simulation_names)} sims, {len(REAL_DATASET_NAMES)} real)")
    
    df_structure_list = []

    for data_id in DATA_IDS:
        logging.info(f"  > Processing dataset: {data_id}")

        # 1. Data Loading & Predictor Routing
        if data_id in SIMULATION_CONFIGS:
            simulation_config = SIMULATION_CONFIGS.get(data_id)
            
            try:
                X, y = generate_simulation(
                    **simulation_config,
                    random_state=123,
                    return_outlier_idx=False
                )
                
                # Safe Polars conversion
                if 'pandas' in str(type(X)).lower():
                    X = pl.from_pandas(X)
                else:
                    cols = [f"X{i+1}" for i in range(X.shape[1])]
                    X = pl.DataFrame(X, schema=cols)
                    
                y = np.array(y).flatten()

                quant_predictors, n_binary, n_multiclass = None, None, None
                
            except Exception as e:
                logging.error(f"  ❌ Error generating simulation for {data_id}: {e}")
                continue

        else:
            # Flujo para Datasets Reales
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
                
            except Exception as e:
                logging.error(f"  ❌ Error loading real dataset {data_id}: {e}")
                continue

        # 2. Extract Structure Data
        try:
            data_structure = get_dataset_structure(
                X, y, data_id, quant_predictors, n_binary, n_multiclass, simulation_config
            )
            df_structure_list.append(pl.DataFrame([data_structure]))
            logging.info(f"  ✅ Structure extracted successfully.")
        except Exception as e:
            logging.error(f"  ❌ Error calculating metrics for {data_id}: {e}")

    # 3. Join and Save Final Table
    if df_structure_list:
        df_structure_concat = pl.concat(df_structure_list, how='vertical')
        logging.info(f"  📊 Final structure table shape: {df_structure_concat.shape}")

        try:
            df_structure_concat.write_parquet(output_save_path)
            logging.info(f"  💾 Final file saved: {output_save_path}")
        except Exception as e:
            logging.error(f"  ❌ Failed to save final file: {e}")
            sys.exit(1)
    else:
        logging.error("  ❌ No data was processed. Master table not created.")
        sys.exit(1)

    logging.info("✅ DATASET STRUCTURE PIPELINE FINISHED SUCCESSFULLY")

###########################################################################################

if __name__ == "__main__":
    main()