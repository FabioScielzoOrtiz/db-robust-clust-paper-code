###########################################################################################

# --- IMPORTS ---

import os
import sys
import logging
import polars as pl

###########################################################################################

# --- LOGGING CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

###########################################################################################

# --- PATH CONFIGURATION ---

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(script_path, '..', '..', '..'))

# Path for the pre-calculated datasets structure
structure_data_path = os.path.join(project_path, 'data', 'processed_data', 'datasets_structure.parquet')
sys.path.append(project_path)

###########################################################################################

# --- CUSTOM IMPORTS ---

from src.utils.experiments_exploration_utils import process_experiment_3_results
from config.config_experiment_2 import CONFIG_EXPERIMENT, PROP_ERRORS_THRESHOLD

###########################################################################################

# --- MAIN EXECUTION ---

def main():

    logging.info("▶️ STARTING MASTER TABLE GENERATION (EXPERIMENT 3)")
    
    # 1. Load the pre-calculated datasets structure
    if not os.path.exists(structure_data_path):
        logging.error(f" ❌ Structure file not found: {structure_data_path}")
        logging.error("    Please run get_datasets_structure.py first.")
        sys.exit(1)
        
    logging.info(f"  > Loading datasets structure from: {structure_data_path}")
    df_structure = pl.read_parquet(structure_data_path)
    
    DATA_IDS = list(CONFIG_EXPERIMENT.keys())
    MODES = ['3a', '3b']

    # 2. Iterate over modes
    for mode in MODES:

        logging.info(f"\n" + "="*50)
        logging.info(f"📦 PROCESSING SUB-EXPERIMENT: {mode.upper()}")
        logging.info("="*50)
        
        df_avg_results_list = []
        
        # Guardaremos la tabla maestra en la misma carpeta raíz del mode
        results_dir = os.path.join(project_path, 'results', f'experiment_{mode}')
        results_save_path = os.path.join(results_dir, f'master_table_exp_{mode}.parquet')
        os.makedirs(results_dir, exist_ok=True)

        for data_id in DATA_IDS:
            logging.info(f"  > Processing dataset results: {data_id}")
            
            # Recreamos la ruta exacta basada en tu snippet
            dataset_dir = os.path.join(results_dir, data_id)
            filename = f'results_exp_{mode}_{data_id}.pkl'
            results_path = os.path.join(dataset_dir, filename)
            
            if not os.path.exists(results_path):
                logging.warning(f"  ⚠️ Not found: {results_path}. Skipping...")
                continue

            try:
                # Extraemos los DataFrames usando tu función específica
                _, df_avg = process_experiment_3_results(
                    results_path=results_path,
                    prop_errors_threshold=PROP_ERRORS_THRESHOLD
                )
                
                # Añadir la columna data_id y clonar por seguridad
                df_avg = df_avg.with_columns(pl.lit(data_id).alias("data_id")).clone()
                df_avg_results_list.append(df_avg)
                
            except Exception as e:
                logging.error(f"  ❌ Error processing results for {data_id}: {e}")
                continue

        # 3. Join and Save Final Master Table for the mode
        if df_avg_results_list:
            logging.info(f"  > Concatenating results for {mode}...")
            df_concat = pl.concat(df_avg_results_list, how='vertical')
            
            logging.info("  > Joining results with dataset structure metadata...")
            df_master = df_concat.join(df_structure, on='data_id', how='left')
            
            logging.info(f"  📊 Master table {mode} shape: {df_master.shape}")

            try:
                df_master.write_parquet(results_save_path)
                logging.info(f"  💾 Final file saved: {results_save_path}")
            except Exception as e:
                logging.error(f"  ❌ Failed to save final file for {mode}: {e}")
        else:
            logging.error(f"  ❌ No data was processed for {mode}. Master table not created.")

    logging.info("\n✅ MASTER TABLES FOR EXPERIMENT 3 PIPELINE FINISHED SUCCESSFULLY")

###########################################################################################

if __name__ == "__main__":
    main()