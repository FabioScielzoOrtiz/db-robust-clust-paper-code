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

# Paths for results
results_dir = os.path.join(project_path, 'results', 'experiment_5')
results_save_path = os.path.join(results_dir, 'master_table_exp_5.parquet')
os.makedirs(results_dir, exist_ok=True)

# Path for the pre-calculated datasets structure
structure_data_path = os.path.join(project_path, 'data', 'processed_data', 'datasets_structure.parquet')

sys.path.append(project_path)

###########################################################################################

# --- CUSTOM IMPORTS ---

from src.utils.experiments_exploration_utils import process_experiment_5_results
from config.config_experiment_5 import CONFIG_EXPERIMENT, PROP_ERRORS_THRESHOLD

###########################################################################################

# --- MAIN EXECUTION ---

def main():
    
    logging.info(f"▶️ STARTING MASTER TABLE GENERATION (EXPERIMENT 5)")
    
    # 1. Load the pre-calculated datasets structure
    if not os.path.exists(structure_data_path):
        logging.error(f" ❌ Structure file not found: {structure_data_path}")
        logging.error("    Please run get_datasets_structure.py first.")
        sys.exit(1)
        
    logging.info(f"  > Loading datasets structure from: {structure_data_path}")
    df_structure = pl.read_parquet(structure_data_path)
    
    DATA_IDS = list(CONFIG_EXPERIMENT.keys())
    df_avg_results_list = []

    # 2. Process Experiment Results
    for data_id in DATA_IDS:
        
        logging.info(f"  > Processing dataset results: {data_id}")
        dataset_results_path = os.path.join(results_dir, data_id, f'results_exp_5_{data_id}.pkl')
        
        if not os.path.exists(dataset_results_path):
            logging.warning(f"  ⚠️ Not found: {dataset_results_path}. Skipping...")
            continue

        try:
            _, df_avg, _, _ = process_experiment_5_results(
                results_path=dataset_results_path, 
                prop_errors_threshold=PROP_ERRORS_THRESHOLD,
            )
            
            # Append data_id column and save to list
            df_avg = df_avg.with_columns(
                pl.lit(data_id).alias("data_id"),
                pl.col('model_name').replace({'KMedoids-euclidean': 'KMedoids-pam'})
            )
            df_avg_results_list.append(df_avg)
            
        except Exception as e:
            logging.error(f"  ❌ Error processing results for {data_id}: {e}")
            continue

    # 3. Join and Save Final Master Table
    if df_avg_results_list:
        logging.info("  > Concatenating experiment results...")
        df_avg_results_concat = pl.concat(df_avg_results_list, how='vertical')
        
        logging.info("  > Joining results with dataset structure metadata...")
        df_master = df_avg_results_concat.join(df_structure, on='data_id', how='left')
        
        logging.info(f"  📊 Master table shape: {df_master.shape}")

        try:
            df_master.write_parquet(results_save_path)
            logging.info(f"  💾 Final file saved: {results_save_path}")
        except Exception as e:
            logging.error(f"  ❌ Failed to save final file: {e}")
            sys.exit(1)
    else:
        logging.error("  ❌ No data was processed. Master table not created.")
        sys.exit(1)

    logging.info("✅ MASTER TABLE PIPELINE FINISHED SUCCESSFULLY")

###########################################################################################

if __name__ == "__main__":
    main()