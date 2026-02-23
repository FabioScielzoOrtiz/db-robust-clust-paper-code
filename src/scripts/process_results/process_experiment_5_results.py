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

# --- PATH CONFIGURATION ---

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(script_path, '..', '..', '..'))
results_dir = os.path.join(project_path, 'results', 'experiment_5')
results_save_path = os.path.join(results_dir, 'master_table_exp_5.parquet')
os.makedirs(results_dir, exist_ok=True)

sys.path.append(project_path)

###########################################################################################

# --- CUSTOM IMPORTS ---

from src.utils.simulations_utils import generate_simulation
from src.utils.experiments_exploration_utils import process_experiment_5_results, get_dataset_structure

from config.config_experiment_5 import CONFIG_EXPERIMENT, PROP_ERRORS_THRESHOLD
from config.config_simulations import SIMULATION_CONFIGS

###########################################################################################

# --- CONSTANTS & HELPER FUNCTIONS ---

NOT_FEASIBLE_METHODS_TO_ADD = {
    k: ['SpectralClustering', 'Dipinit']
    for k in ['simulation_1', 'kc_houses']
}

NOT_FEASIBLE_METHODS_TO_ADD.update({
    k: ['SpectralClustering', 'KMedoids-euclidean', 'Diana', 'Birch', 'Dipinit', 'AgglomerativeClustering']
    for k in [f'simulation_{i}' for i in range(2, 7 + 1)]
})

NOT_FEASIBLE_METHODS_TO_ADD.update({
    k: [] for k in ['dubai_houses', 'heart_disease']
})

###########################################################################################

# --- MAIN EXECUTION ---

def main():
    
    logging.info(f"▶️ STARTING MASTER TABLE GENERATION (EXPERIMENT 5)")
    
    DATA_IDS = list(CONFIG_EXPERIMENT.keys())
    df_avg_results_dict = {}
    df_structure_dict = {}

    for data_id in DATA_IDS:
        
        logging.info(f"  > Processing dataset: {data_id}")
        dataset_results_path = os.path.join(results_dir, data_id, f'results_exp_5_{data_id}.pkl')
        
        if not os.path.exists(dataset_results_path):
            logging.warning(f"  ⚠️ Not found: {dataset_results_path}. Skipping...")
            continue

        # 1. Process Experiment Results
        _, df_avg, _, _ = process_experiment_5_results(
            results_path=dataset_results_path, 
            not_feasible_methods_to_add=NOT_FEASIBLE_METHODS_TO_ADD.get(data_id, []),
            prop_errors_threshold=PROP_ERRORS_THRESHOLD
        )
        df_avg_results_dict[data_id] = df_avg.with_columns(pl.lit(data_id).alias("data_id"))

        # 2. Data Loading & Predictor Routing
        if 'simulation' in data_id:
            simulation_config = SIMULATION_CONFIGS[data_id]
            X, y = generate_simulation(
                **simulation_config,
                random_state=123,
                return_outlier_idx=False
            )
            
            # Safe Polars conversion regardless of backend return type
            if 'pandas' in str(type(X)).lower():
                X = pl.from_pandas(X)
            else:
                cols = [f"X{i+1}" for i in range(X.shape[1])]
                X = pl.DataFrame(X, schema=cols)
                
            y = np.array(y).flatten()

            binary_predictors, multiclass_predictors = ['X5', 'X6'], ['X7', 'X8']
            quant_predictors = [c for c in X.columns if c not in binary_predictors + multiclass_predictors]
            sim_cfg_to_pass = simulation_config

        else:
            processed_data_dir = os.path.join(project_path, 'data', 'processed_data')
            metadata_path = os.path.join(processed_data_dir, f'metadata_{data_id}.json')
            processed_data_path = os.path.join(processed_data_dir, f'{data_id}_processed.parquet')
            
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                
            data = pl.read_parquet(processed_data_path)
            quant_predictors = metadata['quant_predictors']
            binary_predictors = metadata['binary_predictors']
            multiclass_predictors = metadata['multiclass_predictors']
            
            all_cols = quant_predictors + binary_predictors + multiclass_predictors
            X = data.select(all_cols)
            y = data.select(metadata['response']).to_series().to_numpy()
            sim_cfg_to_pass = None

        # 3. Extract Structure Data
        data_structure = get_dataset_structure(
            X, y, data_id, quant_predictors, binary_predictors, multiclass_predictors, sim_cfg_to_pass
        )
        df_structure_dict[data_id] = pl.DataFrame([data_structure])

    # 4. Join and Save Final Table
    if df_avg_results_dict and df_structure_dict:
        df_avg_results_concat = pl.concat(list(df_avg_results_dict.values()), how='vertical')
        df_structure_concat = pl.concat(list(df_structure_dict.values()), how='vertical')
        
        df_master = df_avg_results_concat.join(df_structure_concat, on='data_id', how='left')
        logging.info(f"  📊 Master table shape: {df_master.shape}")

        try:
            df_master.write_parquet(results_save_path)
            logging.info(f"  💾 Final file saved: {results_save_path}")
        except Exception as e:
            logging.error(f"  ❌ Failed to save final file: {e}")
            sys.exit(1)
    else:
        logging.error("  ❌ No data was processed. Master table not created.")
        sys.exit(1)

    logging.info("✅ MASTER TABLE PIPELINE FINISHED SUCCESSFULLY")

###########################################################################################

if __name__ == "__main__":
    main()