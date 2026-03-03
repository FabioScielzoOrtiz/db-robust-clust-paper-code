import os
import sys
import random
import json
import pickle
import logging
import argparse
import polars as pl
from tqdm import tqdm

# --- LOGGING CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Unified Pipeline for Experiment 2 (2a, 2b, full)")
parser.add_argument('--data_id', type=str, required=True, help="ID of the simulation data configuration")
parser.add_argument('--mode', type=str, choices=['2a', '2b', 'full'], required=True, 
                    help="2a: Data varies, 2b: Model varies, full: Both vary")
parser.add_argument('--force', action='store_true', help="Force execution, ignoring previous results.")
args = parser.parse_args()

DATA_ID = args.data_id
MODE = args.mode
FORCE_EXECUTION = args.force

# --- PATH CONFIGURATION ---
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..', '..')
processed_data_dir = os.path.join(project_path, 'data', 'processed_data')
results_dir = os.path.join(project_path, 'results', f'experiment_{MODE}', DATA_ID)

sys.path.append(project_path)

# --- CUSTOM IMPORTS ---
from src.utils.experiments_run_utils import split_list_in_chunks, make_experiment_2
from src.utils.simulations_utils import generate_simulation
from config.config_experiment_2 import (
    CONFIG_EXPERIMENT, EXPERIMENT_RANDOM_STATE, N_REALIZATIONS, CHUNK_SIZE
)
from config.config_simulations import SIMULATION_CONFIGS

def main():
    logging.info(f"▶️ STARTING EXPERIMENT 2 - MODE: {MODE} | DATA_ID: {DATA_ID}")
    
    if DATA_ID not in CONFIG_EXPERIMENT:
        logging.error(f"DATA_ID '{DATA_ID}' not found in configuration.")
        sys.exit(1)
    
    experiment_config = CONFIG_EXPERIMENT[DATA_ID]
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # =========================================================================
    # 2. Load Global Merged History (Source of Truth)
    # =========================================================================
    final_filename = f'results_exp_{MODE}_{DATA_ID}.pkl'
    final_save_path = os.path.join(results_dir, final_filename)
    GLOBAL_MERGED_DATA = {} 
    
    if os.path.exists(final_save_path):
        logging.info(f"STEP 2: Loading existing merged file for mode {MODE}...")
        try:
            with open(final_save_path, 'rb') as f:
                GLOBAL_MERGED_DATA = pickle.load(f)
            logging.info(f" -> Loaded history for {len(GLOBAL_MERGED_DATA)} realizations.")
        except Exception as e:
            logging.warning(f" -> Could not read merged file ({e}). Starting fresh.")

    # =========================================================================
    # 3. Prepare Random States & Chunks
    # =========================================================================
    if GLOBAL_MERGED_DATA:
        existing_seeds = list(GLOBAL_MERGED_DATA.keys())
        num_existing = len(existing_seeds)
        
        if N_REALIZATIONS < num_existing:
            logging.error(f"❌ PELIGRO: N_REALIZATIONS ({N_REALIZATIONS}) < existentes ({num_existing}).")
            sys.exit(1)
        elif N_REALIZATIONS == num_existing:
            random_state_list = existing_seeds
        else:
            new_seeds_needed = N_REALIZATIONS - num_existing
            random.seed(EXPERIMENT_RANDOM_STATE)
            pool_size = N_REALIZATIONS * 1000
            all_possible_seeds = random.sample(range(pool_size), pool_size)
            new_seeds = [s for s in all_possible_seeds if s not in existing_seeds][:new_seeds_needed]
            random_state_list = existing_seeds + new_seeds
    else:
        random.seed(EXPERIMENT_RANDOM_STATE)
        random_state_list = random.sample(range(N_REALIZATIONS * 1000), N_REALIZATIONS)
        
    all_chunks = split_list_in_chunks(random_state_list, chunk_size=CHUNK_SIZE)
    
    # =========================================================================
    # 4. Data Preparation (Real Data Optimization)
    # =========================================================================
    simulation_names = list(SIMULATION_CONFIGS.keys())
    is_simulation = DATA_ID in simulation_names
    X_real, y_real = None, None

    if not is_simulation:
        logging.info("STEP 3: Loading Real Data...")
        try:
            metadata_file_path = os.path.join(processed_data_dir, f"metadata_{DATA_ID}.json")
            processed_data_file_path = os.path.join(processed_data_dir, f"{DATA_ID}_processed.parquet")
            with open(metadata_file_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            df = pl.read_parquet(processed_data_file_path)
            X_real = df[metadata['quant_predictors'] + metadata['binary_predictors'] + metadata['multiclass_predictors']]
            y_real = df[metadata['response']]
            experiment_config.update({'p1': metadata['p1'], 'p2': metadata['p2'], 'p3': metadata['p3'], 'n_clusters': metadata['n_clusters']})
        except Exception as e:
            logging.error(f"Failed to fetch real data: {e}"); sys.exit(1)

    # =========================================================================
    # 5. Process Chunks Loop
    # =========================================================================
    logging.info(f"STEP 4: Processing Chunks for Mode {MODE}...")
    chunks_processed_count = 0

    for chunk_id, chunk_seeds in enumerate(tqdm(all_chunks, desc=f'Mode {MODE} Chunks')):
        chunk_filename = f'results_exp_{MODE}_{DATA_ID}_chunk_{chunk_id}.pkl'
        chunk_path = os.path.join(results_dir, chunk_filename)
        chunk_results = {}
        chunk_needs_save = False 
        
        # --- NUEVA LÓGICA DE SALTO DE CHUNK COMPLETO ---
        if os.path.exists(chunk_path) and not FORCE_EXECUTION:
            try:
                with open(chunk_path, 'rb') as f:
                    chunk_results = pickle.load(f)
                
                # Verificamos si todas las semillas del chunk ya están procesadas
                if all(seed in chunk_results for seed in chunk_seeds):
                    logging.info(f"⏭️  Chunk {chunk_id}: Already complete on disk. Skipping.")
                    continue
                else:
                    logging.info(f"📦 Chunk {chunk_id}: Partial data found. Resuming missing seeds.")
            except Exception as e:
                logging.error(f"Error loading chunk {chunk_id}: {e}. Will attempt to re-run.")
                chunk_results = {}

        # Si llegamos aquí, es que el chunk no existe, está incompleto o FORCE_EXECUTION es True
        
        # 5.1 Recuperar de Global (si no estaba en el pkl del chunk)
        for seed in chunk_seeds:
            if seed not in chunk_results and seed in GLOBAL_MERGED_DATA:
                chunk_results[seed] = GLOBAL_MERGED_DATA[seed]
                chunk_needs_save = True 

        # 5.2 Iterar sobre semillas del Chunk
        for rs_loop in chunk_seeds:
            # Doble check por si se recuperó del Global o estaba en el pkl parcial
            if rs_loop in chunk_results and not FORCE_EXECUTION:
                continue 

            # --- LÓGICA DE CONTROL DE SEMILLAS ---
            if MODE == '2a':    # Muestreo: Dato varía, Modelo fijo
                data_seed, model_seed = rs_loop, EXPERIMENT_RANDOM_STATE
            elif MODE == '2b':  # Estabilidad: Dato fijo, Modelo varía
                data_seed, model_seed = EXPERIMENT_RANDOM_STATE, rs_loop
            else:               # Full: Ambos varían
                data_seed, model_seed = rs_loop, rs_loop

            logging.info(f"📋 Experiment Random Seed: {EXPERIMENT_RANDOM_STATE} | Data Random Seed: {data_seed} | Models Random Seed: {model_seed}")

            try:
                if is_simulation:
                    X, y = generate_simulation(**SIMULATION_CONFIGS[DATA_ID], random_state=data_seed, return_outlier_idx=False)
                else:
                    X, y = X_real, y_real

                new_results = make_experiment_2(**experiment_config, X=X, y=y, random_state=model_seed)
                chunk_results[rs_loop] = new_results
                chunk_needs_save = True
            except Exception as e:
                logging.error(f"Error seed {rs_loop} in mode {MODE}: {e}")
                continue

        # Guardar solo si hubo cambios o ejecuciones nuevas
        if chunk_needs_save:
            with open(chunk_path, 'wb') as f:
                pickle.dump(chunk_results, f)
            chunks_processed_count += 1

    # =========================================================================
    # 6. Final Consolidation
    # =========================================================================
    logging.info("STEP 5: Consolidating final results...")
    final_merged_results = {}
    for chunk_id in range(len(all_chunks)):
        chunk_path = os.path.join(results_dir, f'results_exp_{MODE}_{DATA_ID}_chunk_{chunk_id}.pkl')
        if os.path.exists(chunk_path):
            with open(chunk_path, 'rb') as f:
                final_merged_results.update(pickle.load(f))
    
    with open(final_save_path, 'wb') as f:
        pickle.dump(final_merged_results, f)

    logging.info(f"✅ EXPERIMENT {MODE} FINISHED. Total realizations: {len(final_merged_results)}")

if __name__ == "__main__":
    main()