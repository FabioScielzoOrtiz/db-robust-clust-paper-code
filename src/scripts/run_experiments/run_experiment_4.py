###########################################################################################
# --- IMPORTS ---
import os
import sys
import random
import pickle
import logging
import argparse
from tqdm import tqdm

###########################################################################################
# --- LOGGING CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

###########################################################################################
# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Run Experiment 4 Simulations")
parser.add_argument('--force', action='store_true', help="Force execution for all models, ignoring existing results.")
args = parser.parse_args()

# Simplificación: Fijamos el DATA_ID a 'simulation_1'
DATA_ID = 'simulation_1'
FORCE_EXECUTION = args.force

###########################################################################################
# --- PATH CONFIGURATION ---
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..', '..')
results_dir = os.path.join(project_path, 'results', 'experiment_4')
sys.path.append(project_path)

###########################################################################################
# --- CUSTOM IMPORTS ---
from src.utils.experiments_utils import (
    split_list_in_chunks,
    get_clustering_models_experiment_4,
    make_experiment_4
)
# Asegúrate de importar desde la config de tu experimento 4
from config.config_experiment_4 import (
    CONFIG_EXPERIMENT, 
    EXPERIMENT_RANDOM_STATE,
    N_REALIZATIONS, 
    CHUNK_SIZE
)
from config.config_simulations import SIMULATION_CONFIGS

###########################################################################################
# --- MAIN EXECUTION ---

def main():
    """
    Main execution flow of the Experiment 4 pipeline.
    Simplified to run only for 'simulation_1' and a fixed set of 3 models.
    """
    logging.info(f"▶️ STARTING EXPERIMENT 4:")
    logging.info(f"▶️ FORCE EXECUTION: {FORCE_EXECUTION}")

    # 0. Validate Configuration
    if DATA_ID not in SIMULATION_CONFIGS:
        logging.error(f"DATA_ID '{DATA_ID}' not found in configuration files.")
        sys.exit(1)
    
    simulation_config = SIMULATION_CONFIGS[DATA_ID]

    # 1. Setup Environment
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # 2. Define All Possible Models (Names)
    dummy_models = get_clustering_models_experiment_4(
        experiment_config=CONFIG_EXPERIMENT,
        random_state=42
    )
    all_model_names = set(dummy_models.keys())
    logging.info(f" -> Total defined models available: {len(all_model_names)}")

    # =========================================================================
    # 3. Load Global Merged History (To handle missing chunks)
    # =========================================================================
    final_filename = f'results_exp_4.pkl'
    final_save_path = os.path.join(results_dir, final_filename)
    
    GLOBAL_MERGED_DATA = {} # Store full history here
    
    if os.path.exists(final_save_path) and not FORCE_EXECUTION:
        logging.info("STEP 2: Loading existing merged file (as source of truth)...")
        try:
            with open(final_save_path, 'rb') as f:
                GLOBAL_MERGED_DATA = pickle.load(f)
            logging.info(f" -> Loaded history for {len(GLOBAL_MERGED_DATA)} realizations from merged file.")
        except Exception as e:
            logging.warning(f" -> Could not read existing merged file ({e}). Starting fresh.")
    else:
        logging.info("STEP 2: No previous merged file found (or Force active). Starting fresh.")

    # 4. Prepare Random States & Chunks
    random.seed(EXPERIMENT_RANDOM_STATE)
    random_state_list = random.sample(range(N_REALIZATIONS * 1000), N_REALIZATIONS)
    all_chunks = split_list_in_chunks(random_state_list, chunk_size=CHUNK_SIZE)

    # =========================================================================
    # 5. Process Chunks Loop (Reconstruction + Update)
    # =========================================================================
    logging.info("STEP 3: Processing Chunks (Recovering -> Updating -> Saving)...")
    
    chunks_processed_count = 0

    for chunk_id, chunk_seeds in enumerate(tqdm(all_chunks, desc='Processing Chunks')):
        
        chunk_filename = f'results_exp_4_chunk_{chunk_id}.pkl'
        chunk_path = os.path.join(results_dir, chunk_filename)
        
        chunk_results = {}
        chunk_needs_save = False 

        # --- A. Carga de Chunk o Recuperación del Merge Global ---
        if os.path.exists(chunk_path) and not FORCE_EXECUTION:
            try:
                with open(chunk_path, 'rb') as f:
                    chunk_results = pickle.load(f)
            except Exception as e:
                logging.error(f"Error loading chunk {chunk_id}, will try to recover from global merge: {e}")
                chunk_results = {}

        if not chunk_results and not FORCE_EXECUTION:
            recovered_count = 0
            for seed in chunk_seeds:
                if seed in GLOBAL_MERGED_DATA:
                    chunk_results[seed] = GLOBAL_MERGED_DATA[seed]
                    recovered_count += 1
            
            if recovered_count > 0:
                chunk_needs_save = True 

        # --- B. Iterar sobre semillas del Chunk ---
        for random_state in chunk_seeds:
            
            # 1. Determinar qué modelos faltan
            current_seed_results = chunk_results.get(random_state, {})
            existing_models_for_seed = set()
            
            if not FORCE_EXECUTION and 'time' in current_seed_results and current_seed_results['time']:
                # Extraemos los modelos ya ejecutados buscando en el primer data_size registrado
                first_data_size = list(current_seed_results['time'].keys())[0]
                existing_models_for_seed = set(current_seed_results['time'][first_data_size].keys())

            models_to_run_names = all_model_names - existing_models_for_seed
            
            # Si no faltan modelos, pasamos a la siguiente semilla
            if not models_to_run_names:
                continue 

            # 2. Ejecutar los modelos faltantes
            try:
                # Instanciar solo los modelos necesarios para esta semilla
                all_models_instances = get_clustering_models_experiment_4(
                    experiment_config=CONFIG_EXPERIMENT,
                    random_state=random_state
                )
                models_subset = {name: model for name, model in all_models_instances.items() if name in models_to_run_names}
                
                if not models_subset:
                    continue

                # Llamamos a la función make_experiment_4 (que internamente genera la data)
                new_results = make_experiment_4(
                    data_sizes=CONFIG_EXPERIMENT['data_sizes'], # Asegúrate de que esto exista en tu config
                    centers=simulation_config['centers'],
                    cluster_std=simulation_config['cluster_std'],
                    n_features=simulation_config['n_features'],
                    outlier_configs=simulation_config['outlier_configs'],
                    random_state=random_state,
                    models=models_subset,
                    score_metric=CONFIG_EXPERIMENT['score_metric']
                )

                # 3. Merge inteligente a nivel de semilla y tamaño de datos
                if random_state not in chunk_results:
                    chunk_results[random_state] = new_results
                else:
                    # Update recursivo para: dict[metric][n_samples][model_name]
                    for metric_key, sizes_dict in new_results.items():
                        if metric_key not in chunk_results[random_state]:
                            chunk_results[random_state][metric_key] = {}
                            
                        for n_samples, models_dict in sizes_dict.items():
                            if n_samples not in chunk_results[random_state][metric_key]:
                                chunk_results[random_state][metric_key][n_samples] = {}
                            chunk_results[random_state][metric_key][n_samples].update(models_dict)
                
                chunk_needs_save = True

            except Exception as e:
                logging.error(f"Error processing seed {random_state} in chunk {chunk_id}: {e}")
                continue

        # --- C. Guardar Chunk ---
        if chunk_needs_save:
            try:
                with open(chunk_path, 'wb') as f:
                    pickle.dump(chunk_results, f)
                chunks_processed_count += 1
            except Exception as e:
                logging.error(f"Failed to save chunk {chunk_id}: {e}")

    logging.info(f" -> Chunks updated/restored: {chunks_processed_count}")

    # =========================================================================
    # 6. Final Consolidation (Merge Chunk Files)
    # =========================================================================
    logging.info("STEP 4: Consolidating final results...")

    final_merged_results = {}
    missing_chunks_during_merge = 0
    n_total = len(all_chunks)
    
    for chunk_id in range(n_total):
        chunk_filename = f'results_exp_4_chunk_{chunk_id}.pkl'
        chunk_path = os.path.join(results_dir, chunk_filename)
        
        if not os.path.exists(chunk_path):
            logging.warning(f"   ⚠️ Chunk file missing: {chunk_filename}")
            missing_chunks_during_merge += 1
            continue

        try:
            with open(chunk_path, 'rb') as f:
                chunk_data = pickle.load(f)
                final_merged_results.update(chunk_data)
        except Exception as e:
            logging.error(f"   ❌ Error loading chunk {chunk_id} for merge: {e}")
            missing_chunks_during_merge += 1

    total_loaded = len(final_merged_results)
    logging.info(f" -> Final Merge: {n_total - missing_chunks_during_merge}/{n_total} chunks.")
    
    if total_loaded == 0:
        logging.error("No results loaded. Exiting.")
        sys.exit(1)

    # 7. Save Final Merged Results
    try:
        with open(final_save_path, 'wb') as f:
            pickle.dump(final_merged_results, f)
        logging.info(f"   💾 Final file updated: {final_filename}")
    except Exception as e:
        logging.error(f"   ❌ Failed to save final file: {e}")
        sys.exit(1)

    logging.info("✅ EXPERIMENT 4 PIPELINE FINISHED")

if __name__ == "__main__":
    main()