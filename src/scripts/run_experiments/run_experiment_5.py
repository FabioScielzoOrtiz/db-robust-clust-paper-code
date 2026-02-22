###########################################################################################
# --- IMPORTS ---
import os
import sys
import random
import json
import pickle
import logging
import argparse
import polars as pl
from tqdm import tqdm

###########################################################################################
# --- LOGGING CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

###########################################################################################
# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Run Experiment 5 Simulations")
parser.add_argument('--data_id', type=str, required=True, help="ID of the simulation data configuration (e.g., 'simulation_1')")
parser.add_argument('--force', action='store_true', help="Force execution for all models, ignoring existing results.")
args = parser.parse_args()

DATA_ID = args.data_id
FORCE_EXECUTION = args.force

###########################################################################################
# --- PATH CONFIGURATION ---
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..', '..')
processed_data_dir = os.path.join(project_path, 'data', 'processed_data')
results_dir = os.path.join(project_path, 'results', 'experiment_5', DATA_ID)
sys.path.append(project_path)

###########################################################################################
# --- CUSTOM IMPORTS ---
from src.utils.experiments_run_utils import (
    split_list_in_chunks,
    get_clustering_models_experiment_5,
    get_mixed_distances_names,
    make_experiment_5
)
from src.utils.simulations_utils import generate_simulation

from config.config_experiment_5 import (
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
    Main execution flow of the Experiment 5 pipeline with Incremental Model Update Capability.
    Handles missing chunk files by reconstructing history from the merged file.
    """
    logging.info(f"▶️ STARTING EXPERIMENT 5 FOR DATA_ID: {DATA_ID}")
    logging.info(f"▶️ FORCE EXECUTION: {FORCE_EXECUTION}")

    # 0. Validate Configuration
    if DATA_ID not in CONFIG_EXPERIMENT:
        logging.error(f"DATA_ID '{DATA_ID}' not found in experiment configuration file.")
        sys.exit(1)
    
    experiment_config = CONFIG_EXPERIMENT[DATA_ID]
 
    # 1. Setup Environment
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # 2. Define All Possible Models (Names)
    quant_distances_names = ['robust_mahalanobis']
    binary_distances_names = ['jaccard', 'sokal']
    multiclass_distances_names = ['hamming']
    robust_method = ['MAD', 'trimmed', 'winsorized']

    mixed_distances_names = get_mixed_distances_names(
        quant_distances_names, 
        binary_distances_names, 
        multiclass_distances_names, 
        robust_method
    )
    
    # Instanciamos dummy para obtener nombres
    dummy_models = get_clustering_models_experiment_5(
        random_state=42,
        experiment_config=experiment_config,
        mixed_distances_names=mixed_distances_names
    )
    all_model_names = set(dummy_models.keys())
    logging.info(f" -> Total defined models available: {len(all_model_names)}")

    # =========================================================================
    # 3. Load Global Merged History (To handle missing chunks)
    # =========================================================================
    final_filename = f'results_exp_5_{DATA_ID}.pkl'
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
    
    # 5. Data Preparation (Load Real Data OR Prepare for Simulation)
    simulation_names = list(SIMULATION_CONFIGS.keys())
    is_simulation = DATA_ID in simulation_names
    X_real, y_real = None, None

    if not is_simulation:
        logging.info("STEP 3: Loading Real Data...")
        try:
            metadata_file_name = f"metadata_{DATA_ID}.json"
            processed_data_file_name = f"{DATA_ID}_processed.parquet"
            metadata_file_path = os.path.join(processed_data_dir, metadata_file_name)
            processed_data_file_path = os.path.join(processed_data_dir, processed_data_file_name)

            with open(metadata_file_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            df = pl.read_parquet(processed_data_file_path)
            X_real = df[metadata['quant_predictors'] + metadata['binary_predictors'] + metadata['multiclass_predictors']]
            y_real = df[metadata['response']]
            
            experiment_config.update({
                'p1': metadata['p1'], 'p2': metadata['p2'], 'p3': metadata['p3'],
                'n_clusters': metadata['n_clusters']
            })
        except Exception as e:
            logging.error(f"Failed to fetch real data: {e}")
            sys.exit(1)
    else:
        logging.info("STEP 3: Simulation mode configured.")

    # =========================================================================
    # 6. Process Chunks Loop (Reconstruction + Update)
    # =========================================================================
    logging.info("STEP 4: Processing Chunks (Recovering -> Updating -> Saving)...")
    
    chunks_processed_count = 0

    for chunk_id, chunk_seeds in enumerate(tqdm(all_chunks, desc='Processing Chunks')):
        
        chunk_filename = f'results_exp_5_{DATA_ID}_chunk_{chunk_id}.pkl'
        chunk_path = os.path.join(results_dir, chunk_filename)
        
        chunk_results = {}
        chunk_needs_save = False # Flag para saber si guardamos a disco

        # --- A. Estrategia de Carga: Archivo VS Global Merge ---
        if os.path.exists(chunk_path) and not FORCE_EXECUTION:
            # Opción 1: El archivo chunk existe, es la fuente más confiable
            try:
                with open(chunk_path, 'rb') as f:
                    chunk_results = pickle.load(f)
            except Exception as e:
                logging.error(f"Error loading chunk {chunk_id}, will try to recover from global merge: {e}")
                chunk_results = {}

        # Opción 2: Si el chunk estaba vacío o no existía, intentamos RECUPERAR del Global Merge
        # Esto soluciona el caso de "archivos chunks borrados"
        if not chunk_results and not FORCE_EXECUTION:
            recovered_count = 0
            for seed in chunk_seeds:
                if seed in GLOBAL_MERGED_DATA:
                    chunk_results[seed] = GLOBAL_MERGED_DATA[seed]
                    recovered_count += 1
            
            if recovered_count > 0:
                # Si recuperamos datos del merge, marcamos para guardar, 
                # así "regeneramos" el archivo chunk físico.
                chunk_needs_save = True 

        # --- B. Iterar sobre semillas del Chunk ---
        for random_state in chunk_seeds:
            
            # 1. Determinar qué modelos faltan
            current_seed_results = chunk_results.get(random_state, {})
            
            if FORCE_EXECUTION:
                existing_models_for_seed = set()
            else:
                if 'time' in current_seed_results:
                    existing_models_for_seed = set(current_seed_results['time'].keys())
                else:
                    existing_models_for_seed = set()

            models_to_run_names = all_model_names - existing_models_for_seed
            
            # Si no faltan modelos, pasamos a la siguiente semilla
            if not models_to_run_names:
                continue 

            # 2. Generar/Obtener Data (Solo si vamos a correr algo)
            try:
                if is_simulation:
                    simulation_config = SIMULATION_CONFIGS[DATA_ID]
                    X, y = generate_simulation(
                        **simulation_config,
                        random_state=random_state,
                        return_outlier_idx=False
                    )
                    models_random_state = experiment_config['random_state']
                else:
                    X, y = X_real, y_real
                    models_random_state = random_state

                # 3. Instanciar modelos necesarios
                all_models_instances = get_clustering_models_experiment_5(
                    random_state=models_random_state,
                    experiment_config=experiment_config,
                    mixed_distances_names=mixed_distances_names
                )
                
                models_subset = {name: model for name, model in all_models_instances.items() if name in models_to_run_names}
                
                if not models_subset:
                    continue

                # 4. Ejecutar nuevos modelos
                new_results = make_experiment_5(
                    X=X, y=y, models=models_subset,
                    score_metric=experiment_config['score_metric']
                )

                # 5. Merge inteligente a nivel de semilla
                if random_state not in chunk_results:
                    chunk_results[random_state] = new_results
                else:
                    # Update recursivo de diccionarios (time, ARI, etc.)
                    for metric_key, model_dict in new_results.items():
                        if metric_key not in chunk_results[random_state]:
                            chunk_results[random_state][metric_key] = {}
                        chunk_results[random_state][metric_key].update(model_dict)
                
                chunk_needs_save = True

            except Exception as e:
                logging.error(f"Error processing seed {random_state} in chunk {chunk_id}: {e}")
                continue

        # --- C. Guardar Chunk (Si hubo cambios o recuperación) ---
        # Guardamos si corrimos modelos nuevos O si recuperamos datos del merge (para restaurar el archivo)
        if chunk_needs_save:
            try:
                with open(chunk_path, 'wb') as f:
                    pickle.dump(chunk_results, f)
                chunks_processed_count += 1
            except Exception as e:
                logging.error(f"Failed to save chunk {chunk_id}: {e}")

    logging.info(f" -> Chunks updated/restored: {chunks_processed_count}")

    # =========================================================================
    # 7. Final Consolidation (Merge Chunk Files)
    # =========================================================================
    logging.info("STEP 5: Consolidating final results...")

    # Reiniciamos el diccionario final para reconstruirlo limpiamente desde los chunks (ahora actualizados)
    final_merged_results = {}
    missing_chunks_during_merge = 0
    n_total = len(all_chunks)
    
    for chunk_id in range(n_total):
        chunk_filename = f'results_exp_5_{DATA_ID}_chunk_{chunk_id}.pkl'
        chunk_path = os.path.join(results_dir, chunk_filename)
        
        if not os.path.exists(chunk_path):
            # Si a estas alturas no existe, es que falló la recuperación y la generación
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

    # 8. Save Final Merged Results
    try:
        with open(final_save_path, 'wb') as f:
            pickle.dump(final_merged_results, f)
        logging.info(f"   💾 Final file updated: {final_filename}")
    except Exception as e:
        logging.error(f"   ❌ Failed to save final file: {e}")
        sys.exit(1)

    logging.info("✅ EXPERIMENT PIPELINE FINISHED")

if __name__ == "__main__":
    main()