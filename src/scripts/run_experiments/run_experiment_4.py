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
parser = argparse.ArgumentParser(description="Run Experiment 4 Simulations")
parser.add_argument('--data_id', type=str, required=True, help="ID of the simulation data configuration (e.g., 'simulation_size_1')")
parser.add_argument('--force', action='store_true', help="Force execution. By default, forces all models. Use with --force_models to restrict.")
# NUEVO: Argumento que acepta una lista separada por espacios de nombres de modelos
parser.add_argument('--force_models', nargs='+', type=str, help="List of specific models to force re-execution (e.g., --force_models KMedoids SubKmeans)")
args = parser.parse_args()

DATA_ID = args.data_id
FORCE_EXECUTION = args.force
FORCE_MODELS_LIST = args.force_models # Será None si no se especifica, o una lista ['modelo1', 'modelo2']

###########################################################################################
# --- PATH CONFIGURATION ---
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..', '..')
processed_data_dir = os.path.join(project_path, 'data', 'processed_data')
results_dir = os.path.join(project_path, 'results', 'experiment_4', DATA_ID)
sys.path.append(project_path)

###########################################################################################
# --- CUSTOM IMPORTS ---
from src.utils.experiments_run_utils import (
    split_list_in_chunks,
    get_clustering_models_experiment_4,
    get_mixed_distances_names,
    make_experiment_4
)
from src.utils.simulations_utils import generate_simulation

from config.config_experiment_4 import (
    CONFIG_EXPERIMENT, 
    EXPERIMENT_RANDOM_STATE,
    N_REALIZATIONS, 
    CHUNK_SIZE,
    MAX_DURATION_MINS
)
from config.config_simulations import SIMULATION_CONFIGS

###########################################################################################
# --- MAIN EXECUTION ---

def main():
    """
    Main execution flow of the Experiment 4 pipeline with Incremental Model Update Capability.
    """
    logging.info(f"▶️ STARTING Experiment 4 FOR DATA_ID: {DATA_ID}")
    logging.info(f"▶️ FORCE EXECUTION: {FORCE_EXECUTION}")
    if FORCE_EXECUTION and FORCE_MODELS_LIST:
        logging.info(f"▶️ FORCING SPECIFIC MODELS: {FORCE_MODELS_LIST}")

    # 0. Validate Configuration
    if DATA_ID not in CONFIG_EXPERIMENT:
        logging.error(f"DATA_ID '{DATA_ID}' not found in experiment configuration file.")
        sys.exit(1)
    
    experiment_config = CONFIG_EXPERIMENT[DATA_ID]
    logging.info('Experiment config loaded...')
    logging.info(f'> Experiment Config: {experiment_config}')

    # 1. Setup Environment
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # 2. Define All Possible Models (Names)
    QUANT_DISTANCE_NAMES = ['robust_mahalanobis']
    BINARY_DISTANCE_NAMES = ['jaccard', 'sokal']
    MULTICLASS_DISTANCE_NAMES = ['hamming']
    ROBUST_METHOD_NAMES = ['MAD', 'trimmed', 'winsorized']

    mixed_distances_names = get_mixed_distances_names(
        QUANT_DISTANCE_NAMES, 
        BINARY_DISTANCE_NAMES, 
        MULTICLASS_DISTANCE_NAMES, 
        ROBUST_METHOD_NAMES
    )
    
    dummy_models = get_clustering_models_experiment_4(
        random_state=42,
        experiment_config=experiment_config,
        mixed_distances_names=mixed_distances_names
    )
    all_model_names = set(dummy_models.keys())
    logging.info(f" -> Total defined models available: {len(all_model_names)}")

    # =========================================================================
    # 3. Load Global Merged History 
    # =========================================================================
    final_filename = f'results_exp_4_{DATA_ID}.pkl'
    final_save_path = os.path.join(results_dir, final_filename)
    
    GLOBAL_MERGED_DATA = {} 
    
    # MODIFICADO: Ahora SIEMPRE cargamos el historial si existe, incluso si hay force.
    # Esto preserva los modelos que NO queremos forzar.
    if os.path.exists(final_save_path):
        logging.info("STEP 2: Loading existing merged file (as source of truth)...")
        try:
            with open(final_save_path, 'rb') as f:
                GLOBAL_MERGED_DATA = pickle.load(f)
            logging.info(f" -> Loaded history for {len(GLOBAL_MERGED_DATA)} realizations.")
        except Exception as e:
            logging.warning(f" -> Could not read existing merged file ({e}). Starting fresh.")
    else:
        logging.info("STEP 2: No previous merged file found. Starting fresh.")

    # =========================================================================
    # 4. Prepare Random States & Chunks (N_REALIZATIONS Protection & Extension)
    # =========================================================================
    if GLOBAL_MERGED_DATA:
        existing_seeds = list(GLOBAL_MERGED_DATA.keys())
        num_existing = len(existing_seeds)
        
        if N_REALIZATIONS < num_existing:
            # PROTECCIÓN: Evitamos pérdida de datos si el usuario baja el número por error
            logging.error(f"❌ PELIGRO: N_REALIZATIONS ({N_REALIZATIONS}) es menor que las realizaciones guardadas ({num_existing}).")
            logging.error("Ejecución detenida para prevenir pérdida de datos. Sube N_REALIZATIONS en tu config, o borra los archivos a mano si realmente quieres empezar de cero.")
            sys.exit(1)
            
        elif N_REALIZATIONS == num_existing:
            # Todo cuadra, usamos las mismas semillas
            logging.info(f"STEP 2.5: Reusing {num_existing} exact seeds from existing global data.")
            random_state_list = existing_seeds
            
        else:
            # Hay que añadir NUEVAS realizaciones a las ya existentes
            new_seeds_needed = N_REALIZATIONS - num_existing
            logging.info(f"STEP 2.5: Found {num_existing} existing seeds. Generating {new_seeds_needed} NEW seeds to reach {N_REALIZATIONS}.")
            
            random.seed(EXPERIMENT_RANDOM_STATE)
            # Generamos un pool muy grande para asegurarnos de no repetir semillas
            pool_size = N_REALIZATIONS * 1000
            all_possible_seeds = random.sample(range(pool_size), pool_size)
            
            new_seeds = []
            for s in all_possible_seeds:
                if s not in existing_seeds:
                    new_seeds.append(s)
                if len(new_seeds) == new_seeds_needed:
                    break
            
            # La lista final será: primero las antiguas (para no reordenar chunks), luego las nuevas
            random_state_list = existing_seeds + new_seeds
            
    else:
        # Primera ejecución desde cero
        logging.info("STEP 2.5: No previous data found. Generating seeds from scratch.")
        random.seed(EXPERIMENT_RANDOM_STATE)
        random_state_list = random.sample(range(N_REALIZATIONS * 1000), N_REALIZATIONS)
        
    all_chunks = split_list_in_chunks(random_state_list, chunk_size=CHUNK_SIZE)
    
    # 5. Data Preparation
    simulation_names = list(SIMULATION_CONFIGS.keys())
    is_simulation = DATA_ID in simulation_names
    X_real, y_real = None, None

    if is_simulation:
        simulation_config = SIMULATION_CONFIGS[DATA_ID]
        logging.info("STEP 3: Simulation Config Loaded...")
        logging.info(f'> Simulation Config: {simulation_config}')
        
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

        except Exception as e:
            logging.error(f"Failed to fetch real data: {e}")
            sys.exit(1)
    else:
        logging.info("STEP 3: Simulation mode configured.")

    # =========================================================================
    # 6. Process Chunks Loop
    # =========================================================================
    logging.info("STEP 4: Processing Chunks (Recovering -> Updating -> Saving)...")
    chunks_processed_count = 0
    total_chunks = len(all_chunks)

    for chunk_id, chunk_seeds in enumerate(tqdm(all_chunks, desc='Processing Chunks')):
        
        logging.info(f"\n" + "="*50)
        logging.info(f"📦 STARTING CHUNK {chunk_id}/{total_chunks - 1} | Contains {len(chunk_seeds)} seeds")
        logging.info(f"="*50)

        chunk_filename = f'results_exp_4_{DATA_ID}_chunk_{chunk_id}.pkl'
        chunk_path = os.path.join(results_dir, chunk_filename)
        
        chunk_results = {}
        chunk_needs_save = False 

        # 1. Intentamos cargar el chunk si existe
        if os.path.exists(chunk_path):
            try:
                with open(chunk_path, 'rb') as f:
                    chunk_results = pickle.load(f)
                logging.info(f"  -> 📂 Loaded existing chunk file with {len(chunk_results)} seeds.")
            except Exception as e:
                logging.error(f"  -> ❌ Error loading chunk {chunk_id}: {e}")
                chunk_results = {}

        # 2. NUEVO: Recuperación a nivel de SEMILLA (Super importante)
        for seed in chunk_seeds:
            if seed not in chunk_results and seed in GLOBAL_MERGED_DATA:
                chunk_results[seed] = GLOBAL_MERGED_DATA[seed]
                chunk_needs_save = True 

        # --- B. Iterar sobre semillas del Chunk ---
        for random_state in chunk_seeds:
            current_seed_results = chunk_results.get(random_state, {})
            
            # Lógica para determinar qué correr
            existing_models_for_seed = set(current_seed_results.get('time', {}).keys())

            if FORCE_EXECUTION:
                if FORCE_MODELS_LIST:
                    existing_models_for_seed = existing_models_for_seed - set(FORCE_MODELS_LIST)
                else:
                    existing_models_for_seed = set()

            # Modelos finales a correr
            models_to_run_names = all_model_names - existing_models_for_seed
            
            if not models_to_run_names:
                logging.info(f"    ✅ Seed {random_state}: All models already completed. Skipping.")
                continue 

            logging.info(f"    ▶️ Seed {random_state}: Missing {len(models_to_run_names)} models. Preparing execution...")

            # 2. Generar/Obtener Data
            try:
                if is_simulation:
                    X, y = generate_simulation(
                        **simulation_config,
                        random_state=random_state,
                        return_outlier_idx=False
                    )
                    models_random_state = EXPERIMENT_RANDOM_STATE
                else:
                    X, y = X_real, y_real
                    models_random_state = random_state

                # 3. Instanciar modelos necesarios
                all_models_instances = get_clustering_models_experiment_4(
                    random_state=models_random_state,
                    experiment_config=experiment_config,
                    mixed_distances_names=mixed_distances_names
                )
                
                models_subset = {name: model for name, model in all_models_instances.items() if name in models_to_run_names}
                
                if not models_subset:
                    continue

                # 4. Ejecutar nuevos modelos
                logging.info(f"       ⚙️ Handing over to make_experiment_4 with {list(models_subset.keys())}...")
                new_results = make_experiment_4(
                    X=X, y=y, models=models_subset,
                    score_metric=experiment_config['score_metric'],
                    max_duration_mins=MAX_DURATION_MINS
                )
                logging.info(f"       ✔️ Execution finished for Seed {random_state}")

                # 5. Merge inteligente a nivel de semilla
                if random_state not in chunk_results:
                    chunk_results[random_state] = new_results
                else:
                    for metric_key, model_dict in new_results.items():
                        if metric_key not in chunk_results[random_state]:
                            chunk_results[random_state][metric_key] = {}
                        chunk_results[random_state][metric_key].update(model_dict)
                
                chunk_needs_save = True

            except Exception as e:
                logging.error(f"       ❌ Error processing seed {random_state} in chunk {chunk_id}: {e}")
                continue

        # --- C. Guardar Chunk ---
        if chunk_needs_save:
            try:
                with open(chunk_path, 'wb') as f:
                    pickle.dump(chunk_results, f)
                chunks_processed_count += 1
                logging.info(f"  💾 Saved updates for chunk {chunk_id}.")
            except Exception as e:
                logging.error(f"  ❌ Failed to save chunk {chunk_id}: {e}")
        else:
            logging.info(f"  ⏭️ No new updates to save for chunk {chunk_id}.")

    logging.info(f"\n -> Total chunks updated/restored: {chunks_processed_count}")

    # =========================================================================
    # 7. Final Consolidation (Merge Chunk Files)
    # =========================================================================
    logging.info("STEP 5: Consolidating final results...")

    final_merged_results = {}
    missing_chunks_during_merge = 0
    n_total = len(all_chunks)
    
    for chunk_id in range(n_total):
        chunk_filename = f'results_exp_4_{DATA_ID}_chunk_{chunk_id}.pkl'
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