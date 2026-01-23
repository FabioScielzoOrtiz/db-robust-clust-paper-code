###########################################################################################

# --- IMPORTS ---

import os
import sys
import random
import pickle
import logging
import argparse
from sklearn.metrics import accuracy_score

###########################################################################################

# --- LOGGING CONFIGURATION ---

# Configuramos el formato para que se vea igual que tu script de ejemplo
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

###########################################################################################

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Run Experiment 1 Simulations")
parser.add_argument('--data_id', type=str, required=True, help="ID of the simulation data configuration (e.g., 'simulation_1')")
args = parser.parse_args()

DATA_ID = args.data_id

###########################################################################################

# --- PATH CONFIGURATION ---

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..')
results_folder_path = os.path.join(project_path, 'results', 'experiment_1', DATA_ID)

sys.path.append(project_path)

###########################################################################################

# --- CUSTOM IMPORTS ---

from src.experiments_utils import (
    split_list_in_chunks,
    make_experiment_1
)
from src.simulations_utils import generate_simulation

from config.config_experiment_1 import (
    CONFIG_EXPERIMENT_1, 
    EXPERIMENT_RANDOM_STATE,
    N_REALIZATIONS, 
    CHUNK_SIZE
)
from config.config_simulations import SIMULATION_CONFIGS


###########################################################################################

# --- MAIN EXECUTION ---

def main():
    """
    Main execution flow of the Experiment 1 pipeline with Resume Capability.
    """
    logging.info(f"▶️ STARTING EXPERIMENT 1 FOR DATA_ID: {DATA_ID}")

    # 0. Validate Configuration
    if DATA_ID not in CONFIG_EXPERIMENT_1:
        logging.error(f"DATA_ID '{DATA_ID}' not found in experiment configuration file.")
        sys.exit(1)
    
    experiment_config = CONFIG_EXPERIMENT_1[DATA_ID]
 
    # 1. Setup Environment
    logging.info("STEP 1: Setting up environment and directories...")
    if not os.path.exists(results_folder_path):
        logging.info(f" -> Creating output directory: {results_folder_path}")
        os.makedirs(results_folder_path, exist_ok=True)
    else:
        logging.info(f" -> Output directory already exists: {results_folder_path}")

    # 2. Prepare Random States & Identify Missing Chunks (ANTES DE GENERAR DATOS)
    logging.info("STEP 2: Checking existing work...")
    
    random.seed(EXPERIMENT_RANDOM_STATE)
    random_state_list = random.sample(range(N_REALIZATIONS * 1000), N_REALIZATIONS)
    # Generamos la lista completa de chunks teóricos
    all_chunks = split_list_in_chunks(random_state_list, chunk_size=CHUNK_SIZE)
    
    # Filtramos: ¿Qué chunks faltan realmente?
    chunks_to_process = []
    for chunk_id, chunk_seeds in enumerate(all_chunks):
        chunk_filename = f'results_exp_1_{DATA_ID}_chunk_{chunk_id}.pkl'
        chunk_path = os.path.join(results_folder_path, chunk_filename)
        
        if not os.path.exists(chunk_path):
            chunks_to_process.append((chunk_id, chunk_seeds))
    
    n_total = len(all_chunks)
    n_missing = len(chunks_to_process)
    n_existing = n_total - n_missing

    logging.info(f" -> Total chunks required: {n_total}")
    logging.info(f" -> Already generated: {n_existing}")
    logging.info(f" -> Remaining to process: {n_missing}")

    # 3. Data Generation (Solo si hace falta procesar algo)
    # Si todo está generado, nos saltamos la generación de datos para ahorrar tiempo/RAM
    X, y = None, None
    
    if n_missing > 0:
        logging.info(f"STEP 3: Fetching Data...")
        try:
            is_simulation = DATA_ID in SIMULATION_CONFIGS
            if is_simulation:
                simulation_config = SIMULATION_CONFIGS[DATA_ID]
                X, y = generate_simulation(
                    **simulation_config,
                    random_state=EXPERIMENT_RANDOM_STATE,
                    return_outlier_idx=False
                )
            else: # real data
                data_filename = f'{DATA_ID}_processed.pkl'
                data_file_path = os.path.join(project_path, 'data', 'processed_data', data_filename)
                with open(data_file_path, "rb") as f:
                    d = pickle.load(f)
                print(d)
                experiment_config.update({
                    'p1': d['p1'],
                    'p2': d['p2'],
                    'p3': d['p3'],
                    'n_clusters': d['n_clusters']
                })
                
                X, y = d['X'], d['y']

            logging.info(f" -> Data fetched successfully. Shape: {X.shape}")
        except Exception as e:
            logging.error(f"Failed to generate simulation data: {e}")
            sys.exit(1)
    else:
        logging.info("STEP 3: Skipping data generation (All chunks exist).")

    # 4. Run Experiments Loop (Solo sobre los faltantes)
    if n_missing > 0:
        logging.info("STEP 4: Running experiments loop for missing chunks...")
        
        # Iteramos solo sobre la lista filtrada 'chunks_to_process'
        for chunk_id, random_state_chunk in chunks_to_process:
            results = {}
            for random_state in random_state_chunk:
                try:
                    results[random_state] = make_experiment_1(
                        **experiment_config,
                        X=X, 
                        y=y,
                        random_state=random_state,
                        metric=accuracy_score
                    )
                except Exception as e:
                    logging.error(f"Error in chunk {chunk_id}, random_state {random_state}: {e}")
                    # Si falla uno, rompemos el bucle inmediatamente para no perder tiempo.
                    # Al hacer break, 'results' quedará incompleto.
                    break 

            # --- LÓGICA DE GUARDADO ---
            # Verificamos si tenemos TODOS los resultados esperados.
            # Si hubo un break o error, len(results) será menor que len(random_state_chunk)
            if len(results) == len(random_state_chunk):
                results_filename = f'results_exp_1_{DATA_ID}_chunk_{chunk_id}.pkl'
                results_save_path = os.path.join(results_folder_path, results_filename)
                try:
                    with open(results_save_path, 'wb') as f:
                        pickle.dump(results, f)
                except Exception as e:
                    logging.error(f"Failed to save results for chunk {chunk_id}: {e}")
            else:
                logging.warning(f"   ⚠️  Skipping save for Chunk {chunk_id}: Incomplete results ({len(results)}/{len(random_state_chunk)}) due to errors.")
    else:
        logging.info("STEP 4: Nothing to process. Proceeding to merge.")

    # 5. Consolidacion inteligente
    logging.info("STEP 5: Consolidating and merging results...")

    # Definimos la ruta final AHORA para verificar si ya existe
    final_filename = f'results_exp_1_{DATA_ID}.pkl'
    final_save_path = os.path.join(results_folder_path, final_filename)

    # --- CONDICIÓN DE SALIDA TEMPRANA ---
    # Si no hubo chunks nuevos que procesar Y el archivo final ya existe: no hacemos nada.
    if n_missing == 0 and os.path.exists(final_save_path):
        logging.info(f" -> No new chunks generated and final file '{final_filename}' already exists.")
        logging.info(" -> Skipping merge process to save time.")
        logging.info("✅ EXPERIMENT PIPELINE FINISHED SUCCESSFULLY")
        return
    # ------------------------------------

    # Si llegamos aquí, es porque o bien generamos chunks nuevos, 
    # o bien el archivo final se borró y hay que regenerarlo.
    
    final_merged_results = {}
    missing_chunks_during_merge = 0

    # Iteramos sobre range(n_total) para asegurar que unimos todo (viejo + nuevo)
    for chunk_id in range(n_total):
        
        chunk_filename = f'results_exp_1_{DATA_ID}_chunk_{chunk_id}.pkl'
        chunk_path = os.path.join(results_folder_path, chunk_filename)
        
        if not os.path.exists(chunk_path):
            logging.warning(f"   ⚠️ Chunk file missing during merge: {chunk_filename}")
            missing_chunks_during_merge += 1
            continue

        try:
            with open(chunk_path, 'rb') as f:
                chunk_data = pickle.load(f)
                final_merged_results.update(chunk_data)
        except Exception as e:
            logging.error(f"   ❌ Error loading chunk {chunk_id}: {e}")
            missing_chunks_during_merge += 1

    # Reporte de consolidación
    total_loaded = len(final_merged_results)
    logging.info(f" -> Merged {n_total - missing_chunks_during_merge}/{n_total} chunks.")
    logging.info(f" -> Total realizations captured: {total_loaded}")

    if total_loaded == 0:
        logging.error("No results were loaded. Exiting before saving.")
        sys.exit(1)

    # 6. Save Final Merged Results
    logging.info("STEP 6: Saving final consolidated file...")

    try:
        with open(final_save_path, 'wb') as f:
            pickle.dump(final_merged_results, f)
        logging.info(f"   💾 Final file saved: {final_filename}")
    except Exception as e:
        logging.error(f"   ❌ Failed to save final merged file: {e}")
        sys.exit(1)

    logging.info("✅ EXPERIMENT PIPELINE FINISHED SUCCESSFULLY")

###########################################################################################

if __name__ == "__main__":
    main()