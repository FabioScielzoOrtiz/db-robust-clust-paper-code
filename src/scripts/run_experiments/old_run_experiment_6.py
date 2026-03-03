# --- IMPORTS ---
import os
import sys
import pickle
import logging
import random
from tqdm import tqdm

###########################################################################################
# --- LOGGING CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

###########################################################################################
# --- PATH CONFIGURATION ---
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..', '..')
results_dir = os.path.join(project_path, 'results', 'experiment_6')
results_save_path = os.path.join(results_dir, 'results_exp_6.pkl')

os.makedirs(results_dir, exist_ok=True)
sys.path.append(project_path)

###########################################################################################
# --- CUSTOM IMPORTS ---
from src.utils.experiments_run_utils import make_experiment_6, split_list_in_chunks
from src.utils.simulations_utils import generate_simulation
from config.config_experiment_6 import (
    EXPERIMENT_RANDOM_STATE, 
    CONFIG_EXPERIMENT,
    N_REALIZATIONS, 
    CHUNK_SIZE
)
from config.config_simulations import SIMULATION_CONFIGS  

###########################################################################################
# --- MAIN EXECUTION ---

def main():

    logging.info(f"▶️ STARTING EXPERIMENT 6: SAMPLING STABILITY")
     
    # 1. Generar la DATA FIJA (Usamos siempre el mismo EXPERIMENT_RANDOM_STATE para los datos)
    logging.info("STEP 1: Generating fixed data (simulation_size_1)...")
    simulation_config = SIMULATION_CONFIGS['simulation_size_1']
    X, y = generate_simulation(**simulation_config, random_state=EXPERIMENT_RANDOM_STATE, return_outlier_idx=False)

    # 2. Cargar historial global si existe
    GLOBAL_MERGED_DATA = {}
    if os.path.exists(results_save_path):
        logging.info("STEP 2: Loading existing merged file...")
        try:
            with open(results_save_path, 'rb') as f:
                GLOBAL_MERGED_DATA = pickle.load(f)
            
            existing_seeds_count = len(GLOBAL_MERGED_DATA.keys())
            logging.info(f" -> Loaded history for {existing_seeds_count} sampling realizations.")
        except Exception as e:
            logging.warning(f" -> Could not read existing merged file ({e}). Starting fresh.")

    # 3. Preparar Semillas de Muestreo (Sampling Seeds) y Chunks
    # Generamos las semillas usando el EXPERIMENT_RANDOM_STATE como base matemática para ser reproducibles
    random.seed(EXPERIMENT_RANDOM_STATE)
    all_possible_sampling_seeds = random.sample(range(100000), N_REALIZATIONS)
    
    # Comprobación rápida por si cambiaste N_REALIZATIONS
    if GLOBAL_MERGED_DATA:
        existing_seeds = list(GLOBAL_MERGED_DATA.keys())
        if N_REALIZATIONS < len(existing_seeds):
            logging.error(f"❌ PELIGRO: Has pedido {N_REALIZATIONS} realizaciones, pero hay {len(existing_seeds)} guardadas. Abortando para no perder datos.")
            sys.exit(1)

    all_chunks = split_list_in_chunks(all_possible_sampling_seeds, chunk_size=CHUNK_SIZE)

    # 4. Bucle de Chunks
    logging.info("STEP 3: Processing Chunks...")
    for chunk_id, chunk_seeds in enumerate(tqdm(all_chunks, desc='Chunks')):
        chunk_filename = f'results_exp_6_chunk_{chunk_id}.pkl'
        chunk_path = os.path.join(results_dir, chunk_filename)
        
        # Diccionario vacío para empezar (estructura [semilla][metrica][modelo])
        chunk_results = {}
        chunk_needs_save = False

        # Intentar cargar chunk previo
        if os.path.exists(chunk_path):
            try:
                with open(chunk_path, 'rb') as f:
                    chunk_results = pickle.load(f)
            except Exception:
                pass
        
        # Averiguar qué semillas de este chunk faltan por calcular
        seeds_to_run = []
        for seed in chunk_seeds:
            # Si no está en el chunk físico, pero sí en el global, la rescatamos
            if seed not in chunk_results and seed in GLOBAL_MERGED_DATA:
                chunk_results[seed] = GLOBAL_MERGED_DATA[seed]
                chunk_needs_save = True
            elif seed not in chunk_results:
                seeds_to_run.append(seed)

        # Ejecutar las que faltan
        if seeds_to_run:
            new_results = make_experiment_6(X=X, y=y, random_states=seeds_to_run, experiment_config=CONFIG_EXPERIMENT)
            
            # Unir los resultados nuevos al dict del chunk a nivel de semilla
            for seed, seed_data in new_results.items():
                chunk_results[seed] = seed_data
            
            chunk_needs_save = True
            
        # Guardar chunk si hubo cambios
        if chunk_needs_save:
            with open(chunk_path, 'wb') as f:
                pickle.dump(chunk_results, f)

    # 5. Consolidación Final
    logging.info("STEP 4: Consolidating final results...")
    final_merged = {}
    
    for chunk_id in range(len(all_chunks)):
        chunk_path = os.path.join(results_dir, f'results_exp_6_chunk_{chunk_id}.pkl')
        if os.path.exists(chunk_path):
            with open(chunk_path, 'rb') as f:
                chunk_data = pickle.load(f)
                # Como las semillas son únicas por chunk, podemos usar update directamente
                final_merged.update(chunk_data)

    # Guardar archivo global
    try:
        with open(results_save_path, 'wb') as f:
            pickle.dump(final_merged, f)
        logging.info(f"   💾 Final file saved: {results_save_path}")
    except Exception as e:
        logging.error(f"   ❌ Failed to save final file: {e}")
        sys.exit(1)

    logging.info("✅ EXPERIMENT PIPELINE FINISHED SUCCESSFULLY")

###########################################################################################
if __name__ == "__main__":
    main()