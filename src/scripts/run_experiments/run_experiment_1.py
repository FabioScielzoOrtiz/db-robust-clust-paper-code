###########################################################################################

# --- IMPORTS ---

import os
import sys
import pickle
import logging

###########################################################################################

# --- LOGGING CONFIGURATION ---

# Configuramos el formato para que se vea igual que tu script de ejemplo
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

###########################################################################################

# --- PATH CONFIGURATION ---

script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(script_path, '..', '..', '..')
results_dir = os.path.join(project_path, 'results', 'experiment_1')
results_save_path = os.path.join(results_dir, 'results_exp_1.pkl')
os.makedirs(results_dir, exist_ok=True)

sys.path.append(project_path)

###########################################################################################

# --- CUSTOM IMPORTS ---

from utils.experiments_run_utils import make_experiment_1

from config.config_experiment_1 import CONFIG_EXPERIMENT
from config.config_simulations import SIMULATION_CONFIGS  

###########################################################################################

# --- MAIN EXECUTION ---

def main():
    
    logging.info(f"▶️ STARTING EXPERIMENT 1")
     
    simulation_config = {k: v for k, v in SIMULATION_CONFIGS['simulation_1'].items() if k != 'n_samples'}

    results = make_experiment_1(
        **simulation_config,
        **CONFIG_EXPERIMENT
    )

    try:
        with open(results_save_path, 'wb') as f:
            pickle.dump(results, f)
        logging.info(f"   💾 Final file saved: {results_save_path}")
    except Exception as e:
        logging.error(f"   ❌ Failed to save final file: {e}")
        sys.exit(1)

    logging.info("✅ EXPERIMENT PIPELINE FINISHED SUCCESSFULLY")

###########################################################################################

if __name__ == "__main__":
    main()