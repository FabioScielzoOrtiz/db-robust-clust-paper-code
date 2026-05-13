import os, sys
import pandas as pd
import logging

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("generate_simulations.log", mode="w")
    ]
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
script_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(script_path, '..', '..', '..'))
output_dir = os.path.join(project_path, 'data', 'skm_budiaji', 'simulations_data')
os.makedirs(output_dir, exist_ok=True)
sys.path.append(project_path)

log.info(f"Project path : {project_path}")
log.info(f"Output dir   : {output_dir}")

import random
from src.utils.simulations_utils import generate_simulation
from config.config_simulations import SIMULATION_CONFIGS
from config.config_experiment_4 import (
    CONFIG_EXPERIMENT,
    EXPERIMENT_RANDOM_STATE,
    N_REALIZATIONS
)

# ── Config ─────────────────────────────────────────────────────────────────────
N_REALIZATIONS = 30
DATA_IDS = [
    'simulation_base', 
    'simulation_outliers_2',
    'simulation_outliers_2a',
    'simulation_outliers_2b',
    'simulation_outliers_6',
    'simulation_sphericity_3',
    'simulation_sphericity_outliers_1',
    'simulation_sphericity_outliers_2',
    "simulation_imbalance_1",
    "simulation_imbalance_2",
    "simulation_imbalance_outliers_1",
    "simulation_sphericity_imbalance_1",
    "simulation_separation_2",
    "simulation_separation_outliers_1",
    "simulation_separation_sphericity_1",
    "simulation_size_outliers_1"
]

log.info(f"N_REALIZATIONS       : {N_REALIZATIONS}")
log.info(f"EXPERIMENT_RANDOM_STATE : {EXPERIMENT_RANDOM_STATE}")
log.info(f"DATA_IDS ({len(DATA_IDS)}): {DATA_IDS}")

random.seed(EXPERIMENT_RANDOM_STATE)
random_state_list = random.sample(range(N_REALIZATIONS * 1000), N_REALIZATIONS)

# ── Main loop ──────────────────────────────────────────────────────────────────
total = len(DATA_IDS) * N_REALIZATIONS
completed = 0

log.info(f"Starting generation — {total} total files pairs to produce")

for data_id in DATA_IDS:
    simulation_config = SIMULATION_CONFIGS[data_id]
    log.info(f"[{data_id}] Starting ({N_REALIZATIONS} realizations)")

    for i, random_state in enumerate(random_state_list):
        log.debug(f"[{data_id}] iter {i:>3} | random_state={random_state}")

        X, y = generate_simulation(
            **simulation_config,
            random_state=random_state,
            return_outlier_idx=False
        )

        y_pd = pd.DataFrame({'y': y})

        X_save_path = os.path.join(output_dir, f'X_{data_id}_iter_{i}.csv')
        y_save_path = os.path.join(output_dir, f'y_{data_id}_iter_{i}.csv')

        X.to_csv(X_save_path, index=False)
        y_pd.to_csv(y_save_path, index=False)

        completed += 1
        log.debug(f"[{data_id}] iter {i:>3} | saved → {X_save_path}")

    log.info(f"[{data_id}] Done — {N_REALIZATIONS} realizations saved  ({completed}/{total} total)")

log.info("All simulations generated successfully.")