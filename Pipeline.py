"""

What this script does:
    1. Preprocess raw Hexoskin CSV → 1 Hz Parquet
    2. Label 1 Hz Parquet files with seizure annotations
    3. Extract features from labeled 1 Hz data
    4. Build dataset (X, y, groups) for each window size
    5. Train models on each dataset

Inputs (from config.py):
    RAW_DIR            = folder with raw Hexoskin CSV
    PROCESSED_1HZ_DIR  = output folder for preprocessed 1 Hz files
    LABELED_1HZ_DIR    = output folder for labeled 1 Hz files
    FEATURES_DIR       = output folder for feature files
    WINDOW_SIZES       = list of feature window sizes (seconds)
    INCLUDE_RAW_IN_DL  = whether to include raw context windows in dataset
    DATASET_CACHE_DIR  = folder for cached X/y/groups

Outputs:
    - Preprocessed 1 Hz files
    - Labeled 1 Hz files
    - Feature files (<stem>_features.parquet)
    - Cached datasets (X, y, groups) per window size
    - trained models / reports

Usage:
    python Pipeline.py

"""

# combiner for all the steps in the pipeline
from src.training.Preprocessing import preprocess_folder
from src.training.data_loader import load_all_data
from src.training.extract_features import process_file as extract_features
from src.training.label import main as label_all
from pathlib import Path
from src.training.Multiple_model_training import train_multiple_models
from src.training.config import WINDOW_SIZES

# --- use config paths & patterns instead of hardcoding ---
from src.training.config import (
    RAW_DIR as RAW_DIR_CFG,
    PROCESSED_1HZ_DIR as PROCESSED_DIR_CFG,
    LABELED_1HZ_DIR as LABELED_DIR_CFG,
    FEATURES_DIR as FEATS_DIR_CFG,
    PATTERN_1HZ_PARQUET,
    INCLUDE_RAW_IN_DL,
    DATASET_CACHE_DIR,
    RESULTS_DIR
)

# Keep local aliases if you like the original variable names
RAW_DIR = RAW_DIR_CFG
PROCESSED_DIR = PROCESSED_DIR_CFG
LABELED_DIR = LABELED_DIR_CFG
FEATS_DIR = FEATS_DIR_CFG

def run_pipeline(force: bool = False, force_feat: bool = True):
    # Step 1: Preprocess the raw data (CSV -> 1 Hz canonique)
    # max_ffill_age_s: tolérance de propagation NON-CORE (en s). Laisse défaut (=3.0) si OK pour toi.
    preprocess_folder(RAW_DIR, PROCESSED_DIR, force=force)  # ou max_ffill_age_s=3.0

    # Step 2: Label the data (ajoute 'label' aux 1 Hz)
    label_all(force=force)

    # Step 3: Extract features pour chaque fichier étiqueté
    for file in LABELED_DIR.rglob(PATTERN_1HZ_PARQUET):
        extract_features(file, force=force_feat)

    # Step 4: Charger dataset et entraîner modèle pour chaque fenêtre
    for win in WINDOW_SIZES:
        print(f"Processing window size: {win}")
        X, y, groups = load_all_data(
            folder_path=FEATS_DIR,
            window_size=win,
            include_raw=INCLUDE_RAW_IN_DL,          # suit la config
            raw_folder=LABELED_DIR,
            cache_dir=Path(DATASET_CACHE_DIR) / f"{win}s"  # un sous-dossier par fenêtre
        )

        if len(X) == 0:
            print(f"Aucune donnée pour fenêtre {win}s - ignoré.")
            continue

        # Step 5: Train models
        out_dir = Path(RESULTS_DIR) / f"{win}s"
        train_multiple_models(X, y, groups, export_dir=out_dir)

if __name__ == "__main__":
    run_pipeline()
    print("Pipeline completed successfully.")
