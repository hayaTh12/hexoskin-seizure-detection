# config.py
from pathlib import Path

# Paths
RAW_DIR            = Path("data/raw_hexoskin")    # CSV bruts
PROCESSED_1HZ_DIR  = Path("processed_1hz")        # sorties preprocessing (1Hz parquet)
ANNOTATIONS_DIR    = Path("/data/annotations")          # .txt d'événements
LABELED_1HZ_DIR    = Path("labeled_1hz")          # 1Hz + label
FEATURES_DIR       = Path("features_out")         # fichiers de features
DATASET_CACHE_DIR  = Path("dataset_cache_15s")    # cache X/y/groups
REPORTS_DIR        = Path("reports")              # rapports d'entraînement
RESULTS_DIR = REPORTS_DIR / "final_results"  # résultats finaux
MODELS_DIR  = Path("models")                # modèles entraînés
# Assure que les dossiers existent 
for _d in [PROCESSED_1HZ_DIR, LABELED_1HZ_DIR, FEATURES_DIR, DATASET_CACHE_DIR, REPORTS_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

#File patterns 
PATTERN_RAW_CSV        = "*.csv"
PATTERN_1HZ_PARQUET    = "*_1Hz.parquet"
PATTERN_FEATURES_PARQUET = "*_features.parquet"
ANNOTATIONS_SUFFIX     = ".txt"

# Feature extraction
WINDOW_SIZES = [15]   
STEP         = 5          

# Data loader (construction X/y/groups)
# Fenêtre brute utilisée si include_raw=True 
RAW_WINDOW_SIZE_FOR_DL = 15
INCLUDE_RAW_IN_DL      = True

# Training 
RANDOM_SEED      = 42
N_SPLITS_OUTER   = 5
N_SPLITS_INNER   = 3
TARGET_COL       = "label"
