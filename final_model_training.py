"""
Train on the full dataset (no outer-CV) and save the final model.

Inputs (from config.py)
- DATASET_CACHE_DIR/<WIN>s/{X,y,groups}.parquet
- STEP, WINDOW_SIZES (to pick default WIN)
- REPORTS_DIR (optional)
- MODELS_DIR (optional)

Outputs
- models/<WIN>s/model.joblib
- models/<WIN>s/feature_names.json
- models/<WIN>s/metadata.json

Usage
    python final_model_training.py --win 15
"""
from pathlib import Path
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.training.Multiple_model_training import FiringWrapper, ForwardFillImputer, compute_scale_pos_weight
from src.training import config as CFG

STEP              = CFG.STEP
DATASET_CACHE_DIR = CFG.DATASET_CACHE_DIR
MODELS_DIR        = getattr(CFG, "MODELS_DIR", Path("models"))
REPORTS_DIR       = getattr(CFG, "REPORTS_DIR", Path("reports"))
DEFAULT_WIN       = (CFG.WINDOW_SIZES[0] if getattr(CFG, "WINDOW_SIZES", None) else 15)

def load_dataset(win: int):
    ds_dir = Path(DATASET_CACHE_DIR) / f"{win}s"
    X = pd.read_parquet(ds_dir / "X.parquet")
    y = pd.read_parquet(ds_dir / "y.parquet").squeeze("columns")
    groups = pd.read_parquet(ds_dir / "groups.parquet").squeeze("columns")
    return X, y, groups

def build_final_estimator(scale_pos_weight: float):
    # même base que dans Multiple_model_training, mais sans GridSearchCV
    pipe = Pipeline([
        ("ffill",  ForwardFillImputer()),
        ("impute", SimpleImputer(strategy="mean")),
        ("scale",  StandardScaler()),
        ("model",  XGBClassifier(
            tree_method="hist",
            objective="binary:logistic",
            eval_metric=["aucpr"],
            missing=np.nan,
            random_state=42,
            # des valeurs stables issues de tes meilleurs essais
            learning_rate=0.1,
            n_estimators=2200,
            grow_policy="lossguide",
            max_leaves=1024,
            max_bin=512,
            min_child_weight=15,
            max_delta_step=1,
            subsample=0.7,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=30,
            scale_pos_weight=scale_pos_weight,
        )),
    ])
    # lissage + seuil final (choisis une config robuste)
    clf = FiringWrapper(base_estimator=pipe, width=6, threshold=0.40)
    return clf

def main():
    parser = argparse.ArgumentParser(description="Train on full dataset and export model.")
    parser.add_argument("--win", type=int, default=DEFAULT_WIN, help="Window size (seconds).")
    args = parser.parse_args()

    win = int(args.win)
    X, y, groups = load_dataset(win)

    spw = compute_scale_pos_weight(y.values) * 10  # même logique que pendant la CV
    est = build_final_estimator(spw)
    est.fit(X, y)

    out_dir = Path(MODELS_DIR) / f"{win}s"
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(est, out_dir / "model.joblib")

    feature_names = list(X.columns)
    (out_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2))

    meta = {
        "window_seconds": win,
        "step_seconds": STEP,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "model_type": "XGBClassifier + FiringWrapper",
        "data_cache": str(Path(DATASET_CACHE_DIR) / f"{win}s"),
        "reports_dir": str(REPORTS_DIR),
        "versions": {
            "xgboost": XGBClassifier.__module__.split(".")[0],
            "sklearn": "scikit-learn",
        }
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved model to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
