"""
What this script does
    - Reads labeled 1Hz Parquet files (with time + CORE signals + label)
    - Slides fixed-size windows over the signals (sizes from config: WINDOW_SIZES)
    - Step between windows is STEP (from config)
    - Computes simple stats per window (mean, std, range, deltas, ratio ΔHR/ΔBR)
    - Window label = max(label) inside the window
    - Saves one features Parquet per input file as "<stem>_features.parquet"

Inputs
    IN_ROOT  = path to the folder with labeled 1Hz Parquet files
               (from config.py)

Outputs
    OUT_ROOT = path to the folder where feature files are written
               (from config.py)
               Each output mirrors the input structure

Usage
    python extract_features.py
    (edit config WINDOW_SIZES/STEP if needed)

Notes
    - WINDOW_SIZES and STEP are imported from config
"""
# extract_features.py
import numpy as np
import polars as pl
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
from config import WINDOW_SIZES, STEP
# chemins & pattern importés depuis config.py
from src.training.config import LABELED_1HZ_DIR as IN_ROOT, FEATURES_DIR as OUT_ROOT, PATTERN_1HZ_PARQUET

# colonnes attendues
TIME_COL = "time_elapsed"  # float (ou int) croissant
CORE_COLS = ["heart_rate", "breathing_rate", "activity", "cadence"]
LABEL_COL = "label"

FEATURE_NAMES = [
    # HR
    "HR_mean", "HR_range", "HR_std", "DeltaHR",
    # BR
    "BR_mean", "BR_std", "DeltaBR",
    # Activity
    "Act_mean", "Act_std", "DeltaAct",
    # Cadence
    "Cad_mean", "Cad_std", "DeltaCad",
    # Couplage
    "Ratio_DHR_DBR",
]

OUT_ROOT.mkdir(parents=True, exist_ok=True)



def make_windows(vec: np.ndarray, win: int) -> np.ndarray:
    """
    Retourne des fenêtres glissantes de taille `win`, stride = STEP (en échantillons).
    Renvoie un array (N_win, win). Si pas assez d'échantillons → (0, win).
    """
    if vec.shape[0] < win:
        return np.empty((0, win), dtype=vec.dtype)
    sv = sliding_window_view(vec, win)       # (N-win+1, win)
    return sv[::STEP]                        # stride cohérent entre séries

def delta(arr: np.ndarray) -> np.ndarray:
    """Δ = dernière - première valeur sur chaque fenêtre (axis=1)."""
    return arr[:, -1] - arr[:, 0]

def nanrange(arr: np.ndarray) -> np.ndarray:
    """max - min avec NaN autorisés (par ligne)."""
    return np.nanmax(arr, axis=1) - np.nanmin(arr, axis=1)



def compute_features(df: pl.DataFrame) -> pl.DataFrame | None:
    """
    Calcule les features pour un DataFrame 1 Hz (déjà trié/unique sur TIME_COL).
    Retourne un Polars DataFrame avec colonnes:
        ["time", "window"] + FEATURE_NAMES + ["label"]
    """
    # passage numpy (rapide)
    arr = df.to_numpy()
    arr = arr.astype("float64", copy=False)  # toutes les colonnes en float64
    col_idx = {c: i for i, c in enumerate(df.columns)}

    # vecteurs (longueur N)
    hr   = arr[:, col_idx.get("heart_rate")]
    br   = arr[:, col_idx.get("breathing_rate")]
    act  = arr[:, col_idx.get("activity")]
    cad  = arr[:, col_idx.get("cadence")]
    lbl  = arr[:, col_idx.get(LABEL_COL)]
    tvec = arr[:, col_idx.get(TIME_COL)]

    rows = []
    for window in WINDOW_SIZES:
        # fenêtres alignées pour chaque signal
        w_hr   = make_windows(hr, window)
        if w_hr.size == 0:
            continue
        w_br   = make_windows(br, window)
        w_act  = make_windows(act, window)
        w_cad  = make_windows(cad, window)
        w_lbl  = make_windows(lbl, window).max(axis=1).astype(np.int8)  # label = max sur la fenêtre
        w_time = make_windows(tvec, window)[:, -1]                      # timestamp = fin fenêtre

        # HR
        hr_mean  = np.nanmean(w_hr, axis=1)
        hr_std   = np.nanstd(w_hr, axis=1)
        hr_range = nanrange(w_hr)
        d_hr     = delta(w_hr)

        # BR
        br_mean  = np.nanmean(w_br, axis=1)
        br_std   = np.nanstd(w_br, axis=1)
        d_br     = delta(w_br)

        # Activity
        act_mean = np.nanmean(w_act, axis=1)
        act_std  = np.nanstd(w_act, axis=1)
        d_act    = delta(w_act)

        # Cadence
        cad_mean = np.nanmean(w_cad, axis=1)
        cad_std  = np.nanstd(w_cad, axis=1)
        d_cad    = delta(w_cad)

        # Couplage ΔHR / ΔBR
        ratio_dhr_dbr = np.divide(
            d_hr, d_br,
            out=np.full_like(d_hr, np.nan),
            where=(d_br != 0) & ~np.isnan(d_br)
        )

        feats = np.column_stack([
            hr_mean, hr_range, hr_std, d_hr,
            br_mean, br_std, d_br,
            act_mean, act_std, d_act,
            cad_mean, cad_std, d_cad,
            ratio_dhr_dbr,
        ])

        win_col = np.full((feats.shape[0], 1), window, dtype=np.int16)
        rows.append(np.column_stack([w_time, win_col, feats, w_lbl]))

    if not rows:
        return None

    out = np.vstack(rows)
    col_names = ["time", "window"] + FEATURE_NAMES + ["label"]
    feats_df = pl.DataFrame(out, schema=col_names).with_columns(
        pl.col("time").cast(pl.Float64),
        pl.col("window").cast(pl.Int16),
        pl.col("label").cast(pl.Int8),
    )
    return feats_df




def process_file(path_par: Path, *, force: bool = True) -> pl.DataFrame | None:
    """
    Charge un fichier 1 Hz + label (CORE uniquement) et écrit <stem>_features.parquet.
    Retourne le DataFrame des features ou None si pas assez de données.
    """
    out_path = OUT_ROOT / path_par.relative_to(IN_ROOT)
    out_path = out_path.parent / (out_path.stem + "_features.parquet")

    if out_path.exists() and not force and out_path.stat().st_mtime > path_par.stat().st_mtime:
        print(f"✓ {out_path.name} déjà prêt — ignoré")
        return None

    df = pl.read_parquet(path_par)

    # Sécurité: tri + unicité temporelle (garde la dernière si doublon)
    if TIME_COL not in df.columns:
        raise ValueError(f"{path_par.name}: colonne {TIME_COL!r} manquante")
    df = df.sort(TIME_COL).unique(subset=[TIME_COL], keep="last")

    # Colonnes attendues (CORE + label + time)
    expected_cols = set(CORE_COLS + [LABEL_COL, TIME_COL])
    present  = set(df.columns)
    missing  = expected_cols - present
    extra    = present - expected_cols
    print(f"{path_par.name} → manquantes: {sorted(missing)} | inattendues: {sorted(extra)}")

    for col in missing:
        if col == LABEL_COL:
            df = df.with_columns(pl.lit(0, dtype=pl.Int8).alias(LABEL_COL))
        elif col == TIME_COL:
            raise ValueError(f"{path_par.name}: {TIME_COL} absent")
        else:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    feats = compute_features(df.select(list(expected_cols)))  # ordre stable
    if feats is None or feats.height == 0:
        print(f"{path_par.name}: pas assez d’échantillons pour les fenêtres → skip")
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.write_parquet(out_path, compression="zstd")
    print(f"Features saved to {out_path}")
    return feats


if __name__ == "__main__":
    files = list(IN_ROOT.rglob(PATTERN_1HZ_PARQUET))
    print(f"{len(files)} files to process")
    for f in files:
        process_file(f, force=True)
    print("Feature extraction completed.")
