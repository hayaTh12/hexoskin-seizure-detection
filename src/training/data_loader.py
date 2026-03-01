"""
What this module does
    - Reads per-record feature files ("*_features.parquet")
    - Optionally augments X with raw 1 Hz context windows (ex: 15 s history)
    - Aligns everything on a consistent MultiIndex: (record, time)
    - Returns:
        X: pd.DataFrame  (features, and optional raw windows like HR_t-14..HR_t-0)
        y: pd.Series     (binary labels)
        groups: pd.Series (group IDs for Group-aware CV)

Inputs
    - folder_path: path to features folder (recursively scanned for *_features.parquet)
    - window_size: required when include_raw=True (size of raw context window in seconds)
    - include_raw: if True, adds raw windows built from *_1Hz.parquet
    - raw_folder: folder to search raw files (default : labeled_1hz)
    - cache_dir: if set, caches X, y, groups to parquet 

Defaults (from config.py)
    - folder_path = FEATURES_DIR
    - pattern     = PATTERN_FEATURES_PARQUET
    - target_col  = TARGET_COL
    - window_size = RAW_WINDOW_SIZE_FOR_DL
    - include_raw = INCLUDE_RAW_IN_DL
    - raw_folder  = LABELED_1HZ_DIR
    - raw_pattern = PATTERN_1HZ_PARQUET
    - cache_dir   = DATASET_CACHE_DIR

Outputs
    - X, y, groups aligned on the same MultiIndex (record, time)

Usage
    # uses config defaults automatically
    X, y, groups = load_all_data()
"""

from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict
import numpy as np
import pandas as pd
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

# --- imports de la configuration (valeurs par défaut) ---
from src.training.config import (
    FEATURES_DIR,
    LABELED_1HZ_DIR,
    DATASET_CACHE_DIR,
    RAW_WINDOW_SIZE_FOR_DL,
    INCLUDE_RAW_IN_DL,
    TARGET_COL,
    PATTERN_FEATURES_PARQUET,
    PATTERN_1HZ_PARQUET,
)

# Colonnes brutes attendues (fenêtrées en *_t-k)
RawCols = ["activity","breathing_rate","cadence","heart_rate"]


# Utils 
def _normalize_record_stem(stem: str) -> str:
    """Uniformise les noms de record entre RAW et FEATURES."""
    return (stem
            .replace("_1Hz", "")
            .replace("_raw", "")
            .replace("_features", ""))

def _ensure_index_names(obj, names=("record","time")):
    """Impose les noms de niveaux d'index pour compatibilité training."""
    if hasattr(obj, "index") and isinstance(obj.index, pd.MultiIndex):
        if list(obj.index.names) != list(names):
            obj.index = obj.index.set_names(list(names))

def _build_record_anchors(records: List[str]) -> Dict[str, pd.Timestamp]:
    """
    Ancrage tz-naive unique par record : base + i jours.
    IMPORTANT: tz-naive (pas de 'Z'), pour éviter les conflits tz-aware/naive.
    """
    base = pd.Timestamp("2000-01-01 00:00:00")  # tz-naive 
    return {rec: (base + pd.Timedelta(days=i)) for i, rec in enumerate(records)}


# FEATURES: time (seconds)
def _scan_features(path: Path, window_size: Optional[int]) -> pl.LazyFrame:
    lf = pl.scan_parquet(path)
    if (window_size is not None) and ("window" in lf.columns):
        lf = lf.filter(pl.col("window") == window_size)
    if "window" in lf.columns:
        lf = lf.drop("window")
    lf = lf.with_columns([
        pl.col("time").cast(pl.Float64, strict=False).alias("time"),  # seconds
        pl.col("label").cast(pl.Int8),
    ])
    group_id  = path.parent.name
    record_id = _normalize_record_stem(path.stem.replace("_features",""))
    lf = lf.with_columns([
        pl.lit(group_id).alias("group"),
        pl.lit(record_id).alias("record"),
    ])
    return lf.select(sorted(lf.columns))


# -RAW: time (seconds RELATIVES par record) 
def _scan_raw_1hz(path: Path) -> pl.LazyFrame:
    lf = pl.scan_parquet(path)

    # Attache 'record' tôt pour .over("record")
    rec_id = _normalize_record_stem(path.stem)
    lf = lf.with_columns(pl.lit(rec_id).alias("record"))

    # Normalisation: toujours sortir 'time' en secondes RELATIVES par record
    if "sec" in lf.columns:
        lf = lf.with_columns(pl.col("sec").cast(pl.Float64, strict=False).alias("time"))
    elif "time_elapsed" in lf.columns:
        lf = lf.with_columns(pl.col("time_elapsed").cast(pl.Float64, strict=False).alias("time"))
    elif "time" in lf.columns:
        # Fallback: peut être epoch/ms/ns → convertir & rebaser
        lf = lf.with_columns(pl.col("time").cast(pl.Float64, strict=False).alias("time"))
        # ns → s, ms → s
        lf = lf.with_columns(
            pl.when(pl.col("time") > 1e14).then(pl.col("time") / 1e9)
             .when(pl.col("time") > 1e11).then(pl.col("time") / 1e3)
             .otherwise(pl.col("time")).alias("time")
        )
        # Si ça ressemble encore à de l'epoch (secondes très grandes), rebase par record
        lf = lf.with_columns(
            pl.when(pl.col("time") > 1e7)
              .then(pl.col("time") - pl.col("time").min().over("record"))
              .otherwise(pl.col("time"))
              .alias("time")
        )
    else:
        raise ValueError(f"{path.name}: aucune colonne 'sec', 'time_elapsed' ni 'time'")

    # Colonnes brutes en float64 ; manquantes → NaN
    exist = [c for c in RawCols if c in lf.columns]
    miss  = [c for c in RawCols if c not in lf.columns]
    if exist:
        lf = lf.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in exist])
    if miss:
        lf = lf.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in miss])

    return lf.select(sorted(RawCols + ["record","time"]))


#  Grille 1 Hz & fenêtrage (secondes)
MAX_SPAN = 5_000_000  # ~58 jours, garde-fou contre spans absurdes

def _build_1hz_grid_seconds(sub_raw: pd.DataFrame) -> pd.DataFrame:
    """
    sub_raw: DataFrame indexé par 'time' (float secondes RELATIVES), colonnes RawCols
    → retourne un DF index=int secondes CONTIGÜES (RangeIndex), 1 Hz.
    """
    if sub_raw.empty:
        return pd.DataFrame(index=pd.RangeIndex(0), columns=RawCols, dtype=np.float64)

    # Nettoyage index -> secondes entières
    t = pd.to_numeric(pd.Index(sub_raw.index), errors="coerce").to_numpy(dtype="float64")
    t[~np.isfinite(t)] = np.nan
    sec_f = np.rint(t)
    ok = ~np.isnan(sec_f)
    if not ok.any():
        return pd.DataFrame(index=pd.RangeIndex(0), columns=RawCols, dtype=np.float64)

    sec = sec_f[ok].astype("int64")
    lo, hi = int(sec.min()), int(sec.max())
    span = hi - lo

    # Garde-fou : refuse spans déraisonnables / epoch-like
    if span < 0 or span > MAX_SPAN or hi > 10_000_000:
        raise ValueError(
            f"SUSPICIOUS time/sec span={span} lo={lo} hi={hi}. "
            "Attendu: secondes RELATIVES par record (pas epoch/ms/ns)."
        )

    # Agrégation par seconde (dernière ligne), puis reindex contigu 1 Hz
    vals = sub_raw.iloc[ok.nonzero()[0]].copy()
    vals["sec"] = sec
    agg = vals.groupby("sec", as_index=True).last()

    full_index = pd.RangeIndex(lo, hi + 1, 1)
    grid = agg.reindex(full_index)  # gaps → NaN
    grid.index.name = "sec"

    for c in RawCols:
        if c not in grid.columns:
            grid[c] = np.nan
    return grid[RawCols]

def _windowize_record_seconds(grid: pd.DataFrame,
                              starts_sec: np.ndarray,
                              window: int) -> pd.DataFrame:
    """
    grid : DF index=int seconds contigu, colonnes RawCols
    starts_sec : np.ndarray (float/int secondes)
    window : taille fenêtre (s)
    → DF len(starts) x (len(RawCols)*window) avec '<col>_t-k'
    """
    cols = sum(([f"{c}_t-{i}" for i in range(window-1, -1, -1)] for c in RawCols), [])
    if grid.empty or len(starts_sec) == 0:
        return pd.DataFrame(np.full((len(starts_sec), len(cols)), np.nan, dtype=np.float64),
                            index=np.arange(len(starts_sec)), columns=cols)
    T = len(grid)
    start0 = int(grid.index[0])
    offs = (np.rint(starts_sec).astype(np.int64) - start0)
    valid = (offs >= 0) & (offs + window <= T)

    out_blocks: List[pd.DataFrame] = []
    for col in RawCols:
        vec = grid[col].to_numpy(dtype=np.float64)
        if len(vec) < window:
            W = np.full((len(starts_sec), window), np.nan, dtype=np.float64)
        else:
            SW = sliding_window_view(vec, window_shape=window)  # (T-window+1, window)
            W = np.full((len(starts_sec), window), np.nan, dtype=np.float64)
            W[valid] = SW[offs[valid]]
        names = [f"{col}_t-{i}" for i in range(window-1, -1, -1)]
        out_blocks.append(pd.DataFrame(W, index=np.arange(len(starts_sec)), columns=names))
    return pd.concat(out_blocks, axis=1)


def load_all_data(
    folder_path: Union[str, Path] = FEATURES_DIR,
    pattern: str = PATTERN_FEATURES_PARQUET,
    target_col: str = TARGET_COL,
    window_size: Optional[int] = RAW_WINDOW_SIZE_FOR_DL,   # requis si include_raw=True
    include_raw: bool = INCLUDE_RAW_IN_DL,
    raw_folder: Union[str, Path, None] = LABELED_1HZ_DIR,
    raw_pattern: str = PATTERN_1HZ_PARQUET,
    cache_dir: Optional[Union[str, Path]] = DATASET_CACHE_DIR,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:

    folder = Path(folder_path)
    cache_path = Path(cache_dir) if cache_dir else None

    # Cache prêt 
    if cache_path and (cache_path / "X.parquet").exists():
        print("→ Chargement du dataset mis en cache depuis", cache_path)
        X      = pd.read_parquet(cache_path / "X.parquet")
        y      = pd.read_parquet(cache_path / "y.parquet")["label"].astype("int8")
        groups = pd.read_parquet(cache_path / "groups.parquet")["group"]
        _ensure_index_names(X)
        _ensure_index_names(y.to_frame())
        _ensure_index_names(groups.to_frame())
        return X, y, groups

    # FEATURES 
    feat_files = list(folder.rglob(pattern))
    if not feat_files:
        raise FileNotFoundError(f"Aucun fichier {pattern} dans {folder}")

    feat_lf = pl.concat([_scan_features(f, window_size) for f in feat_files], how="diagonal")
    feat_pl = (feat_lf.with_row_index("_row")
                      .sort(by=["record","time","_row"])
                      .drop("_row")
                      .collect())

    # pandas: on garde seconds en colonne puis on fabrique time_dt (datetime) tz-naive
    feat_pd = feat_pl.to_pandas().sort_values(["record","time"])
    record_order = feat_pd["record"].drop_duplicates().tolist()
    anchors = _build_record_anchors(record_order)  # tz-naive

    feat_pd["time_dt"] = feat_pd.apply(
        lambda r: anchors[r["record"]] + pd.to_timedelta(float(r["time"]), unit="s"),
        axis=1
    )

    feat_df = (feat_pd
               .set_index(["record","time_dt"])
               .sort_index())
    feat_df.index = feat_df.index.set_names(["record","time"])

    # dédoublonnage éventuel
    if feat_df.index.duplicated(keep="last").any():
        feat_df = feat_df[~feat_df.index.duplicated(keep="last")]

    y      = feat_df[target_col].astype("int8")
    groups = feat_df["group"]
    # on enlève la colonne 'time' (seconds) restée dans feat_pd lors du set_index
    X_feat = feat_df.drop(columns=[target_col, "group", "time"])

    #  RAW 
    if not include_raw:
        X = X_feat
    else:
        if window_size is None:
            raise ValueError("window_size doit être spécifié quand include_raw=True")

        search_dir = Path(raw_folder) if raw_folder else folder
        raw_files  = list(search_dir.rglob(raw_pattern))
        if not raw_files:
            raise FileNotFoundError(f"Aucun fichier {raw_pattern} dans {search_dir}")

        feat_records = set(X_feat.index.get_level_values(0).unique())
        raw_files = [f for f in raw_files if _normalize_record_stem(f.stem) in feat_records]
        if not raw_files:
            raise RuntimeError("Aucun record commun entre features et raw.")

        raw_lf = pl.concat([_scan_raw_1hz(f) for f in raw_files], how="diagonal")
        raw_pl = raw_lf.collect()
        raw_df = (raw_pl.to_pandas()
                          .set_index(["record","time"])   # time seconds (relatives)
                          .sort_index())
        raw_df = raw_df[~raw_df.index.duplicated(keep="last")]

        X_raw_parts: List[pd.DataFrame] = []
        for rec, sub_feat in X_feat.groupby(level=0, sort=False):
            # starts_dt : niveau time (datetime tz-naive) → seconds relatifs via ancre
            starts_dt  = pd.DatetimeIndex(sub_feat.index.get_level_values(1).values)
            if getattr(starts_dt, "tz", None) is not None:
                starts_dt = starts_dt.tz_localize(None)  # force tz-naive par sécurité

            anchor_rec = anchors[rec]
            if getattr(anchor_rec, "tz", None) is not None:
                anchor_rec = anchor_rec.tz_localize(None)  # sécurité

            starts_sec = (starts_dt - anchor_rec).total_seconds().astype(float)

            try:
                sub_raw = raw_df.xs(rec)  # index = time (seconds), colonnes RawCols
            except KeyError:
                # record brut manquant → tout NaN pour RAW
                empty_cols = sum(([f"{c}_t-{i}" for i in range(window_size-1, -1, -1)] for c in RawCols), [])
                empty = pd.DataFrame(np.full((len(starts_sec), len(empty_cols)), np.nan, dtype=np.float64),
                                     index=sub_feat.index, columns=empty_cols)
                X_raw_parts.append(pd.concat([sub_feat, empty], axis=1))
                continue

            grid = _build_1hz_grid_seconds(sub_raw)
            raw_win = _windowize_record_seconds(grid, starts_sec, window_size)
            raw_win.index = sub_feat.index  # MultiIndex ('record','time' datetime tz-naive)
            X_raw_parts.append(pd.concat([sub_feat, raw_win], axis=1))

        X = pd.concat(X_raw_parts, axis=0).sort_index()

    # DIAGNOSTIC NaN (AVANT filtrage) 
    raw_prefixes = [f"{c}_t-" for c in RawCols]
    raw_cols  = [c for c in X.columns if any(c.startswith(p) for p in raw_prefixes)]
    feat_cols = [c for c in X.columns if c not in raw_cols]

    X_feat_only = X[feat_cols] if feat_cols else pd.DataFrame(index=X.index)
    X_raw_only  = X[raw_cols]  if raw_cols  else pd.DataFrame(index=X.index)

    total_nan_feat = int(X_feat_only.isna().sum().sum()) if not X_feat_only.empty else 0
    cells_feat     = int(X_feat_only.size)               if not X_feat_only.empty else 0
    pct_feat       = (total_nan_feat / cells_feat) if cells_feat else 0.0

    total_nan_raw = int(X_raw_only.isna().sum().sum()) if not X_raw_only.empty else 0
    cells_raw     = int(X_raw_only.size)               if not X_raw_only.empty else 0
    pct_raw       = (total_nan_raw / cells_raw) if cells_raw else 0.0

    n_samples, n_cols = X.shape
    total_nan_all = int(X.isna().sum().sum()); cells_all = int(X.size)
    pct_all       = total_nan_all / cells_all

    print(
        "[AVANT] RAW: "
        f"{cells_raw} cells, {total_nan_raw} NaN ({pct_raw:.2%}) | "
        "FEATURES: "
        f"{cells_feat} cells, {total_nan_feat} NaN ({pct_feat:.2%}) | "
        "ALL: "
        f"{cells_all} cells, {total_nan_all} NaN ({pct_all:.2%}) | "
        f"samples={n_samples}, cols={n_cols}"
    )

    # Filtrage colonnes trop vides + drop forcé 
    nan_ratio  = X.isna().mean()
    too_empty  = nan_ratio[nan_ratio > 0.60].index
    drop_vars  = ["expiration", "inspiration"]  # <-- ne plus drop RR_interval
    forced_drop = [c for c in X.columns if any(c.startswith(v + "_t-") for v in drop_vars)]
    cols_to_drop = set(too_empty).union(forced_drop)
    if cols_to_drop:
        print("Colonnes supprimées :", sorted(cols_to_drop))
        X = X.drop(columns=sorted(cols_to_drop))

    # Alignement & noms d'index
    idx_common = X.index.intersection(y.index).intersection(groups.index)
    X, y, groups = X.loc[idx_common], y.loc[idx_common], groups.loc[idx_common]
    assert len(X) == len(y) == len(groups)

    _ensure_index_names(X)
    _ensure_index_names(y.to_frame())
    _ensure_index_names(groups.to_frame())

    # DIAGNOSTIC NaN (APRES)
    raw_cols  = [c for c in X.columns if any(c.startswith(p) for p in raw_prefixes)]
    feat_cols = [c for c in X.columns if c not in raw_cols]
    X_feat_only = X[feat_cols] if feat_cols else pd.DataFrame(index=X.index)
    X_raw_only  = X[raw_cols]  if raw_cols  else pd.DataFrame(index=X.index)

    total_nan_feat = int(X_feat_only.isna().sum().sum()) if not X_feat_only.empty else 0
    cells_feat     = int(X_feat_only.size)               if not X_feat_only.empty else 0
    pct_feat       = (total_nan_feat / cells_feat) if cells_feat else 0.0

    total_nan_raw = int(X_raw_only.isna().sum().sum()) if not X_raw_only.empty else 0
    cells_raw     = int(X_raw_only.size)               if not X_raw_only.empty else 0
    pct_raw       = (total_nan_raw / cells_raw) if cells_raw else 0.0

    n_samples, n_cols = X.shape
    total_nan_all = int(X.isna().sum().sum()); cells_all = int(X.size)
    pct_all       = total_nan_all / cells_all

    print(
        "[APRES] RAW: "
        f"{cells_raw} cells, {total_nan_raw} NaN ({pct_raw:.2%}) | "
        "FEATURES: "
        f"{cells_feat} cells, {total_nan_feat} NaN ({pct_feat:.2%}) | "
        "ALL: "
        f"{cells_all} cells, {total_nan_all} NaN ({pct_all:.2%}) | "
        f"samples={n_samples}, cols={n_cols}"
    )

    # Cache 
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)
        print("→ Sauvegarde du dataset dans", cache_path)
        X.to_parquet(cache_path / "X.parquet", compression="zstd")
        y.to_frame("label").to_parquet(cache_path / "y.parquet", compression="zstd")
        groups.to_frame("group").to_parquet(cache_path / "groups.parquet", compression="zstd")

    return X, y, groups
