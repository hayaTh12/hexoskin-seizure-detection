
"""
This script converts raw Hexoskin CSV files into a clean 1 Hz Parquet format
used in the rest of the pipeline.

Inputs (from config.py):
    RAW_DIR             = path to the folder with raw CSV files
    PROCESSED_1HZ_DIR   = path to the folder where processed Parquet will be saved
    PATTERN_RAW_CSV     = glob pattern for input files (default: "*.csv")

What it does:
    - Reads all CSV files matching PATTERN_RAW_CSV inside RAW_DIR (and subfolders)
    - Cleans and standardizes the columns
    - Creates a "time_elapsed" column in seconds (from raw ticks @ 256 Hz)
    - Keeps only heart_rate, breathing_rate, cadence, activity
    - Removes rows where all CORE columns are missing
    - Saves each file as <name>_1Hz.parquet inside PROCESSED_1HZ_DIR
      while preserving the input folder structure

Output:
    One Parquet file per input CSV, stored in PROCESSED_1HZ_DIR,
    mirroring the input directory structure.

Usage:
    python preprocessing.py [--force]
"""

from __future__ import annotations
import csv, re, argparse
from pathlib import Path
import polars as pl

from src.training.config import RAW_DIR, PROCESSED_1HZ_DIR, PATTERN_RAW_CSV

CORE_COLS = ["heart_rate", "breathing_rate", "cadence", "activity"]

# Détection des noms de colonnes
KW_REGEX = {
    "heart_rate":     re.compile(r"\b(heart(_|\s*)rate|hr)\b", re.I),
    "breathing_rate": re.compile(r"\b(breath(ing)?(_|\s*)rate|br)\b", re.I),
    "activity":       re.compile(r"\bactivity\b", re.I),
    "cadence":        re.compile(r"\bcad(ence)?\b", re.I),
    "time_elapsed":   re.compile(r"\b(time(_|\s*)elapsed|time\s*offset|elapsed)\b", re.I),
}

def _canon_map(cols: list[str]) -> dict[str, str]:
    """Mappe colonnes d'origine → noms canoniques."""
    hits = {c: [] for c in KW_REGEX}
    for col in cols:
        low = col.lower()
        for canon, rx in KW_REGEX.items():
            m = rx.search(low)
            if m:
                hits[canon].append((col, len(m.group(0))))
    rename = {}
    for canon, lst in hits.items():
        if not lst:
            continue
        lst.sort(key=lambda x: x[1], reverse=True)
        src, _ = lst[0]
        rename[src] = canon
    return rename

def _to_float(expr: pl.Expr) -> pl.Expr:
    return (
        expr.cast(pl.Utf8, strict=False)
            .str.replace_all(",", ".")
            .str.replace_all(r"[^\d\.\+\-eE]", "")
            .cast(pl.Float64, strict=False)
    )

def preprocess_file(src: Path, dst: Path, *, force: bool=False) -> None:
    if dst.exists() and not force:
        return

    # nom de la 1ère colonne (temps brut en ticks chez Hexoskin)
    # permet de calculer le temps en secondes plus tard 
    with src.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)   
        header = next(reader, None)
        if not header:
            print(f"[WARN] Fichier vide: {src}")
            return
        time_col_raw = header[0]

    # lecture csv
    df = pl.read_csv(
        src,
        ignore_errors=True,
        null_values=["", "NA", "N/A", "--", "NaN", "nan"],
        infer_schema_length=5000,
    )
    if df.is_empty():
        return

    # renommer les colonnes en noms courts
    rename = _canon_map(df.columns)
    df = df.rename({k: v for k, v in rename.items() if k != time_col_raw})

    # convertir toutes les colonnes en float (sauf time brut)
    other_cols = [c for c in df.columns if c != time_col_raw]
    df = df.with_columns(
        [pl.col(time_col_raw).cast(pl.Float64, strict=False)] +
        [_to_float(pl.col(c)).alias(c) for c in other_cols]
    )

    # calculer time_elapsed (temps relatif)
    t0_series = df[time_col_raw].drop_nulls()
    if t0_series.is_empty():
        print(f"[WARN] Colonne temps vide: {src}")
        return
    t0 = float(t0_series.item(0))
    df = df.with_columns(((pl.col(time_col_raw) - pl.lit(t0)) / 256.0).alias("time_elapsed"))

    # garder que les colonnes CORE et time_elapsed
    present_core = [c for c in CORE_COLS if c in df.columns]
    if not present_core:
        print(f"[WARN] Aucune colonne CORE dans {src.name}, fichier ignoré.")
        return
    df = df.select(["time_elapsed"] + present_core)

    # supprimer lignes où toutes les CORE sont nulles
    any_core = pl.any_horizontal([pl.col(c).is_not_null() for c in present_core])
    df = df.filter(any_core)

    # trier par temps (secondes exactes)
    df = df.sort("time_elapsed")

    # écrire parquet (zstd)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dst, compression="zstd")
    print(f"{src.name} → {dst}")

def preprocess_folder(in_dir: Path, out_dir: Path, pattern: str="*.csv", *, force: bool=False) -> None:
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    for src in in_dir.rglob(pattern):
        rel = src.relative_to(in_dir)
        # On conserve le suffixe _1Hz.parquet pour compatibilité data_loader
        dst = out_dir / rel.parent / f"{src.stem}_1Hz.parquet"
        preprocess_file(src, dst, force=force)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    preprocess_folder(RAW_DIR, PROCESSED_1HZ_DIR, pattern=PATTERN_RAW_CSV, force=args.force)
