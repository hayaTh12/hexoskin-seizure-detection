"""
This script takes the preprocessed 1Hz Parquet files and adds a
binary "label" column (0 = no seizure, 1 = seizure) based on
annotations of seizure events.

Input:
    - INPUT_ROOT  = folder with processed Parquet files
                    (ex: "processed_1hz")
    - EVENTS_ROOT = folder with annotations (.txt files)
                    (ex: "annotations")
                    Each annotation file must contain columns:
                    onset (s), duration (s), description
                    Only rows with "FBTCS" in description are kept.

What it does:
    - Reads all *_1Hz.parquet files in INPUT_ROOT
    - Matches them with annotation files from EVENTS_ROOT
      (based on file basename before "_")
    - For each event, labels every second between
      [onset, onset+duration) as 1
    - Adds this "label" column to the Parquet file

Output:
    - OUTPUT_ROOT = folder with labeled 1Hz Parquet files
                    (e.g. "labeled_1hz")
    - Each file keeps the same structure as input
      but with an extra "label" column

Usage without the pipeline:
    python label.py

Options:
    force = if True, overwrite existing labeled files
            if False, skip files that are already processed
    See the main() function for force parameter (last line).

Example:
    python label.py
"""

from pathlib import Path
import pandas as pd
import polars as pl
import numpy as np

# --- chemins & suffix importés depuis config.py ---
from src.training.config import (
    PROCESSED_1HZ_DIR as INPUT_ROOT,
    ANNOTATIONS_DIR   as EVENTS_ROOT,
    LABELED_1HZ_DIR   as OUTPUT_ROOT,
    ANNOTATIONS_SUFFIX,
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def build_events_index(root: Path, suffix: str = ANNOTATIONS_SUFFIX) -> dict[str, list[Path]]:
    """{ base_name -> [paths...] } ; on garde potentiellement plusieurs fichiers events par base."""
    d: dict[str, list[Path]] = {}
    for p in root.rglob(f"*{suffix}"):
        base = p.stem.split("_")[0]
        d.setdefault(base, []).append(p)
    return d

def read_events_file(path: Path) -> pd.DataFrame:
    """Lit un fichier d’évènements et garde les FBTCS (onset/duration en secondes)."""
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # trouver la ligne d’entête
    start = None
    for i, line in enumerate(lines):
        low = line.strip().lower()
        if ("onset" in low) and ("duration" in low) and ("description" in low):
            start = i
            break
    if start is None:
        raise ValueError(f"Header not found in {path}")

    df = pd.read_csv(path, sep=None, engine="python", skiprows=start)
    # normaliser noms de colonnes
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"# onset", "duration", "description"}
    # certaines exports mettent " onset" avec dièse et espace
    # harmonisation légère :
    rename_map = {}
    for c in df.columns:
        if "onset" in c:      rename_map[c] = "# onset"
        elif "duration" in c: rename_map[c] = "duration"
        elif "description" in c: rename_map[c] = "description"
    df = df.rename(columns=rename_map)
    if not {"# onset","duration","description"}.issubset(df.columns):
        raise ValueError(f"Expected onset/duration/description in {path}")

    df = df[df["description"].str.contains("FBTCS", case=False, na=False)]
    out = df[["# onset", "duration"]].copy()
    out["# onset"] = out["# onset"].astype(int)
    out["duration"] = out["duration"].astype(int)
    return out

def load_all_events(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame(columns=["# onset", "duration"])
    frames = [read_events_file(p) for p in paths]
    if not frames:
        return pd.DataFrame(columns=["# onset", "duration"])
    return pd.concat(frames, ignore_index=True)

def add_labels(par_path: Path, ev_paths: list[Path] | None) -> pl.DataFrame:
    df = pl.read_parquet(par_path)

    # choisir l’index de temps entier : priorité à `sec` si présent, sinon round(time_elapsed)
    if "sec" in df.columns:
        times = df["sec"].cast(pl.Int64).to_numpy()
    else:
        times = df["time_elapsed"].round(0).cast(pl.Int64).to_numpy()

    if not ev_paths:
        return df.with_columns(pl.lit(0, dtype=pl.Int8).alias("label"))

    ev = load_all_events(ev_paths)
    if ev.empty:
        return df.with_columns(pl.lit(0, dtype=pl.Int8).alias("label"))

    lab = np.zeros_like(times, dtype=np.int8)
    # Règle d’inclusion : [onset, onset+duration)
    for onset, duration in ev.itertuples(index=False):
        if duration <= 0:
            continue
        mask = (times >= onset) & (times < onset + duration)
        lab[mask] = 1

    return df.with_columns(pl.Series("label", lab))

def main(force: bool = False):
    idx = build_events_index(EVENTS_ROOT)

    par_files = list(INPUT_ROOT.rglob("*_1Hz.parquet"))
    print(f"{len(par_files)} files 1 Hz found")

    for par_path in par_files:
        base = par_path.stem.replace("_1Hz", "")
        ev_paths = idx.get(base, [])
        out_path = OUTPUT_ROOT / par_path.relative_to(INPUT_ROOT)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not force:
            print(f"{out_path.name} déjà prêt — ignoré")
            continue

        labeled = add_labels(par_path, ev_paths)
        labeled.write_parquet(out_path, compression="zstd")
        print(f"{out_path}  |  events: {len(ev_paths)} file(s)")

    print("All files are labeled.")

if __name__ == "__main__":
    # ajout d'une option --force pour éviter de modifier le code
    import argparse
    p = argparse.ArgumentParser(description="Add labels from annotations to 1Hz Parquet.")
    p.add_argument("--force", action="store_true", help="Overwrite existing labeled files.")
    args = p.parse_args()
    main(force=args.force)
