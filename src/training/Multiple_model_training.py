"""
What this module does
    - Loads a cached dataset (X, y, groups) for a given window size WIN
      from: DATASET_CACHE_DIR/<WIN>s/{X,y,groups}.parquet
    - Trains one or more models 
    - Exports detailed CSVs (TP/FP/FN/GT) and a JSON summary

Inputs (from config.py)
    - STEP                 : stride between samples (seconds)
    - DATASET_CACHE_DIR    : root folder where datasets are cached
    - ANNOTATIONS_DIR      : folder of .txt annotations (FBTCS)
    - REPORTS_DIR: where to write results

Usage
    # Use default WIN = first value from WINDOW_SIZES in config
    python Multiple_model_training.py

    # Or pick a specific window (seconds)
    python Multiple_model_training.py --win 15

"""

from pathlib import Path
from typing import List, Dict
from collections import Counter, defaultdict
import json, time
from time import perf_counter
import argparse

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# from Earlystopping import EarlyStopping
from timescoring.annotations import Annotation
from timescoring import scoring as ts_scoring


from src.training.config import (
    STEP,
    WINDOW_SIZES,
    DATASET_CACHE_DIR,
    ANNOTATIONS_DIR,
    REPORTS_DIR,
    RESULTS_DIR,   
)

# Ensure output roots exist
Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

DEFAULT_WIN = WINDOW_SIZES[0] if WINDOW_SIZES else 15

USE_SCALE_POS       = True
FP_WIDTHS           = [4, 6, 8]
FP_THRESHOLD        = [0.85, 0.90, 0.95, 1.0]
FP_WIDTHS_XG        = [4, 6]
FP_THRESHOLD_XG     = [0.30]
FP_THRESHOLD_SVM    = [0.55, 0.65, 0.75]
CAL_FRAC            = 0.15
VAL_FRAC            = 0.2
ES_ROUNDS           = 50
INNER_SPLITS        = 6
OUTER_SPLITS        = 7

# Root results directory (under reports/)
RESULT_ROOT = REPORTS_DIR / "final_results"

# Build events index from config annotations dir
EVENTS_ROOT = ANNOTATIONS_DIR
events_index: dict[str, Path] = {
    p.stem.split("_")[0]: p  # « record » = 1er segment avant "_"
    for p in EVENTS_ROOT.rglob("*.txt")
}

def load_dataset(win: int):
    """Loads cached dataset for a given window size from DATASET_CACHE_DIR/<win>s."""
    ds_dir = Path(DATASET_CACHE_DIR) / f"{win}s"
    X = pd.read_parquet(ds_dir / "X.parquet")
    y = pd.read_parquet(ds_dir / "y.parquet").squeeze("columns")  # Series
    groups = pd.read_parquet(ds_dir / "groups.parquet").squeeze("columns")
    return X, y, groups


# GridSearchCV verbeux (annonce le refit)
class VerboseGridSearchCV(GridSearchCV):
    """GridSearchCV qui annonce quand le refit global commence / termine."""
    def _refit(self, X, y, **fit_params):
        print("\n  Inner-CV terminé, début du refit global sur l’ensemble TRAIN…")
        best_est = super()._refit(X, y, **fit_params)
        print(" Refit global terminé\n")
        return best_est


# Lecture flexible des .txt d'annotations (FBTCS)
def load_onset_gen(txt_path: Path):
    """
    Retourne deux arrays (onsets, gen) pour les FBTCS du fichier .txt.
    - Normalise les noms de colonnes (strip/lower).
    - Si 'generalization' absente, on met 0.
    - Ignore les autres événements (garde FBTCS).
    """
    if not txt_path or not txt_path.exists():
        return np.array([]), np.array([])

    header_row = 0
    with txt_path.open() as f:
        for i, line in enumerate(f):
            low = line.lower()
            if "# onset" in low and "duration" in low:
                header_row = i
                break

    df = pd.read_csv(txt_path, sep=None, engine="python", skiprows=header_row)

    df.columns = df.columns.str.strip()
    norm_map = {c.lower(): c for c in df.columns}

    onset_col = norm_map.get("# onset") or norm_map.get("onset")
    descr_col = norm_map.get("description")
    gen_col   = norm_map.get("generalization")  # peut être None

    if onset_col is None:
        return np.array([]), np.array([])

    if descr_col is not None:
        df = df[df[descr_col].astype(str).str.contains("FBTCS", case=False, na=False)]

    if df.empty:
        return np.array([]), np.array([])

    on = pd.to_numeric(df[onset_col], errors="coerce").dropna().astype(int)

    if gen_col is not None:
        gen = pd.to_numeric(df[gen_col], errors="coerce").fillna(0).astype(int)
        gen = gen.loc[on.index]
    else:
        gen = pd.Series(0, index=on.index, dtype=int)

    return on.to_numpy(), gen.to_numpy()


# Utils événements & latences
def firing_power(preds, width: int, threshold: float):
    """Lisse une probabilité avec une fenêtre glissante et applique un seuil."""
    if width <= 1:
        return (preds >= threshold).astype(int)
    kernel = np.ones(width) / width
    smooth = np.convolve(preds, kernel, mode="same")
    return (smooth >= threshold).astype(int)

def _elapsed_seconds_by_record(recs: np.ndarray, step_sec: int) -> np.ndarray:
    """Temps écoulé (s) par record, pas fixe = step_sec."""
    elapsed = np.empty(len(recs), dtype=int)
    counters = {}
    for i, r in enumerate(recs):
        k = counters.get(r, 0)
        elapsed[i] = k * step_sec
        counters[r] = k + 1
    return elapsed

def vec_to_events_seconds(bin_vec: np.ndarray, recs: np.ndarray, step_sec: int):
    """Retourne [(record, start_s, end_s)] pour chaque run de '1' (en secondes)."""
    times_sec = _elapsed_seconds_by_record(recs, step_sec)
    events = []
    in_evt, r0 = False, None
    for i, v in enumerate(np.append(bin_vec, 0)):  # +0 pour flush final
        if v == 1 and not in_evt:
            in_evt, r0 = True, i
        elif v == 0 and in_evt:
            rec = recs[r0]
            events.append((rec, int(times_sec[r0]), int(times_sec[i-1])))
            in_evt = False
    return events

def overlap_seconds(p_start_s: int, p_end_s: int,
                    t_start_s: int, t_end_s: int,
                    tol_start_s: int = 20, tol_end_s: int = 0) -> bool:
    """Chevauchement (avec tolérances) en secondes entières."""
    return (p_end_s   >= (t_start_s - tol_start_s)) and \
           (p_start_s <= (t_end_s   + tol_end_s))

def detection_latencies(ref_bin: np.ndarray, hyp_bin: np.ndarray,
                        tol_start=20, tol_end=0, min_len=2, step=5):
    """Liste des latences (s) sur les événements GT détectés dans la fenêtre tolérée."""
    lat = []
    in_evt, r0, length = False, None, 0
    for i, v in enumerate(list(ref_bin) + [0]):  # flush
        if v == 1:
            if not in_evt:
                in_evt, r0, length = True, i, 1
            else:
                length += 1
            continue

        if in_evt and length >= min_len:
            win_start = max(0, r0 - tol_start // step)
            win_end   = min(len(hyp_bin), i + tol_end // step)
            for j in range(win_start, win_end):
                if hyp_bin[j] == 1:
                    lat.append((j - r0) * step)
                    break
        in_evt = False
    return lat

def _build_ann_cache(events_index: dict[str, Path]):
    """
    {record: (onsets_abs_s, gen_offset_s)} ; convertit 'generalization' absolu en offset si besoin.
    """
    cache = {}
    for rec, p in events_index.items():
        on, gen = load_onset_gen(p)
        on  = np.asarray(on, dtype=int)
        gen = np.asarray(gen, dtype=int) if len(gen) else np.zeros_like(on)
        if on.size == 0:
            cache[rec] = (on, gen)
            continue
        if gen.size and (gen >= on).mean() > 0.8:
            gen = np.clip(gen - on, 0, None)
        cache[rec] = (on, gen)
    return cache

def gen_latencies_aligned(y_true: np.ndarray, y_pred: np.ndarray,
                          recs: np.ndarray, step_sec: int,
                          events_index: dict[str, Path],
                          tol_start_s: int, tol_end_s: int,
                          onset_match_tol_s: int = 30) -> list[int]:
    """Latence après généralisation alignée sur le même hit que la détection."""
    times_sec = _elapsed_seconds_by_record(recs, step_sec)
    ann_cache = _build_ann_cache(events_index)

    gen_lat_list = []
    in_evt, r0, length = False, None, 0
    y_true_seq = list(y_true) + [0]
    N = len(y_pred)

    for i, v in enumerate(y_true_seq):
        if v == 1:
            if not in_evt:
                in_evt, r0, length = True, i, 1
            else:
                length += 1
            continue

        if in_evt and length >= 2:
            gt_start = r0
            gt_end   = i - 1
            win_start = max(0, gt_start - (tol_start_s // step_sec))
            win_end   = min(N, gt_end + 1 + (tol_end_s // step_sec))
            j_hit = None
            for j in range(win_start, win_end):
                if y_pred[j] == 1:
                    j_hit = j
                    break

            if j_hit is not None:
                det_lat = max(0, (j_hit - gt_start) * step_sec)
                rec    = recs[gt_start]
                t0_sec = int(times_sec[gt_start])
                on_arr, gen_off_arr = ann_cache.get(rec, (np.array([]), np.array([])))
                if on_arr.size:
                    k = int(np.argmin(np.abs(on_arr - t0_sec)))
                    if abs(int(on_arr[k]) - t0_sec) <= onset_match_tol_s:
                        gen_off = int(gen_off_arr[k]) if gen_off_arr.size else 0
                        gen_lat = max(0, det_lat - gen_off)
                        gen_lat_list.append(int(gen_lat))
        in_evt = False
    return gen_lat_list


# Imputer
class ForwardFillImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if isinstance(X_df.index, pd.MultiIndex) and "record" in X_df.index.names:
            X_filled = X_df.groupby(level=0).ffill().bfill()
        else:
            X_filled = X_df.ffill().bfill()
        return X_filled.values


# Wrapper lissage + seuillage
class FiringWrapper(BaseEstimator, ClassifierMixin):
    """Étape finale : transforme proba→0/1 via firing_power."""
    _estimator_type = "classifier"
    def __init__(self, base_estimator=None, width=4, threshold=0.65):
        self.base_estimator = base_estimator
        self.width = width
        self.threshold = threshold
    def fit(self, X, y, **fit_params):
        self.est_ = clone(self.base_estimator)
        self.est_.fit(X, y, **fit_params)
        self.classes_ = getattr(self.est_, "classes_", np.array([0,1]))
        return self
    def predict_proba(self, X):
        if not hasattr(self, "est_"):
            raise RuntimeError("Estimator not fitted")
        return self.est_.predict_proba(X)
    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return firing_power(p, width=self.width, threshold=self.threshold)


# Divers
def compute_scale_pos_weight(y):
    pos = np.sum(y == 1)
    neg = len(y) - pos
    return neg / pos if pos else 1.0

def _underlying_estimator(model):
    if isinstance(model, Pipeline):
        return _underlying_estimator(model.steps[-1][1])
    if hasattr(model, "est_"):
        return _underlying_estimator(model.est_)
    for attr in ("base_estimator_", "base_estimator"):
        if hasattr(model, attr):
            return _underlying_estimator(getattr(model, attr))
    return model

def _feature_importances(est, feature_names: List[str]) -> Dict[str, float]:
    est = _underlying_estimator(est)
    if hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
    elif est.__class__.__name__.startswith("CatBoost"):
        imp = np.array(est.get_feature_importance(type="FeatureImportance"))
    elif isinstance(est, LogisticRegression):
        imp = np.abs(est.coef_[0])
    else:
        return {}
    imp = imp / imp.sum() if imp.sum() else imp
    return {f: float(val) for f, val in zip(feature_names, imp)}

class GridSearchCVClassifier(GridSearchCV, ClassifierMixin):
    _estimator_type = "classifier"


# Modèles
def build_models():
    """Retourne dict {nom: (estimator, param_grid)} prêt pour GridSearchCV."""
    return {
        "XGBoost": (
            FiringWrapper(
                base_estimator=Pipeline([
                    ("ffill", ForwardFillImputer()),
                    ("impute", SimpleImputer(strategy="mean")),
                    ("scale", StandardScaler()),
                    ("model", XGBClassifier(
                        tree_method="hist",
                        objective="binary:logistic",
                        eval_metric=["aucpr"],
                        missing=np.nan,
                        random_state=42,
                    )),
                ])
            ),
            {
                "base_estimator__model__learning_rate":    [0.1],
                "base_estimator__model__n_estimators":     [2200],
                "base_estimator__model__grow_policy":      ["lossguide"],
                "base_estimator__model__max_leaves":       [1024],
                "base_estimator__model__max_bin":          [512],
                "base_estimator__model__min_child_weight": [15],
                "base_estimator__model__max_delta_step":   [1],
                "base_estimator__model__subsample":        [0.7],
                "base_estimator__model__colsample_bytree": [0.8],
                "base_estimator__model__reg_lambda":       [1],
                "width":     FP_WIDTHS_XG,
                "threshold": [0.40],
                "base_estimator__model__n_jobs": [30],
            },
        ),
    }


# Entraînement + évaluations + exports
def train_multiple_models(X, y, groups, export_dir: str = "results"):
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    nan_total = int(X.isna().sum().sum())
    total_cells = int(X.shape[0] * X.shape[1])
    pct_total = nan_total / total_cells * 100
    print(f"Total NaN values in features: {nan_total:,}/ {total_cells:,} ({pct_total:.2f}%)")

    models = build_models()
    outer_cv = StratifiedGroupKFold(n_splits=OUTER_SPLITS, shuffle=False)

    results, failures = {}, {}
    feat_imps = defaultdict(lambda: Counter())
    inner_summary_all_models = {}

    # cache annotations (pour gen_latency) une seule fois 
    ann_cache = _build_ann_cache(events_index)

    def _n_events(bin_vec: np.ndarray) -> int:
        d = np.diff(np.concatenate(([0], bin_vec, [0])))
        return int((d == 1).sum())

    def _summary(split_idx, tag):
        g_list  = groups.iloc[split_idx].unique().tolist()
        y_split = y.iloc[split_idx].astype(int).values

        n_evt   = _n_events(y_split)
        n_pos   = int(y_split.sum())
        n_total = len(split_idx)

        times = X.index.get_level_values("time")[split_idx]
        duration_sec = (times.max() - times.min()).total_seconds()
        if n_total > 1:
            median_step = times.to_series().diff().median().total_seconds()
            duration_sec += median_step
        duration_min = duration_sec / 60

        print(f"  [{tag}] {len(g_list)} patients :", g_list)
        print(f"        {n_total:,} samples | {n_evt} événements | "
              f"{n_pos} epochs pos ({n_pos/n_total:.4%}) | durée {duration_min:.1f} min")

    for name, (estimator, param_grid) in models.items():
        print(f"\nModèle : {name}")
        inner_summary = defaultdict(list)

        try:
            inner_cv = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=False)
            scoring = {
                "f1": make_scorer(f1_score),
                "recall": make_scorer(recall_score, zero_division=0),
                "precision": make_scorer(precision_score, zero_division=0)
            }

            fold_stats = []

            # Paramètres event-based
            evt_param = ts_scoring.EventScoring.Parameters()
            evt_param.toleranceEnd   = DEFAULT_WIN
            evt_param.toleranceStart = DEFAULT_WIN
            evt_param.minDurationBetweenEvents = 240
            evt_param.minOverlap = 0
            evt_param.maxEventDuration = 1000

            # Collecteurs globaux par modèle
            all_tp, all_fp = [], []
            all_fn = []
            all_gt = []  # une ligne par évènement GT (detected/latences)

            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups), start=1):
                grp_train = groups.iloc[train_idx]
                grp_test  = groups.iloc[test_idx]

                print(f"Fold {fold_idx}/{OUTER_SPLITS}")
                _summary(train_idx, "TRAIN")
                _summary(test_idx, "TEST")

                tic = time.time()
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                grp_train = groups.iloc[train_idx]

                spw = compute_scale_pos_weight(y_train.values) * 10 if USE_SCALE_POS else 1.0
                est_fold = clone(estimator)
                if name == "XGBoost":
                    est_fold.set_params(**{"base_estimator__model__scale_pos_weight": spw})
                elif name == "CatBoost":
                    est_fold.set_params(**{"base_estimator__model__auto_class_weights": "Balanced"})
                else:
                    est_fold.set_params(**{"base_estimator__model__class_weight": "balanced"})

                search = VerboseGridSearchCV(
                    estimator   = est_fold,
                    param_grid  = param_grid,
                    cv          = inner_cv,
                    scoring     = scoring,
                    refit       = "f1",
                    error_score = "raise",
                    n_jobs      = 1,
                    verbose     = 10
                )
                search.fit(X_train, y_train, groups=grp_train)
                best_est = search.best_estimator_

                cv_res = pd.DataFrame(search.cv_results_)
                pos_mask  = y_test == 1
                neg_mask  = ~pos_mask

                proba_raw = best_est.predict_proba(X_test)[:, 1]
                def _stats(m): return f"min={m.min():.3f}, mean={m.mean():.3f}, max={m.max():.3f}"
                print(f"Positives ({pos_mask.sum()} échantillons) → {_stats(proba_raw[pos_mask])}\n"
                      f"Négatifs  ({neg_mask.sum()} échantillons) → {_stats(proba_raw[neg_mask])}")

                best_width = best_est.width
                best_thr   = best_est.threshold
                cols_params = [c for c in cv_res.columns if c.startswith("param_")]
                top_k = (cv_res.sort_values("mean_test_f1", ascending=False)
                              .head(5)[["mean_test_recall", "mean_test_f1"] + cols_params])
                print("\n TOP 5 hyper-params (inner CV) pour ce fold")
                print(top_k.to_string(index=False))

                for _, row in cv_res.iterrows():
                    key = tuple(sorted(row[[c for c in cols_params]].items()))
                    inner_summary[key].append(row["mean_test_f1"])

                imp_dict = _feature_importances(best_est, list(X.columns))
                if imp_dict:
                    feat_imps[name].update(imp_dict)

                t0 = perf_counter()
                y_pred_fp = best_est.predict(X_test)
                comput_latency = (perf_counter() - t0) / max(1, len(X_test))

                pct_pos_pred = np.mean(y_pred_fp[y_test == 1] == 1) * 100
                pct_neg_pred = np.mean(y_pred_fp[y_test == 0] == 1) * 100
                print(f"    Avec w={best_width}, thr={best_thr:.2f} : "
                      f"{pct_pos_pred:.1f}% des pos, {pct_neg_pred:.1f}% des neg prédit '1'")

                # Préparation par fold (seconds, bornes, mapping) 
                recs_fold = X.index.get_level_values("record")[test_idx].to_numpy()
                times_sec_fold = _elapsed_seconds_by_record(recs_fold, STEP)

                starts, ends = {}, {}
                _last = None
                for i, r in enumerate(recs_fold):
                    if r != _last:
                        if _last is not None:
                            ends[_last] = i
                        starts[r] = i
                        _last = r
                if _last is not None:
                    ends[_last] = len(recs_fold)

                index_of = {}
                for idx_i, (r_i, s_i) in enumerate(zip(recs_fold, times_sec_fold)):
                    index_of.setdefault(r_i, {})[int(s_i)] = idx_i

                # Événements prédits/vrais en secondes 
                pred_evts_sec = vec_to_events_seconds(y_pred_fp,     recs_fold, STEP)
                true_evts_sec = vec_to_events_seconds(y_test.values, recs_fold, STEP)

                # Extraction TP/FP + latences pour TP 
                tp_evts, fp_evts = [], []

                def _overlap_len(p_start, p_end, t_start, t_end):
                    return max(0, min(p_end, t_end) - max(p_start, t_start))

                for prec, p_start_s, p_end_s in pred_evts_sec:
                    # candidats GT qui chevauchent
                    cand_true = [
                        (trec, t_start_s, t_end_s)
                        for trec, t_start_s, t_end_s in true_evts_sec
                        if (prec == trec) and overlap_seconds(
                            p_start_s, p_end_s, t_start_s, t_end_s,
                            tol_start_s=evt_param.toleranceStart,
                            tol_end_s=evt_param.toleranceEnd
                        )
                    ]
                    if not cand_true:
                        fp_evts.append((fold_idx, prec, p_start_s, p_end_s))
                        continue

                    cand_true.sort(
                        key=lambda x: (-_overlap_len(p_start_s, p_end_s, x[1], x[2]),
                                       abs(x[1] - p_start_s))
                    )
                    trec, t_start_s, t_end_s = cand_true[0]

                    gt_start_idx = index_of.get(trec, {}).get(int(t_start_s))
                    gt_end_idx   = index_of.get(trec, {}).get(int(t_end_s))
                    if gt_start_idx is None or gt_end_idx is None:
                        tp_evts.append((fold_idx, prec, p_start_s, p_end_s, None, None))
                        continue

                    rec_i0, rec_i1 = starts[trec], ends[trec]
                    win_start_idx = max(rec_i0, gt_start_idx - evt_param.toleranceStart // STEP)
                    win_end_idx   = min(rec_i1, gt_end_idx + 1 + evt_param.toleranceEnd // STEP)

                    j_hit = None
                    for j in range(win_start_idx, win_end_idx):
                        if y_pred_fp[j] == 1:
                            j_hit = j
                            break

                    det_latency_s = None
                    if j_hit is not None:
                        det_latency_s = max(0, (j_hit - gt_start_idx) * STEP)

                    gen_latency_s = None
                    on_arr, gen_off_arr = ann_cache.get(trec, (np.array([]), np.array([])))
                    if det_latency_s is not None and on_arr.size:
                        k = int(np.argmin(np.abs(on_arr - int(t_start_s))))
                        if abs(int(on_arr[k]) - int(t_start_s)) <= 30:
                            gen_off = int(gen_off_arr[k]) if gen_off_arr.size else 0
                            gen_latency_s = max(0, det_latency_s - gen_off)

                    tp_evts.append((fold_idx, prec, p_start_s, p_end_s, det_latency_s, gen_latency_s))

                # Faux négatifs (événements GT non couverts) 
                fn_evts = []
                for trec, t_start_s, t_end_s in true_evts_sec:
                    matched = any(
                        (prec == trec) and overlap_seconds(
                            p_start_s, p_end_s, t_start_s, t_end_s,
                            tol_start_s=evt_param.toleranceStart,
                            tol_end_s=evt_param.toleranceEnd
                        )
                        for prec, p_start_s, p_end_s in pred_evts_sec
                    )
                    if not matched:
                        fn_evts.append((fold_idx, trec, t_start_s, t_end_s))

                # Stockage global
                all_tp.extend(tp_evts)
                all_fp.extend(fp_evts)
                all_fn.extend(fn_evts)

                # GT-centré : une ligne par crise vraie 
                for trec, t_start_s, t_end_s in true_evts_sec:
                    gt_start_idx = index_of.get(trec, {}).get(int(t_start_s))
                    gt_end_idx   = index_of.get(trec, {}).get(int(t_end_s))
                    if gt_start_idx is None or gt_end_idx is None:
                        all_gt.append((fold_idx, trec, t_start_s, t_end_s, 0, None, None))
                        continue

                    rec_i0, rec_i1 = starts[trec], ends[trec]
                    win_start_idx = max(rec_i0, gt_start_idx - evt_param.toleranceStart // STEP)
                    win_end_idx   = min(rec_i1, gt_end_idx + 1 + evt_param.toleranceEnd // STEP)

                    j_hit = None
                    for j in range(win_start_idx, win_end_idx):
                        if y_pred_fp[j] == 1:
                            j_hit = j
                            break

                    if j_hit is None:
                        all_gt.append((fold_idx, trec, t_start_s, t_end_s, 0, None, None))
                    else:
                        det_latency_s = max(0, (j_hit - gt_start_idx) * STEP)
                        gen_latency_s = None
                        on_arr, gen_off_arr = ann_cache.get(trec, (np.array([]), np.array([])))
                        if on_arr.size:
                            k = int(np.argmin(np.abs(on_arr - int(t_start_s))))
                            if abs(int(on_arr[k]) - int(t_start_s)) <= 30:
                                gen_off = int(gen_off_arr[k]) if gen_off_arr.size else 0
                                gen_latency_s = max(0, det_latency_s - gen_off)
                        all_gt.append((fold_idx, trec, t_start_s, t_end_s, 1, det_latency_s, gen_latency_s))

                # Métriques epoch & event 
                recall_epoch = recall_score(y_test, y_pred_fp, zero_division=0)
                precision_epoch = precision_score(y_test, y_pred_fp, zero_division=0)
                f1_epoch = f1_score(y_test, y_pred_fp, zero_division=0)
                print(f"Fold terminé en {time.time() - tic:.1f} s")

                fs = 1 / STEP
                ref = Annotation(list(y_test.astype(int)), fs)
                hyp = Annotation(list(y_pred_fp.astype(int)), fs)
                evt_score = ts_scoring.EventScoring(ref, hyp, evt_param)

                tp_evt  = evt_score.tp
                fp_evt  = evt_score.fp
                far_24h = evt_score.fpRate
                precision_evt = evt_score.precision
                recall_evt    = evt_score.sensitivity
                f1_evt        = evt_score.f1

                if fold_idx == 1 and name == "XGBoost":
                    n_test = len(test_idx)
                    duration_h = (n_test * STEP) / 3600.0
                    far_24h_manual = (fp_evt * 24.0) / duration_h
                    print(f"[CHECK] FAR24h(lib)={far_24h:.3f} | FAR24h(manuel)={far_24h_manual:.3f} "
                          f"| n_test={n_test} | duration_h={duration_h:.2f}")

                lat_tp = detection_latencies(y_test.values, y_pred_fp,
                                             tol_start=evt_param.toleranceStart,
                                             tol_end=evt_param.toleranceEnd,
                                             min_len=2, step=STEP)
                lat_gen = gen_latencies_aligned(
                    y_true=y_test.values, y_pred=y_pred_fp,
                    recs=recs_fold, step_sec=STEP,
                    events_index=events_index,
                    tol_start_s=evt_param.toleranceStart,
                    tol_end_s=evt_param.toleranceEnd,
                    onset_match_tol_s=30,
                )

                avg_det_latency = np.mean(lat_tp) if lat_tp else np.nan
                avg_gen_latency = np.mean(lat_gen) if lat_gen else np.nan

                fold_stats.append({
                    "f1": float(f1_evt),
                    "f1_epoch": float(f1_epoch),
                    "precision": float(precision_evt),
                    "precision_epoch": float(precision_epoch),
                    "recall": float(recall_evt),
                    "recall_epoch": float(recall_epoch),
                    "sens_evt": float(recall_evt),
                    "fp_evt": float(fp_evt),
                    "far_evt": float(far_24h),
                    "width": float(best_width),
                    "threshold": float(best_thr),
                    "detection_latency": float(avg_det_latency) if not np.isnan(avg_det_latency) else None,
                    "gen_latency": float(avg_gen_latency) if not np.isnan(avg_gen_latency) else None,
                    "comput_latency": float(comput_latency),
                })

                print(f"Fold {fold_idx}/{OUTER_SPLITS} terminé en {time.time() - tic:.1f}s |"
                      f" F1_evt {f1_evt:.3f} | P {precision_evt:.3f} | R {recall_evt:.3f} | "
                      f"R_ep {recall_epoch:.3f} | F1_ep {f1_epoch:.3f} | P_ep {precision_epoch:.3f} | "
                      f"FP {fp_evt} | FAR24h {far_24h:.3f} | "
                      f"LatDétect {avg_det_latency:.1f}s | GenLat {avg_gen_latency:.2f}s | "
                      f"LatPréd {comput_latency*1e3:.1f} ms")

            # Agrégation
            def _mean(key):
                return float(np.mean([fsd[key] for fsd in fold_stats]))

            results[name] = {
                "f1": _mean("f1"),
                "f1_epoch": _mean("f1_epoch"),
                "precision": _mean("precision"),
                "precision_epoch": _mean("precision_epoch"),
                "recall": _mean("recall"),
                "recall_epoch": _mean("recall_epoch"),
                "sens_evt": _mean("sens_evt"),
                "fp_evt": _mean("fp_evt"),
                "far_evt": _mean("far_evt"),
                "comput_latency": _mean("comput_latency"),
                "folds": fold_stats,
                "detection_latency": _mean("detection_latency"),
                "gen_latency": _mean("gen_latency"),
            }

            print(f"{name:<10s} | F1_evt {results[name]['f1']:.3f} | "
                  f"P {results[name]['precision']:.3f} | R {results[name]['recall']:.3f} | "
                  f"FP {results[name]['fp_evt']:.3f} | FAR24h {results[name]['far_evt']:.3f} | "
                  f"LatDétect {results[name]['detection_latency']:.1f}s | "
                  f"GenLat {results[name]['gen_latency']:.1f}s | "
                  f"LatPréd {results[name]['comput_latency']*1e3:.1f} ms")

            inner_summary_all_models[name] = inner_summary

            # Écriture CSV détaillés (TP/FP/FN) + GT-centré 
            tp_df = pd.DataFrame(
                all_tp,
                columns=["fold", "record", "start_s", "end_s", "det_latency_s", "gen_latency_s"]
            )
            fp_df = pd.DataFrame(all_fp, columns=["fold", "record", "start_s", "end_s"])
            fn_df = pd.DataFrame(all_fn, columns=["fold", "record", "start_s", "end_s"])
            gt_df = pd.DataFrame(
                all_gt,
                columns=["fold","record","start_s","end_s","detected","det_latency_s","gen_latency_s"]
            )

            tp_df["duration_s"] = tp_df["end_s"] - tp_df["start_s"]
            fp_df["duration_s"] = fp_df["end_s"] - fp_df["start_s"]
            fn_df["duration_s"] = fn_df["end_s"] - fn_df["start_s"]
            gt_df["duration_s"] = gt_df["end_s"] - gt_df["start_s"]

            fname_tp = export_dir / f"{name}_true_positives.csv"
            fname_fp = export_dir / f"{name}_false_positives.csv"
            fname_fn = export_dir / f"{name}_false_negatives.csv"
            fname_gt = export_dir / f"{name}_true_events.csv"

            tp_df.to_csv(fname_tp, index=False)
            fp_df.to_csv(fname_fp, index=False)
            fn_df.to_csv(fname_fn, index=False)
            gt_df.to_csv(fname_gt, index=False)

            print(f"\nRésumé FP par fold :\n{fp_df.groupby('fold')['record'].count()}")
            print(f"Résumé FN par fold :\n{fn_df.groupby('fold')['record'].count()}")
            print(f"Résumé GT détectés par fold :\n{gt_df.groupby('fold')['detected'].sum()}")
            print("Fichiers détaillés :", fname_tp.name, ",", fname_fp.name, ",", fname_fn.name, ",", fname_gt.name)

        except Exception as e:
            failures[name] = str(e)
            print(f"   Échec {name}: {e}")

    # Feature importances agrégées (si dispos)
    feat_summary = {}
    for mdl, counter in feat_imps.items():
        total = sum(counter.values()) or 1.0
        avg_imp = {f: float(v / total) for f, v in counter.items()}
        top = [(f, float(v)) for f, v in sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)]
        feat_summary[mdl] = top
        print(f"\n--- Top features pour {mdl} ---")
        for f, v in top[:30]:
            print(f"{f:<30s}: {v:.4f}")

    # Récap hyperparams
    print("\nRÉCAPITULATIF DES HYPER-PARAMÈTRES (inner-CV) ")
    for mdl, summary in inner_summary_all_models.items():
        print(f"\n {mdl}")
        for key, scores in sorted(summary.items(), key=lambda x: -np.mean(x[1]))[:10]:
            params = {k.replace('param_', ''): v for k, v in key}
            print(f"  F1={np.mean(scores):.3f} ±{np.std(scores):.3f}  → {params}")

    # Résumé global
    print("\n RÉCAPITULATIF EVENT-BASED ")
    for mdl, res in results.items():
        print(
            f"{mdl:<12s} │ F1_evt {res['f1']:.3f} │ F1_ep {res['f1_epoch']:.3f} | "
            f"P {res['precision']:.3f} │ P_epoch {res['precision_epoch']:.3f} │ "
            f"R {res['recall']:.3f} │ R_ep {res['recall_epoch']:.3f} | "
            f"Sens {res['sens_evt']:.3f} │ FP {res['fp_evt']:.3f}| FAR24h {res['far_evt']:.3f} │ "
            f"LatDétect {res['detection_latency']:.1f}s │ GenLat {res['gen_latency']:.1f}s │ "
            f"LatPréd {res['comput_latency']*1e3:.1f} ms"
        )

    (export_dir / "results.json").write_text(json.dumps(results, indent=2))
    (export_dir / "failures.json").write_text(json.dumps(failures, indent=2))
    (export_dir / "feature_importances.json").write_text(json.dumps(feat_summary, indent=2))
    print("\nRésultats sauvegardés dans", export_dir.resolve())

    return results, failures, feat_summary


def main():
    parser = argparse.ArgumentParser(description="Train models with event-based scoring.")
    parser.add_argument("--win", type=int, default=DEFAULT_WIN, help="Window size (seconds) to load dataset for.")
    parser.add_argument("--export-dir", type=str, default=None, help="Where to save results.")
    args = parser.parse_args()

    win = int(args.win)
    try:
        X, y, groups = load_dataset(win)
    except FileNotFoundError:
        print(f" Dataset {win}s absent – as-tu lancé le pipeline et généré le cache ?")
        return

    out_dir = Path(args.export_dir) if args.export_dir else (RESULT_ROOT / f"{win}s")
    train_multiple_models(X, y, groups, export_dir=out_dir)

if __name__ == "__main__":
    main()
