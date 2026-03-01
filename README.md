# Real-Time Seizure Detection from Wearable Physiological Signals

ML pipeline for epileptic seizure detection using Hexoskin wearable signals, developed during a clinical research internship at **CRCHUM** (Centre de Recherche du Centre Hospitalier de l'Université de Montréal).

> Data is not included in this repository (clinical data — confidential).

---

## Overview

This project builds an end-to-end pipeline that takes raw physiological recordings from a Hexoskin wearable vest and trains classifiers to detect epileptic seizures in real time. The pipeline covers everything from raw signal preprocessing to model training and clinical-grade evaluation.

**Signals used:** Heart rate · Respiration · Accelerometry (3-axis)  
**Target:** Binary seizure/non-seizure classification over 15-second sliding windows

---

## Pipeline

```
Raw Hexoskin CSV
      ↓
  Preprocessing → 1 Hz resampling, forward-fill, Parquet export
      ↓
  Labeling → seizure event annotation alignment (onset/duration/type)
      ↓
  Feature Extraction → time-domain & frequency-domain features per window
      ↓
  Dataset Builder → sliding window (15s, stride 5s), X/y/groups cache
      ↓
  Model Training → nested cross-validation, multiple classifiers
      ↓
  Evaluation → ROC-AUC, F1, clinical event-level scoring
```

Run the full pipeline with:
```bash
python Pipeline.py
```

---

## Models & Evaluation

Three classifiers trained and compared:
- **XGBoost + FiringWrapper** *(best performer — saved in `models/`)*
- Logistic Regression
- SVM

Training uses **nested cross-validation** (7 outer folds × 6 inner folds) with `StratifiedGroupKFold` to prevent subject leakage across folds. Models are optimized for **F1 score** and assessed with clinical event-level scoring via the `timescoring` library.

---

## Project Structure

```
├── Pipeline.py                    # Main entry point — runs full pipeline
├── final_model_training.py        # Standalone script for final model export
├── src/training/
│   ├── Preprocessing.py           # Raw CSV → 1 Hz Parquet
│   ├── label.py                   # Seizure annotation alignment
│   ├── extract_features.py        # Feature extraction per window
│   ├── data_loader.py             # Dataset builder (X, y, groups)
│   ├── Multiple_model_training.py # Nested CV training & evaluation
│   └── config.py                  # All paths and hyperparameters
└── models/15s/
    ├── model.joblib               # Trained XGBoost + FiringWrapper
    ├── feature_names.json         # Feature list for inference
    └── metadata.json              # Training metadata
```

---

## Setup

```bash
git clone https://github.com/<your-username>/Real-time-hexoskin-seizure-detection.git
cd Real-time-hexoskin-seizure-detection
pip install -r requirements.txt
```

Update paths in `src/training/config.py` to point to your local data directories.

---

## Tech Stack

**Python · scikit-learn · XGBoost · pandas · NumPy · timescoring**

---

## Context

Developed as part of a clinical research internship at CRCHUM (May–Aug 2025), under the supervision of the epilepsy monitoring unit. The goal was to assess the feasibility of automated seizure detection from ambulatory wearable data for patients with drug-resistant epilepsy.
