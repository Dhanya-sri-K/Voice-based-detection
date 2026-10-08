# Voice-Based Deepfake Detection using XGBoost & Mel-Spectrogram Analysis

Official implementation of the **XGBoost Mel-Spectrogram Deepfake Speech Detection Pipeline** for the ASVspoof 2019 Logical Access (LA) dataset.

## Features & Architecture

- **Signal Preprocessing**: Standardizes 16 kHz sampling rate, 4.0-second fixed duration, silence trimming (30 dB), and normalization.
- **768 Mel Feature Representation**: Computes log-Mel spectrograms across 128 Mel bands and extracts 6 temporal statistics per band (mean, std, min, max, median, IQR).
- **Partition Alignment**: Strict speaker-disjoint mapping across TRAIN, DEV (Validation), and EVAL (Test) partitions.
- **Pre-Pruning & Tree Growth Control**: Hyperparameter search with `GroupKFold` CV and validation early stopping to prevent over-fitting.
- **Feature Selection & Ablation Study**: SHAP feature ranking and evaluation on Top 100%, 75%, 50%, and 25% feature subsets.
- **SHAP Interpretability**: Global beeswarm plots, local waterfall explanations for Real/Spoof audio, and Mel-band frequency importance heatmaps (Hz).

## Project Structure

```
├── src/
│   ├── config.py                 # Configuration parameters, paths, and EER calculation
│   ├── extract_mel_features.py   # Audio loading, Mel spectrogram & 768 feature extraction
│   ├── train_xgb_baseline.py     # Stage A: Baseline XGBoost model training
│   ├── tune_xgb.py               # Stage B: Pre-pruning & tree growth control tuning
│   ├── prune_features.py         # Stage C: Feature pruning ablation study
│   ├── evaluate_xgb.py           # Evaluation, frozen threshold metrics & CNN comparison
│   └── explain_shap.py           # SHAP interpretability & Mel frequency heatmaps
├── run_pipeline.py               # End-to-end pipeline orchestrator
├── FINAL_xgboost_model.json      # Final trained winning XGBoost model
├── FINAL_xgboost_features.json     # Feature list of winning model
└── outputs/
    ├── metrics/                  # CSV & JSON performance metrics
    └── plots/                    # Evaluation & SHAP visualization plots
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete End-to-End Pipeline
```bash
python run_pipeline.py
```

## Results Summary

| Model | Partition | Accuracy | Macro F1 | ROC-AUC | EER (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **CNN Reference** | DEV (In-Distribution) | 98.00% | 0.9300 | 0.9909 | 5.18% |
| **XGBoost Pipeline** | DEV (Validation Set) | 84.75% | 0.6906 | 0.9381 | 14.66% |
| **CNN Reference** | EVAL (Unseen Test Set) | 77.00% | 0.6300 | N/A | N/A |
| **XGBoost Pipeline** | **EVAL (Final Test Set)** | **87.88%** | **0.7453** | **0.9521** | **11.25%** |

*XGBoost outperforms the CNN baseline on unseen test evaluation by **+10.88% Accuracy** and **+0.1153 Macro F1**.*
