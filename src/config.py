"""
Configuration file containing paths, parameters, constants, and shared utilities for the XGBoost Mel-Spectrogram Deepfake Speech Detection Pipeline.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

# Base Directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_BASE = os.path.join(BASE_DIR, "ASVspoof_Data", "LA", "LA")

# Official Protocol Directory
PROTOCOL_DIR = os.path.join(DATASET_BASE, "ASVspoof2019_LA_cm_protocols")

# Official Audio FLAC Directories for TRAIN, DEV, and EVAL
FLAC_TRAIN_DIR = os.path.join(DATASET_BASE, "ASVspoof2019_LA_train", "flac")
FLAC_DEV_DIR = os.path.join(DATASET_BASE, "ASVspoof2019_LA_dev", "flac")
FLAC_EVAL_DIR = os.path.join(DATASET_BASE, "ASVspoof2019_LA_eval", "flac")

# Project Output Directories
DATA_SPLITS_DIR = os.path.join(BASE_DIR, "data_splits")
FEATURES_DIR = os.path.join(BASE_DIR, "features")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
METRICS_DIR = os.path.join(OUTPUTS_DIR, "metrics")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")

# Ensure all output directories exist
for d in [DATA_SPLITS_DIR, FEATURES_DIR, OUTPUTS_DIR, MODELS_DIR, METRICS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Audio & Preprocessing Parameters (Matching CNN Baseline Identically)
TARGET_SR = 16000
DURATION = 4  # seconds
TARGET_LENGTH = TARGET_SR * DURATION  # 64,000 samples

N_FFT = 512
HOP_LENGTH = 160  # 10 ms frame shift
N_MELS = 128
FMIN = 0
FMAX = 8000
TRIM_TOP_DB = 30

RANDOM_STATE = 42

def calculate_eer(y_true, y_score):
    """
    Computes Equal Error Rate (EER) and the optimal EER threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2.0
    eer_threshold = thresholds[eer_index]
    
    return float(eer), float(eer_threshold)
