"""
Stage A: Baseline XGBoost Model Training and Evaluation.
Trains unpruned baseline XGBoost classifier with initial hyperparameters and class weighting.
Computes complete metrics including Bonafide Specificity, Macro F1, and Bootstrap 95% Confidence Intervals.
"""

import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample

from src.config import (
    FEATURES_DIR,
    MODELS_DIR,
    METRICS_DIR,
    RANDOM_STATE,
    calculate_eer
)


def load_feature_splits():
    train_df = pd.read_csv(os.path.join(FEATURES_DIR, "mel_xgb_train.csv"))
    val_df = pd.read_csv(os.path.join(FEATURES_DIR, "mel_xgb_val.csv"))
    test_df = pd.read_csv(os.path.join(FEATURES_DIR, "mel_xgb_test.csv"))
    
    meta_cols = ["file_id", "speaker_id", "attack_id", "label", "label_id"]
    ref_df = train_df if len(train_df) > 0 else val_df
    feature_cols = [c for c in ref_df.columns if c not in meta_cols]
    
    if len(train_df) == 0:
        raise ValueError(
            "TRAIN dataset features are empty. Please ensure 'ASVspoof2019_LA_train' "
            "directory containing FLAC files is placed in ASVspoof_Data/LA/LA and re-run extraction."
        )
        
    stds = train_df[feature_cols].std()
    active_cols = stds[stds >= 1e-6].index.tolist()
    
    X_train = train_df[active_cols]
    y_train = train_df["label_id"]
    
    X_val = val_df[active_cols]
    y_val = val_df["label_id"]
    
    X_test = test_df[active_cols] if len(test_df) > 0 else pd.DataFrame(columns=active_cols)
    y_test = test_df["label_id"] if len(test_df) > 0 else pd.Series(dtype=int)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), active_cols


def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    
    # Spoof (1) Metrics
    spoof_prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    spoof_rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    spoof_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # Bonafide (0) Metrics / Specificity
    bonafide_prec = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    bonafide_rec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    bonafide_f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    eer, eer_thresh = calculate_eer(y_true, y_prob)
    
    return {
        "Accuracy": float(acc),
        "Spoof_Precision": float(spoof_prec),
        "Spoof_Recall": float(spoof_rec),
        "Spoof_F1": float(spoof_f1),
        "Bonafide_Precision": float(bonafide_prec),
        "Bonafide_Recall": float(bonafide_rec),
        "Bonafide_F1": float(bonafide_f1),
        "Macro_F1": float(macro_f1),
        "ROC-AUC": float(auc),
        "EER": float(eer),
        "EER_Threshold": float(eer_thresh)
    }


def compute_bootstrap_ci(y_true, y_prob, n_bootstraps=1000, ci=95):
    """
    Computes 95% Bootstrap Confidence Intervals for ROC-AUC and EER.
    """
    np.random.seed(RANDOM_STATE)
    auc_scores = []
    eer_scores = []
    
    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    n_samples = len(y_true_arr)
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_t_boot = y_true_arr[indices]
        y_p_boot = y_prob_arr[indices]
        
        if len(np.unique(y_t_boot)) < 2:
            continue
            
        auc_scores.append(roc_auc_score(y_t_boot, y_p_boot))
        eer, _ = calculate_eer(y_t_boot, y_p_boot)
        eer_scores.append(eer)
        
    lower_p = (100 - ci) / 2.0
    upper_p = 100 - lower_p
    
    auc_ci = (float(np.percentile(auc_scores, lower_p)), float(np.percentile(auc_scores, upper_p)))
    eer_ci = (float(np.percentile(eer_scores, lower_p)), float(np.percentile(eer_scores, upper_p)))
    
    return {"ROC-AUC_CI95": auc_ci, "EER_CI95": eer_ci}


def main():
    print("=" * 60)
    print("STAGE A: Baseline XGBoost Classifier Training")
    print("=" * 60)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols = load_feature_splits()
    print(f"Active Features Retained: {len(feature_cols)}")
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
    
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / max(1, pos_count)
    print(f"Class distribution: Bonafide (0)={neg_count}, Spoof (1)={pos_count}")
    print(f"Computed scale_pos_weight: {scale_pos_weight:.4f}")
    
    baseline_xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    print("\nTraining Baseline XGBoost...")
    baseline_xgb.fit(X_train, y_train)
    
    train_prob = baseline_xgb.predict_proba(X_train)[:, 1]
    val_prob = baseline_xgb.predict_proba(X_val)[:, 1]
    test_prob = baseline_xgb.predict_proba(X_test)[:, 1]
    
    train_metrics = evaluate_predictions(y_train, train_prob)
    val_metrics = evaluate_predictions(y_val, val_prob)
    
    # Validation-tuned EER threshold
    val_thresh = val_metrics["EER_Threshold"]
    test_metrics = evaluate_predictions(y_test, test_prob, threshold=val_thresh)
    test_ci = compute_bootstrap_ci(y_test, test_prob)
    
    print("\n--- BASELINE RESULTS ---")
    print("Validation Metrics:")
    for k, v in val_metrics.items():
        print(f"  {k:20s}: {v:.4f}")
        
    print("\nTest Metrics (Validation-Tuned Threshold):")
    for k, v in test_metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print(f"  Test ROC-AUC 95% CI : {test_ci['ROC-AUC_CI95']}")
    print(f"  Test EER 95% CI     : {test_ci['EER_CI95']}")
    
    model_path = os.path.join(MODELS_DIR, "xgb_baseline.json")
    baseline_xgb.save_model(model_path)
    print(f"\nBaseline model saved to {model_path}")
    
    metrics_path = os.path.join(METRICS_DIR, "baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "test_ci95": test_ci
        }, f, indent=4)
    print(f"Baseline metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
