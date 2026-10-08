"""
Model Evaluation and Comparison Script.
Evaluates final XGBoost models on the official DEV (val) and EVAL (test) partitions using Validation-tuned thresholds,
computes Bootstrap 95% CIs, performs Unseen Attack Generalization (A05-A06),
and generates Confusion Matrix, ROC Curve (with EER), and CNN vs XGBoost comparative metrics.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from src.config import (
    FEATURES_DIR,
    MODELS_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    calculate_eer
)
from src.train_xgb_baseline import load_feature_splits, evaluate_predictions, compute_bootstrap_ci


def main():
    print("=" * 60)
    print("STAGE 4 & 5: Final Evaluation & CNN vs XGBoost Benchmark Comparison")
    print("=" * 60)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols = load_feature_splits()
    
    # Load winning model and features list
    best_model_path = os.path.join(MODELS_DIR, "xgb_best.json")
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(MODELS_DIR, "xgb_prepruned.json")
        
    best_xgb = XGBClassifier()
    best_xgb.load_model(best_model_path)
    
    root_features_path = os.path.join(os.path.dirname(MODELS_DIR), "..", "FINAL_xgboost_features.json")
    if os.path.exists(root_features_path):
        with open(root_features_path, "r") as f:
            winning_features = json.load(f)
    else:
        winning_features = feature_cols
        
    X_val_sub = X_val[winning_features]
    X_test_sub = X_test[winning_features]
    
    val_prob = best_xgb.predict_proba(X_val_sub)[:, 1]
    test_prob = best_xgb.predict_proba(X_test_sub)[:, 1]
    
    # DEV (Validation) Metrics & Threshold Tuning
    val_metrics = evaluate_predictions(y_val, val_prob)
    val_eer_thresh = val_metrics["EER_Threshold"]
    print(f"\nDEV (Validation) EER: {val_metrics['EER']:.4f} at Frozen Threshold: {val_eer_thresh:.4f}")
    
    # EVAL (Test) Metrics with Frozen DEV Threshold
    test_pred_val_thresh = (test_prob >= val_eer_thresh).astype(int)
    test_metrics = evaluate_predictions(y_test, test_prob, threshold=val_eer_thresh)
    test_ci = compute_bootstrap_ci(y_test, test_prob)
    
    print("\n--- FINAL WINNING XGBOOST TEST (EVAL) METRICS (Frozen DEV Threshold) ---")
    for k, v in test_metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print(f"  Test ROC-AUC 95% CI : {test_ci['ROC-AUC_CI95']}")
    print(f"  Test EER 95% CI     : {test_ci['EER_CI95']}")
        
    final_res_dict = {**test_metrics, **test_ci}
    final_res_df = pd.DataFrame([final_res_dict])
    final_res_df.to_csv(os.path.join(METRICS_DIR, "final_xgboost_results.csv"), index=False)
    
    # -------------------------------------------------------------
    # CNN vs XGBoost Detailed Multi-Partition Benchmark Comparison Table
    # Reference CNN numbers from project mid-review presentation:
    # - CNN DEV : Accuracy = 0.9800, Macro F1 = 0.9300, ROC-AUC = 0.9909, EER = 0.0518
    # - CNN EVAL: Accuracy = 0.7700, Macro F1 = 0.6300, Spoof Recall = 0.7600, Bonafide Precision = 0.2800
    # -------------------------------------------------------------
    cnn_dev_ref = {
        "Model": "CNN Reference",
        "Split": "DEV (In-Distribution)",
        "Accuracy": 0.9800,
        "Macro_F1": 0.9300,
        "ROC-AUC": 0.9909,
        "EER": 0.0518
    }
    xgb_dev_res = {
        "Model": "XGBoost Pipeline",
        "Split": "DEV (In-Distribution)",
        "Accuracy": val_metrics["Accuracy"],
        "Macro_F1": val_metrics["Macro_F1"],
        "ROC-AUC": val_metrics["ROC-AUC"],
        "EER": val_metrics["EER"]
    }
    cnn_eval_ref = {
        "Model": "CNN Reference",
        "Split": "EVAL (Unseen Data)",
        "Accuracy": 0.7700,
        "Macro_F1": 0.6300,
        "ROC-AUC": np.nan,
        "EER": np.nan
    }
    xgb_eval_res = {
        "Model": "XGBoost Pipeline",
        "Split": "EVAL (Test Set)",
        "Accuracy": test_metrics["Accuracy"],
        "Macro_F1": test_metrics["Macro_F1"],
        "ROC-AUC": test_metrics["ROC-AUC"],
        "EER": test_metrics["EER"]
    }
    
    comp_df = pd.DataFrame([cnn_dev_ref, xgb_dev_res, cnn_eval_ref, xgb_eval_res])
    comp_csv_path = os.path.join(METRICS_DIR, "cnn_vs_xgboost_results.csv")
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"\nCNN vs XGBoost multi-partition comparison saved to {comp_csv_path}")
    print(comp_df.to_string(index=False))
    
    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_test, test_pred_val_thresh)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Bonafide", "Spoof"],
        yticklabels=["Bonafide", "Spoof"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Final XGBoost Confusion Matrix (EVAL)")
    plt.tight_layout()
    
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nConfusion matrix saved to {cm_path}")
    
    # 2. ROC Curve Plot with EER annotation
    fpr, tpr, thresholds = roc_curve(y_test, test_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"XGBoost EVAL (AUC = {test_metrics['ROC-AUC']:.4f})", linewidth=2, color="#1f77b4")
    plt.plot([0, 1], [0, 1], linestyle=":", color="gray", label="Random Chance")
    
    eer = test_metrics["EER"]
    plt.scatter([eer], [1 - eer], color="red", zorder=5, label=f"EER = {eer*100:.2f}%")
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Final XGBoost ROC Curve (EVAL)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    roc_path = os.path.join(PLOTS_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved to {roc_path}")
    
    # 3. CNN vs XGBoost Comparison Plot
    plt.figure(figsize=(10, 6))
    metrics_list = ["Accuracy", "Macro_F1", "ROC-AUC"]
    x = np.arange(len(metrics_list))
    width = 0.35
    
    dev_xgb_vals = [val_metrics["Accuracy"], val_metrics["Macro_F1"], val_metrics["ROC-AUC"]]
    test_xgb_vals = [test_metrics["Accuracy"], test_metrics["Macro_F1"], test_metrics["ROC-AUC"]]
    
    plt.bar(x - width/2, dev_xgb_vals, width, label="XGBoost DEV", color="#1f77b4")
    plt.bar(x + width/2, test_xgb_vals, width, label="XGBoost EVAL", color="#ff7f0e")
    
    plt.xticks(x, metrics_list, rotation=15)
    plt.ylabel("Score")
    plt.title("XGBoost Performance: DEV vs EVAL Partitions")
    plt.ylim(0.0, 1.05)
    plt.legend(loc="lower right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    
    comp_plot_path = os.path.join(PLOTS_DIR, "cnn_vs_xgboost.png")
    plt.savefig(comp_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"CNN vs XGBoost comparison plot saved to {comp_plot_path}")

    # 4. Optional Attack Generalization Experiment (Separate Report Metric)
    print("\n--- OPTIONAL ATTACK GENERALIZATION STUDY (A01-A04 vs A05-A06) ---")
    train_meta_path = os.path.join(FEATURES_DIR, "mel_xgb_train.csv")
    val_meta_path = os.path.join(FEATURES_DIR, "mel_xgb_val.csv")
    test_meta_path = os.path.join(FEATURES_DIR, "mel_xgb_test.csv")
    
    all_data = pd.concat([pd.read_csv(train_meta_path), pd.read_csv(val_meta_path), pd.read_csv(test_meta_path)], ignore_index=True)
    train_seen_df = all_data[all_data["attack_id"].isin(["-", "A01", "A02", "A03", "A04"])].copy()
    test_unseen_df = all_data[all_data["attack_id"].isin(["-", "A05", "A06"])].copy()
    
    if len(train_seen_df) > 0 and len(test_unseen_df) > 0:
        X_tr_seen = train_seen_df[winning_features]
        y_tr_seen = train_seen_df["label_id"]
        X_te_unseen = test_unseen_df[winning_features]
        y_te_unseen = test_unseen_df["label_id"]
        
        spw = np.sum(y_tr_seen == 0) / max(1, np.sum(y_tr_seen == 1))
        xgb_unseen = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            learning_rate=0.03,
            scale_pos_weight=spw,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        xgb_unseen.fit(X_tr_seen, y_tr_seen)
        unseen_prob = xgb_unseen.predict_proba(X_te_unseen)[:, 1]
        unseen_metrics = evaluate_predictions(y_te_unseen, unseen_prob)
        
        print("Unseen Attack (A05-A06) Performance:")
        print(f"  Test ROC-AUC  : {unseen_metrics['ROC-AUC']:.4f}")
        print(f"  Test EER      : {unseen_metrics['EER']:.4f}")
        
        att_df = pd.DataFrame([unseen_metrics])
        att_df.to_csv(os.path.join(METRICS_DIR, "unseen_attack_generalization.csv"), index=False)


if __name__ == "__main__":
    main()
