"""
Stage C: Feature Pruning Ablation Study.
Ranks features using Gain and SHAP, retrains models on Top 75%, Top 50%, and Top 25% feature subsets
REUSING tuned pre-pruning hyperparameters and early stopping.
Selects the winning pruned model based on validation metrics.
"""

import os
import json
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from src.config import (
    FEATURES_DIR,
    MODELS_DIR,
    METRICS_DIR,
    RANDOM_STATE
)
from src.train_xgb_baseline import load_feature_splits, evaluate_predictions, compute_bootstrap_ci


def main():
    print("=" * 60)
    print("STAGE C: Feature Pruning Ablation Study")
    print("=" * 60)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols = load_feature_splits()
    
    prepruned_path = os.path.join(MODELS_DIR, "xgb_prepruned.json")
    metrics_path = os.path.join(METRICS_DIR, "prepruned_metrics.json")
    
    if not os.path.exists(prepruned_path) or not os.path.exists(metrics_path):
        raise FileNotFoundError("Pre-pruned model or metrics not found. Run src.tune_xgb first.")
        
    with open(metrics_path, "r") as f:
        prepruned_info = json.load(f)
        
    best_params = prepruned_info.get("best_params", {})
    best_iteration = prepruned_info.get("best_iteration", 500)
    print(f"Reusing Tuned Pre-Pruning Hyperparameters: {best_params}")
    print(f"Reusing Optimal Early Stopped Iterations: {best_iteration}")
    
    prepruned_xgb = XGBClassifier()
    prepruned_xgb.load_model(prepruned_path)
    
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / max(1, pos_count)
    
    # SHAP Feature Ranking
    print("\nComputing SHAP values for feature ranking...")
    explainer = shap.TreeExplainer(prepruned_xgb)
    
    sample_size = min(2000, len(X_val))
    X_shap_sample = X_val.sample(n=sample_size, random_state=RANDOM_STATE) if len(X_val) > sample_size else X_val
    shap_values = explainer.shap_values(X_shap_sample)
    
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    
    ranked_features = shap_importance["feature"].tolist()
    total_features = len(ranked_features)
    
    subsets = {
        "All": ranked_features,
        "Top 75%": ranked_features[:int(total_features * 0.75)],
        "Top 50%": ranked_features[:int(total_features * 0.50)],
        "Top 25%": ranked_features[:int(total_features * 0.25)]
    }
    
    ablation_results = []
    trained_models = {}
    
    print("\nTraining & Evaluating Models on Feature Subsets (Using Tuned Parameters)...")
    for name, feats in subsets.items():
        print(f"\n--- Model: XGB + {name} ({len(feats)} features) ---")
        
        X_tr_sub = X_train[feats]
        X_val_sub = X_val[feats]
        X_te_sub = X_test[feats]
        
        # Retrain with tuned parameters and best_iteration ceiling
        xgb_sub = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=best_iteration,
            learning_rate=0.03,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **best_params
        )
        xgb_sub.fit(X_tr_sub, y_train, eval_set=[(X_val_sub, y_val)], verbose=False)
        
        val_prob = xgb_sub.predict_proba(X_val_sub)[:, 1]
        test_prob = xgb_sub.predict_proba(X_te_sub)[:, 1]
        
        val_m = evaluate_predictions(y_val, val_prob)
        val_thresh = val_m["EER_Threshold"]
        test_m = evaluate_predictions(y_test, test_prob, threshold=val_thresh)
        test_ci = compute_bootstrap_ci(y_test, test_prob)
        
        model_filename = f"xgb_{name.lower().replace(' ', '').replace('%', '')}.json"
        sub_model_path = os.path.join(MODELS_DIR, model_filename)
        xgb_sub.save_model(sub_model_path)
        trained_models[name] = (xgb_sub, feats, sub_model_path, val_m, test_m, test_ci)
        
        ablation_results.append({
            "Model": f"XGB + {name}",
            "Features": len(feats),
            "Val_Accuracy": val_m["Accuracy"],
            "Val_Macro_F1": val_m["Macro_F1"],
            "Val_ROC-AUC": val_m["ROC-AUC"],
            "Val_EER": val_m["EER"],
            "Test_Accuracy": test_m["Accuracy"],
            "Test_Macro_F1": test_m["Macro_F1"],
            "Test_ROC-AUC": test_m["ROC-AUC"],
            "Test_EER": test_m["EER"],
            "Test_AUC_CI95": f"[{test_ci['ROC-AUC_CI95'][0]:.4f}, {test_ci['ROC-AUC_CI95'][1]:.4f}]",
            "Test_EER_CI95": f"[{test_ci['EER_CI95'][0]:.4f}, {test_ci['EER_CI95'][1]:.4f}]"
        })
        
    ablation_df = pd.DataFrame(ablation_results)
    print("\n--- ABLATION STUDY RESULTS SUMMARY ---")
    print(ablation_df[["Model", "Features", "Val_Macro_F1", "Val_ROC-AUC", "Val_EER", "Test_ROC-AUC", "Test_EER", "Test_AUC_CI95"]].to_string(index=False))
    
    ablation_csv_path = os.path.join(METRICS_DIR, "ablation_table.csv")
    ablation_df.to_csv(ablation_csv_path, index=False)
    print(f"\nAblation table saved to {ablation_csv_path}")
    
    # Select Best Model Based on Validation ROC-AUC / Macro F1
    best_row = ablation_df.sort_values(by=["Val_ROC-AUC", "Val_Macro_F1"], ascending=False).iloc[0]
    best_model_name_clean = best_row["Model"].replace("XGB + ", "")
    winning_model, winning_feats, winning_path, win_val_m, win_test_m, win_ci = trained_models[best_model_name_clean]
    
    print(f"\nWINNING MODEL SELECTED: {best_row['Model']} ({len(winning_feats)} features)")
    print(f"  Validation ROC-AUC: {win_val_m['ROC-AUC']:.4f}, EER: {win_val_m['EER']:.4f}")
    print(f"  Test ROC-AUC      : {win_test_m['ROC-AUC']:.4f}, EER: {win_test_m['EER']:.4f}")
    
    final_model_path = os.path.join(MODELS_DIR, "xgb_best.json")
    winning_model.save_model(final_model_path)
    
    root_model_path = os.path.join(os.path.dirname(MODELS_DIR), "..", "FINAL_xgboost_model.json")
    root_features_path = os.path.join(os.path.dirname(MODELS_DIR), "..", "FINAL_xgboost_features.json")
    winning_model.save_model(os.path.abspath(root_model_path))
    
    with open(os.path.abspath(root_features_path), "w") as f:
        json.dump(winning_feats, f, indent=4)
        
    print(f"Final model saved to {final_model_path} and {root_model_path}")
    print(f"Final feature list ({len(winning_feats)} features) saved to {root_features_path}")


if __name__ == "__main__":
    main()
