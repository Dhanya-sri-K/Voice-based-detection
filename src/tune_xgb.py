"""
Stage B: Pre-Pruning / Controlling Tree Growth Hyperparameter Tuning Script.
Uses validation early stopping and Group-Aware CV hyperparameter search to restrict tree depth and complexity.
"""

import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from src.config import (
    FEATURES_DIR,
    MODELS_DIR,
    METRICS_DIR,
    RANDOM_STATE
)
from src.train_xgb_baseline import load_feature_splits, evaluate_predictions, compute_bootstrap_ci


def main():
    print("=" * 60)
    print("STAGE B: Pre-Pruning & Tree Growth Control Tuning")
    print("=" * 60)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols = load_feature_splits()
    
    train_meta_path = os.path.join(FEATURES_DIR, "mel_xgb_train.csv")
    train_meta_df = pd.read_csv(train_meta_path)
    speakers = train_meta_df["speaker_id"].values
    
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / max(1, pos_count)
    
    # Controlled Tree Growth Parameter Space
    param_grid = {
        "max_depth": [3, 4, 5],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.1, 0.5, 1.0],
        "subsample": [0.6, 0.8],
        "colsample_bytree": [0.6, 0.8],
        "reg_alpha": [0.01, 0.1, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0]
    }
    
    print("\nRunning Speaker-Aware GroupKFold Search for Tree Growth Constraints...")
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=150,
        learning_rate=0.03,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    gkf = GroupKFold(n_splits=min(3, len(np.unique(speakers))))
    cv_splits = list(gkf.split(X_train, y_train, groups=speakers))
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=12,
        scoring="roc_auc",
        cv=cv_splits,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)
    
    best_params = search.best_params_
    print(f"\nBest Regularized/Pre-Pruned Parameters: {best_params}")
    
    # Early Stopping on Validation Set to determine optimal boosting rounds
    print("\nTraining Final Pre-Pruned XGBoost with Early Stopping on Validation Set...")
    prepruned_xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=500,
        learning_rate=0.03,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        early_stopping_rounds=30,
        n_jobs=-1,
        **best_params
    )
    
    prepruned_xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    best_iteration = getattr(prepruned_xgb, "best_iteration", prepruned_xgb.n_estimators)
    print(f"Optimal Boosting Iteration (Early Stopped): {best_iteration}")
    
    # Evaluate
    val_prob = prepruned_xgb.predict_proba(X_val)[:, 1]
    test_prob = prepruned_xgb.predict_proba(X_test)[:, 1]
    
    val_metrics = evaluate_predictions(y_val, val_prob)
    val_thresh = val_metrics["EER_Threshold"]
    test_metrics = evaluate_predictions(y_test, test_prob, threshold=val_thresh)
    test_ci = compute_bootstrap_ci(y_test, test_prob)
    
    print("\n--- PRE-PRUNED RESULTS ---")
    print("Validation Metrics:")
    for k, v in val_metrics.items():
        print(f"  {k:20s}: {v:.4f}")
        
    print("\nTest Metrics (Validation-Tuned Threshold):")
    for k, v in test_metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print(f"  Test ROC-AUC 95% CI : {test_ci['ROC-AUC_CI95']}")
    print(f"  Test EER 95% CI     : {test_ci['EER_CI95']}")
        
    model_path = os.path.join(MODELS_DIR, "xgb_prepruned.json")
    prepruned_xgb.save_model(model_path)
    print(f"\nPre-pruned model saved to {model_path}")
    
    metrics_path = os.path.join(METRICS_DIR, "prepruned_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "best_params": best_params,
            "best_iteration": int(best_iteration),
            "val": val_metrics,
            "test": test_metrics,
            "test_ci95": test_ci
        }, f, indent=4)
    print(f"Pre-pruned metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
