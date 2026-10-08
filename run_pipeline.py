"""
End-to-End Orchestrator Script for XGBoost Mel-Spectrogram Deepfake Speech Detection Pipeline.

Executes all 5 stages in sequence:
1. Feature Extraction & Data Splitting (Train, Validation, Test)
2. Stage A: Baseline XGBoost Model Training & Metrics
3. Stage B: Pre-Pruning & Growth Control Hyperparameter Search
4. Stage C: Feature Pruning Ablation Study & Winning Model Selection
5. Evaluation & SHAP Interpretability Plots
"""

import sys
import os

from src.extract_mel_features import main as extract_main
from src.train_xgb_baseline import main as baseline_main
from src.tune_xgb import main as tune_main
from src.prune_features import main as prune_main
from src.evaluate_xgb import main as eval_main
from src.explain_shap import main as explain_main


def run_all(max_samples=None):
    print("=" * 70)
    print("  XGBOOST MEL-SPECTROGRAM DEEPFAKE DETECTION PIPELINE  ")
    print("=" * 70)
    
    print("\n>>> STEP 1: Preprocessing & Mel Feature Extraction")
    extract_main(max_samples=max_samples)
    
    print("\n>>> STEP 2: Training Baseline XGBoost Model")
    baseline_main()
    
    print("\n>>> STEP 3: Pre-Pruning & Tree Growth Control Tuning")
    tune_main()
    
    print("\n>>> STEP 4: Feature Pruning Ablation Study")
    prune_main()
    
    print("\n>>> STEP 5: Final Evaluation & Comparison Plots")
    eval_main()
    
    print("\n>>> STEP 6: Generating SHAP Interpretability Plots")
    explain_main()
    
    print("\n" + "=" * 70)
    print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!  ")
    print("=" * 70)


if __name__ == "__main__":
    max_s = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_all(max_samples=max_s)
