"""
Interpretability and SHAP Explanation Script.
Generates Top-20 Gain Feature Importance, Global SHAP Beeswarm Plot, Individual Waterfall Explanations for Real & Spoof,
and Mel-Band Frequency-Mapped Heatmap (Hz).
"""

import os
import json
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from xgboost import XGBClassifier

from src.config import (
    FEATURES_DIR,
    MODELS_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    N_MELS,
    FMIN,
    FMAX,
    RANDOM_STATE
)
from src.train_xgb_baseline import load_feature_splits


def main():
    print("=" * 60)
    print("STAGE 5: SHAP Interpretability & Mel-Band Frequency Analysis")
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
    
    # 1. Top-20 Gain Feature Importance Plot
    gain_importance = pd.DataFrame({
        "feature": winning_features,
        "importance": best_xgb.feature_importances_
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    
    top20_gain = gain_importance.head(20)
    plt.figure(figsize=(10, 7))
    plt.barh(top20_gain["feature"][::-1], top20_gain["importance"][::-1], color="#2ca02c")
    plt.xlabel("XGBoost Feature Importance (Gain)")
    plt.ylabel("Feature")
    plt.title("Top 20 XGBoost Features (Gain Importance)")
    plt.tight_layout()
    
    top20_path = os.path.join(PLOTS_DIR, "top20_gain.png")
    plt.savefig(top20_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Top 20 gain plot saved to {top20_path}")
    
    # 2. Global SHAP Summary Beeswarm Plot
    print("\nComputing SHAP values for interpretability plots...")
    explainer = shap.TreeExplainer(best_xgb)
    
    sample_size = min(1000, len(X_val_sub))
    X_shap = X_val_sub.sample(n=sample_size, random_state=RANDOM_STATE) if len(X_val_sub) > sample_size else X_val_sub
    shap_values = explainer(X_shap)
    
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title("Global SHAP Summary Beeswarm Plot")
    plt.tight_layout()
    
    shap_summary_path = os.path.join(PLOTS_DIR, "shap_summary.png")
    plt.savefig(shap_summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Global SHAP summary plot saved to {shap_summary_path}")
    
    # 3. Individual Real (Bonafide) and Spoof Waterfall Explanations
    val_probs = best_xgb.predict_proba(X_val_sub)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    
    # Correct Real sample (y_true=0, y_pred=0)
    real_indices = np.where((y_val.values == 0) & (val_preds == 0))[0]
    # Correct Spoof sample (y_true=1, y_pred=1)
    spoof_indices = np.where((y_val.values == 1) & (val_preds == 1))[0]
    
    if len(real_indices) > 0:
        real_idx = real_indices[0]
        real_shap_single = explainer(X_val_sub.iloc[[real_idx]])[0]
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(real_shap_single, max_display=12, show=False)
        plt.title(f"SHAP Local Explanation - Bonafide (Real) Sample #{real_idx}")
        plt.tight_layout()
        shap_real_path = os.path.join(PLOTS_DIR, "shap_real.png")
        plt.savefig(shap_real_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Local SHAP Real explanation saved to {shap_real_path}")
        
    if len(spoof_indices) > 0:
        spoof_idx = spoof_indices[0]
        spoof_shap_single = explainer(X_val_sub.iloc[[spoof_idx]])[0]
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(spoof_shap_single, max_display=12, show=False)
        plt.title(f"SHAP Local Explanation - Spoof (Deepfake) Sample #{spoof_idx}")
        plt.tight_layout()
        shap_spoof_path = os.path.join(PLOTS_DIR, "shap_spoof.png")
        plt.savefig(shap_spoof_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Local SHAP Spoof explanation saved to {shap_spoof_path}")
        
    # 4. Mel-Band Importance Heatmap Mapped to Frequency Regions (Hz)
    print("\nMapping Mel-band importance back to physical frequency spectrum (Hz)...")
    mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
    shap_df = pd.DataFrame({
        "feature": winning_features,
        "mean_abs_shap": mean_abs_shap
    })
    
    metrics_list = ["mean", "std", "min", "max", "median", "iqr"]
    heatmap_matrix = np.zeros((N_MELS, len(metrics_list)))
    
    for _, row in shap_df.iterrows():
        feat = row["feature"]
        val = row["mean_abs_shap"]
        # Format: mel_083_mean
        parts = feat.split("_")
        if len(parts) >= 3 and parts[0] == "mel":
            try:
                band_idx = int(parts[1])
                metric_name = parts[2]
                if 0 <= band_idx < N_MELS and metric_name in metrics_list:
                    metric_idx = metrics_list.index(metric_name)
                    heatmap_matrix[band_idx, metric_idx] = val
            except ValueError:
                pass
                
    # Map Mel band indices to Frequency centers in Hz
    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    
    # Sample 16 frequency ticks for clean visualization
    tick_indices = np.linspace(0, N_MELS - 1, 16, dtype=int)
    tick_labels = [f"{int(mel_freqs[i])} Hz" for i in tick_indices]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_matrix,
        cmap="YlOrRd",
        xticklabels=metrics_list,
        yticklabels=False,
        cbar_kws={"label": "Mean Absolute SHAP Value"}
    )
    plt.yticks(tick_indices, tick_labels)
    plt.xlabel("Temporal Metric Type across Frames")
    plt.ylabel("Mel Band Center Frequency (Hz)")
    plt.title("Mel-Band Importance Heatmap Mapped to Frequency Regions (Hz)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    heatmap_path = os.path.join(PLOTS_DIR, "mel_band_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Mel-band importance heatmap saved to {heatmap_path}")
    print("\nAll SHAP interpretability visualizations generated successfully.")


if __name__ == "__main__":
    main()
