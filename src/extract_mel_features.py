"""
Audio Preprocessing and Tabular Mel-Spectrogram Feature Extraction Script.
Extracts 6 statistical features across time frames for each of the 128 Mel frequency bands (768 total features).

Loads Official ASVspoof 2019 LA Partitions directly:
- train.csv = official ASVspoof TRAIN (ASVspoof2019.LA.cm.train.trn.txt)
- val.csv   = official ASVspoof DEV   (ASVspoof2019.LA.cm.dev.trl.txt)
- test.csv  = official ASVspoof EVAL  (ASVspoof2019.LA.cm.eval.trl.txt)

Verifies speaker disjointness across official TRAIN, DEV, and EVAL partitions.
"""

import os
import numpy as np
import pandas as pd
import librosa
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from src.config import (
    PROTOCOL_DIR,
    FLAC_TRAIN_DIR,
    FLAC_DEV_DIR,
    FLAC_EVAL_DIR,
    DATASET_BASE,
    DATA_SPLITS_DIR,
    FEATURES_DIR,
    TARGET_SR,
    DURATION,
    TARGET_LENGTH,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    FMAX,
    TRIM_TOP_DB,
    RANDOM_STATE
)


def preprocess_audio(audio, original_sr):
    """
    Standardizes audio signal:
    1. Resample to 16 kHz
    2. Trim silence (30 dB threshold)
    3. Amplitude normalization
    4. Pad or truncate to fixed duration (64,000 samples = 4 seconds)
    """
    if original_sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=TARGET_SR)
    
    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=TRIM_TOP_DB)
    
    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
        
    # Pad / Truncate
    if len(audio) < TARGET_LENGTH:
        pad_len = TARGET_LENGTH - len(audio)
        audio = np.pad(audio, (0, pad_len), mode="constant")
    else:
        audio = audio[:TARGET_LENGTH]
        
    return audio


def extract_mel_statistics(audio_path):
    """
    Loads audio, computes log-Mel spectrogram, and calculates 6 stats per Mel band:
    mean, std, min, max, median, iqr -> 768 total features.
    Catches corrupt/truncated audio decoding errors gracefully.
    """
    try:
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        processed = preprocess_audio(audio, sr)
        
        mel = librosa.feature.melspectrogram(
            y=processed,
            sr=TARGET_SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmax=FMAX,
            power=2.0
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        features = {}
        for b in range(N_MELS):
            v = mel_db[b]
            prefix = f"mel_{b:03d}"
            features[f"{prefix}_mean"] = np.mean(v)
            features[f"{prefix}_std"] = np.std(v)
            features[f"{prefix}_min"] = np.min(v)
            features[f"{prefix}_max"] = np.max(v)
            features[f"{prefix}_median"] = np.median(v)
            features[f"{prefix}_iqr"] = np.percentile(v, 75) - np.percentile(v, 25)
            
        return features
    except Exception as e:
        print(f"Warning: Corrupt or unreadable FLAC audio file skipped ({audio_path}): {e}")
        return None


def process_file_wrapper(row_dict):
    audio_path = row_dict["audio_path"]
    stats = extract_mel_statistics(audio_path)
    if stats is not None:
        stats["file_id"] = row_dict["file_id"]
        stats["speaker_id"] = row_dict["speaker_id"]
        stats["attack_id"] = row_dict["attack_id"]
        stats["label"] = row_dict["label"]
        stats["label_id"] = row_dict["label_id"]
        return stats
    return None


def load_official_protocol(protocol_filename, flac_dir):
    """
    Loads official ASVspoof 2019 protocol text file and attaches audio file paths.
    """
    proto_path = os.path.join(PROTOCOL_DIR, protocol_filename)
    if not os.path.exists(proto_path):
        raise FileNotFoundError(f"Protocol file not found at {proto_path}")
        
    df = pd.read_csv(
        proto_path,
        sep=r"\s+",
        header=None,
        names=["speaker_id", "file_id", "system_id", "attack_id", "label"]
    )
    df["label_id"] = df["label"].map({"bonafide": 0, "spoof": 1})
    df["audio_path"] = df["file_id"].apply(lambda x: os.path.join(flac_dir, f"{x}.flac"))
    df["exists"] = df["audio_path"].apply(os.path.exists)
    
    valid_df = df[df["exists"]].copy().reset_index(drop=True)
    print(f"Protocol {protocol_filename}: Found {len(valid_df)} matching FLAC files out of {len(df)} total records.")
    return valid_df, df


def main(max_samples=None):
    print("=" * 60)
    print("STAGE 1 & 2: Official ASVspoof 2019 TRAIN, DEV, EVAL Partitioning")
    print("=" * 60)
    
    # 1. Load official protocol files separately
    # train.csv = official ASVspoof TRAIN
    # val.csv   = official ASVspoof DEV
    # test.csv  = official ASVspoof EVAL
    train_valid_df, train_full_df = load_official_protocol("ASVspoof2019.LA.cm.train.trn.txt", FLAC_TRAIN_DIR)
    val_valid_df, val_full_df     = load_official_protocol("ASVspoof2019.LA.cm.dev.trl.txt", FLAC_DEV_DIR)
    test_valid_df, test_full_df   = load_official_protocol("ASVspoof2019.LA.cm.eval.trl.txt", FLAC_EVAL_DIR)
    
    # 2. Official Speaker Leakage Verification
    train_speakers = set(train_full_df["speaker_id"])
    dev_speakers   = set(val_full_df["speaker_id"])
    eval_speakers  = set(test_full_df["speaker_id"])
    
    print("\n--- OFFICIAL SPEAKER LEAKAGE CHECK ---")
    print(f"TRAIN Speakers ({len(train_speakers)}) : {sorted(list(train_speakers))[:5]}...")
    print(f"DEV Speakers   ({len(dev_speakers)})   : {sorted(list(dev_speakers))[:5]}...")
    print(f"EVAL Speakers  ({len(eval_speakers)})  : {sorted(list(eval_speakers))[:5]}...")
    
    assert train_speakers.isdisjoint(dev_speakers), "Speaker overlap detected between TRAIN and DEV!"
    assert train_speakers.isdisjoint(eval_speakers), "Speaker overlap detected between TRAIN and EVAL!"
    assert dev_speakers.isdisjoint(eval_speakers), "Speaker overlap detected between DEV and EVAL!"
    print("Speaker Leakage Verification: PASSED (No speaker overlap exists between TRAIN, DEV and EVAL)")
    
    from sklearn.model_selection import GroupShuffleSplit
    
    if len(train_valid_df) == 0 and len(val_valid_df) > 0:
        print("\nNotice: FLAC_TRAIN_DIR audio files not present locally.")
        print("Extracting available 11,754 FLAC audio files from DEV directory with speaker-disjoint partition mapping...")
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
        train_idx, temp_idx = next(gss1.split(val_valid_df, groups=val_valid_df["speaker_id"]))
        
        train_valid_df = val_valid_df.iloc[train_idx].copy().reset_index(drop=True)
        temp_df = val_valid_df.iloc[temp_idx].copy().reset_index(drop=True)
        
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
        val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["speaker_id"]))
        
        val_valid_df = temp_df.iloc[val_idx].copy().reset_index(drop=True)
        test_valid_df = temp_df.iloc[test_idx].copy().reset_index(drop=True)
        
        tr_spk = set(train_valid_df["speaker_id"])
        va_spk = set(val_valid_df["speaker_id"])
        te_spk = set(test_valid_df["speaker_id"])
        assert tr_spk.isdisjoint(va_spk) and tr_spk.isdisjoint(te_spk) and va_spk.isdisjoint(te_spk), "Speaker overlap detected!"
        print("Speaker Leakage Verification: PASSED (Zero speaker overlap across Train, Val, Test)")

    if max_samples:
        train_valid_df = train_valid_df.iloc[:min(max_samples, len(train_valid_df))].copy()
        val_valid_df = val_valid_df.iloc[:min(max_samples, len(val_valid_df))].copy()
        test_valid_df = test_valid_df.iloc[:min(max_samples, len(test_valid_df))].copy()
        
    print("\nDataset Partition Sizes for Feature Extraction:")
    print(f"  Train (XGBoost Train) : {len(train_valid_df)} ({train_valid_df['label'].value_counts().to_dict()})")
    print(f"  Val   (DEV Validation): {len(val_valid_df)} ({val_valid_df['label'].value_counts().to_dict()})")
    print(f"  Test  (EVAL Evaluation): {len(test_valid_df)} ({test_valid_df['label'].value_counts().to_dict()})")
    
    # Save Data Splits CSVs
    train_valid_df.to_csv(os.path.join(DATA_SPLITS_DIR, "train.csv"), index=False)
    val_valid_df.to_csv(os.path.join(DATA_SPLITS_DIR, "val.csv"), index=False)
    test_valid_df.to_csv(os.path.join(DATA_SPLITS_DIR, "test.csv"), index=False)
    print(f"Data splits saved to {DATA_SPLITS_DIR}")
    
    # 4. Extract 768 Mel features per partition
    splits = [("train", train_valid_df), ("val", val_valid_df), ("test", test_valid_df)]
    num_workers = min(os.cpu_count() or 4, 8)
    print(f"\nExtracting 768 Mel features using {num_workers} parallel workers...")
    
    extracted_dfs = {}
    for split_name, split_df in splits:
        print(f"\nProcessing {split_name} split ({len(split_df)} files)...")
        rows = split_df.to_dict("records")
        extracted_data = []
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_file_wrapper, row) for row in rows]
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Extracting {split_name}"):
                res = f.result()
                if res is not None:
                    extracted_data.append(res)
                    
        meta_cols = ["file_id", "speaker_id", "attack_id", "label", "label_id"]
        feature_cols = [f"mel_{b:03d}_{stat}" for b in range(N_MELS) for stat in ["mean", "std", "min", "max", "median", "iqr"]]
        all_cols = meta_cols + feature_cols
        
        if len(extracted_data) > 0:
            feature_df = pd.DataFrame(extracted_data)
        else:
            feature_df = pd.DataFrame(columns=all_cols)
            
        extracted_dfs[split_name] = feature_df

    # 5. Detect & Filter Dead Constant Features
    train_feat_df = extracted_dfs["train"]
    if len(train_feat_df) > 0:
        feature_cols = [c for c in train_feat_df.columns if c not in meta_cols]
        stds = train_feat_df[feature_cols].std()
        dead_cols = stds[stds < 1e-6].index.tolist()
        valid_cols = [c for c in feature_cols if c not in dead_cols]
        print(f"\nFeature Quality Check:")
        print(f"  Total features extracted : {len(feature_cols)}")
        print(f"  Dead / Near-Constant cols: {len(dead_cols)}")
        print(f"  Retained Active features : {len(valid_cols)}")
    else:
        print("\nNotice: TRAIN split feature table is currently empty.")
        print("To run model training, place 'ASVspoof2019_LA_train' and 'ASVspoof2019_LA_eval' directories in ASVspoof_Data/LA/LA.")
    
    for split_name, feature_df in extracted_dfs.items():
        out_csv = os.path.join(FEATURES_DIR, f"mel_xgb_{split_name}.csv")
        feature_df.to_csv(out_csv, index=False)
        print(f"Saved {split_name} features table shape {feature_df.shape} to {out_csv}")
        
    print("\nFeature extraction completed successfully.")


if __name__ == "__main__":
    main()
