# src/feature_extraction.py
import os
import librosa
import numpy as np
import pandas as pd
import pickle
import parselmouth
from parselmouth.praat import call

DATA_ROOT = "data"
LANGUAGES = ["Spanish", "German", "Czech"]

def load_audio(path, sr=16000):
    y, _ = librosa.load(path, sr=sr)
    return y, sr

def extract_praat_features(path):
    """Clinical voice features via Praat — jitter, shimmer, HNR"""
    try:
        sound = parselmouth.Sound(path)
        pitch = call(sound, "To Pitch", 0.0, 50, 500)
        point_process = call([sound, pitch],
                             "To PointProcess (cc)")

        jitter_local = call(point_process,
                            "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_rap   = call(point_process,
                            "Get jitter (rap)",   0, 0, 0.0001, 0.02, 1.3)
        shimmer_local= call([sound, point_process],
                            "Get shimmer (local)",0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_db   = call([sound, point_process],
                            "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        harmonicity  = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr          = call(harmonicity, "Get mean", 0, 0)

        f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
        f0_std  = call(pitch, "Get standard deviation", 0, 0, "Hertz")

        return {
            "jitter_local":   jitter_local  or 0.0,
            "jitter_rap":     jitter_rap    or 0.0,
            "shimmer_local":  shimmer_local or 0.0,
            "shimmer_db":     shimmer_db    or 0.0,
            "hnr":            hnr           or 0.0,
            "praat_f0_mean":  f0_mean       or 0.0,
            "praat_f0_std":   f0_std        or 0.0,
        }
    except Exception as e:
        print(f"    Praat error: {e}")
        return {k: 0.0 for k in [
            "jitter_local","jitter_rap","shimmer_local",
            "shimmer_db","hnr","praat_f0_mean","praat_f0_std"
        ]}

def extract_librosa_features(y, sr):
    """Spectral and prosodic features via librosa"""
    mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
    delta_mfcc  = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    energy      = librosa.feature.rms(y=y)[0]
    zcr         = librosa.feature.zero_crossing_rate(y)[0]
    spectral_centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    f0 = librosa.yin(y, fmin=50, fmax=500)
    f0_voiced = f0[f0 > 0]

    return {
        "mfcc_mean":   mfcc.mean(axis=1),
        "mfcc_std":    mfcc.std(axis=1),
        "delta_mfcc_mean":  delta_mfcc.mean(axis=1),
        "delta2_mfcc_mean": delta2_mfcc.mean(axis=1),
        "energy_mean": float(energy.mean()),
        "energy_std":  float(energy.std()),
        "zcr_mean":    float(zcr.mean()),
        "zcr_std":     float(zcr.std()),
        "spectral_centroid_mean":  float(spectral_centroid.mean()),
        "spectral_bandwidth_mean": float(spectral_bandwidth.mean()),
        "spectral_rolloff_mean":   float(spectral_rolloff.mean()),
        "f0_mean": float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0,
        "f0_std":  float(np.std(f0_voiced))  if len(f0_voiced) > 0 else 0.0,
    }

def scan_tasks(language_path):
    tasks = []
    for root, dirs, _ in os.walk(language_path):
        if "HC" in dirs or "PD" in dirs:
            task_name = os.path.relpath(root, language_path)
            tasks.append((root, task_name))
    return tasks

def build_dataset(language):
    records = []
    language_path = os.path.join(DATA_ROOT, language)
    if not os.path.exists(language_path):
        print(f"  Language folder not found: {language_path}")
        return records

    tasks = scan_tasks(language_path)
    for task_path, task_name in tasks:
        for label in ["HC", "PD"]:
            folder = os.path.join(task_path, label)
            if not os.path.exists(folder):
                continue
            files = [f for f in os.listdir(folder) if f.endswith(".wav")]
            print(f"  {language}/{task_name}/{label}: {len(files)} files")
            for fname in files:
                path = os.path.join(folder, fname)
                try:
                    y, sr    = load_audio(path)
                    lib_feats = extract_librosa_features(y, sr)
                    praat_feats = extract_praat_features(path)
                    record = {**lib_feats, **praat_feats}
                    record["speaker_id"] = fname.replace(".wav", "")
                    record["label"]      = label
                    record["task"]       = task_name
                    record["language"]   = language
                    records.append(record)
                except Exception as e:
                    print(f"    ERROR {fname}: {e}")
    return records

if __name__ == "__main__":
    # install parselmouth if needed
    # pip install praat-parselmouth

    all_records = []
    for language in LANGUAGES:
        print(f"\n{'='*40}")
        print(f"Processing {language}...")
        records = build_dataset(language)
        all_records.extend(records)

    print(f"\n{'='*40}")
    print(f"Total files processed: {len(all_records)}")
    print(f"Languages: {set(r['language'] for r in all_records)}")
    print(f"Tasks:     {set(r['task']     for r in all_records)}")
    print(f"Labels:    {set(r['label']    for r in all_records)}")

    os.makedirs("features", exist_ok=True)
    with open("features/all_records.pkl", "wb") as f:
        pickle.dump(all_records, f)
    print("\nSaved to features/all_records.pkl")