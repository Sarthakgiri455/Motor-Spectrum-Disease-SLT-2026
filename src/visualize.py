# src/visualize.py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

with open("features/all_records.pkl", "rb") as f:
    records = pickle.load(f)

# flatten mfcc arrays into individual columns
rows = []
for r in records:
    row = {}
    row["label"]    = r["label"]
    row["language"] = r["language"]
    row["task"]     = r["task"]
    row["speaker_id"] = r["speaker_id"]
    for i, v in enumerate(r["mfcc_mean"]):
        row[f"mfcc_mean_{i}"] = v
    for key in ["f0_mean","f0_std","energy_mean","energy_std",
                "zcr_mean","jitter_local","shimmer_local","hnr",
                "praat_f0_mean","praat_f0_std"]:
        row[key] = r.get(key, 0.0)
    rows.append(row)

df = pd.DataFrame(rows)
print(df.groupby(["language","label"]).size())

import os
os.makedirs("notebooks", exist_ok=True)

# # Plot 1 — clinical features HC vs PD per language
# fig, axes = plt.subplots(2, 3, figsize=(15, 8))
# features  = ["f0_mean", "f0_std", "jitter_local",
#              "shimmer_local", "hnr", "energy_mean"]
# titles    = ["F0 Mean", "F0 Std", "Jitter",
#              "Shimmer", "HNR", "Energy"]

# for ax, feat, title in zip(axes.flat, features, titles):
#     for lang in ["Spanish", "German", "Czech"]:
#         subset = df[df["language"] == lang]
#         hc = subset[subset["label"] == "HC"][feat].dropna()
#         pd_ = subset[subset["label"] == "PD"][feat].dropna()
#         ax.boxplot([hc, pd_],
#                    positions=[0.1, 0.4],
#                    widths=0.2,
#                    patch_artist=True,
#                    boxprops=dict(facecolor="skyblue" if lang=="Spanish"
#                                  else "salmon" if lang=="German" else "lightgreen"),
#                    medianprops=dict(color="black"))
#     ax.set_title(title)
#     ax.set_xticks([0.1, 0.4])
#     ax.set_xticklabels(["HC", "PD"])

# plt.suptitle("Clinical Feature Distributions: HC vs PD", fontsize=14)
# plt.tight_layout()
# plt.savefig("notebooks/clinical_features.png", dpi=150)
# print("Saved clinical_features.png")
# Replace Plot 1 in visualize.py with this
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
features = ["f0_mean","f0_std","jitter_local",
            "shimmer_local","hnr","energy_mean"]
titles   = ["F0 Mean","F0 Std","Jitter",
            "Shimmer","HNR","Energy"]
colors   = {"HC": "steelblue", "PD": "tomato"}
langs    = ["Spanish", "German", "Czech"]

for ax, feat, title in zip(axes.flat, features, titles):
    data_to_plot = []
    tick_labels  = []
    tick_colors  = []
    for lang in langs:
        subset = df[df["language"] == lang]
        for label in ["HC", "PD"]:
            vals = subset[subset["label"]==label][feat].dropna().values
            data_to_plot.append(vals)
            tick_labels.append(f"{lang[:3]}\n{label}")
            tick_colors.append(colors[label])

    bp = ax.boxplot(data_to_plot, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], tick_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(tick_labels)+1))
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_title(title, fontweight="bold")

plt.suptitle("Clinical Features: HC vs PD across Languages", fontsize=14)
plt.tight_layout()
plt.savefig("notebooks/clinical_features.png", dpi=150)
print("Saved clinical_features.png")

# Plot 2 — MFCC means HC vs PD (Spanish vowel only)
fig, ax = plt.subplots(figsize=(12, 5))
spanish_vowel = df[(df["language"]=="Spanish") & (df["task"]=="vowel")]
mfcc_cols = [f"mfcc_mean_{i}" for i in range(5)]
hc_mfcc  = spanish_vowel[spanish_vowel["label"]=="HC"][mfcc_cols].mean()
pd_mfcc  = spanish_vowel[spanish_vowel["label"]=="PD"][mfcc_cols].mean()
x = range(5)
ax.plot(x, hc_mfcc.values, label="HC", color="steelblue", linewidth=2)
ax.plot(x, pd_mfcc.values, label="PD", color="tomato",    linewidth=2)
ax.fill_between(x, hc_mfcc.values, pd_mfcc.values, alpha=0.2, color="gray")
ax.set_xlabel("MFCC Coefficient")
ax.set_ylabel("Mean Value")
ax.set_title("MFCC Profile: HC vs PD (Spanish Vowel)")
ax.legend()
plt.tight_layout()
plt.savefig("notebooks/mfcc_comparison.png", dpi=150)
print("Saved mfcc_comparison.png")

# Plot 3 — HNR distribution across languages
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, lang in zip(axes, ["Spanish", "German", "Czech"]):
    subset = df[df["language"] == lang]
    hc  = subset[subset["label"]=="HC"]["hnr"].dropna()
    pd_ = subset[subset["label"]=="PD"]["hnr"].dropna()
    ax.hist(hc,  bins=30, alpha=0.6, color="steelblue", label="HC")
    ax.hist(pd_, bins=30, alpha=0.6, color="tomato",    label="PD")
    ax.set_title(f"HNR Distribution — {lang}")
    ax.set_xlabel("HNR (dB)")
    ax.legend()
plt.suptitle("Harmonics-to-Noise Ratio: HC vs PD", fontsize=13)
plt.tight_layout()
plt.savefig("notebooks/hnr_distribution.png", dpi=150)
print("Saved hnr_distribution.png")

# Plot 4 — MFCC comparison all three languages

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
langs = ["Spanish", "German", "Czech"]
tasks_per_lang = {"Spanish": "vowel", "German": "vowel", "Czech": "vowel"}

for ax, lang in zip(axes, langs):
    subset = df[df["language"] == lang]
    # filter to vowel task only for fair comparison
    vowel_tasks = [t for t in subset["task"].unique() if "vowel" in t]
    subset = subset[subset["task"].isin(vowel_tasks)]
    
    mfcc_cols = [f"mfcc_mean_{i}" for i in range(5)]
    hc_mfcc  = subset[subset["label"] == "HC"][mfcc_cols].mean()
    pd_mfcc  = subset[subset["label"] == "PD"][mfcc_cols].mean()
    
    x = range(5)
    ax.plot(x, hc_mfcc.values, label="HC", color="steelblue", linewidth=2)
    ax.plot(x, pd_mfcc.values, label="PD", color="tomato",    linewidth=2)
    ax.fill_between(x, hc_mfcc.values, pd_mfcc.values, alpha=0.2, color="gray")
    ax.set_title(f"MFCC Profile — {lang}", fontweight="bold")
    ax.set_xlabel("MFCC Coefficient")
    ax.set_ylabel("Mean Value")
    ax.legend()

plt.suptitle("MFCC HC vs PD across Languages (Vowel Task)", fontsize=14)
plt.tight_layout()
plt.savefig("notebooks/mfcc_all_languages.png", dpi=150)
print("Saved mfcc_all_languages.png")

print("\nDone. Check notebooks/ folder for plots.")