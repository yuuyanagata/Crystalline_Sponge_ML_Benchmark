#!/usr/bin/env python3
"""
Regenerate manuscript Figures 3 (benchmark heatmap) and 4 (RF/Mordred
feature importance) from the CV outputs.

Reads:
  results/cv/model_performance_summary_means.csv
  results/cv/feature_importance_mordred_rf.csv

Writes:
  results/figures/figure3_benchmark.{pdf,png}
  results/figures/figure4_importance.{pdf,png}

Run from the repository root:
  python src/regen_figures.py
"""
import os
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Liberation Sans", "Arimo", "Arial"]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CV   = os.path.join(ROOT, "results", "cv")
FIG  = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

# ---------- Figure 3 ----------
means = pd.read_csv(os.path.join(CV, "model_performance_summary_means.csv"))
descriptors_order = ["Avalon", "MACCS", "Mordred", "Morgan", "RDKit"]
classifiers_order = ["LR", "MLP", "RF", "SVM", "kNN"]
alias = {
    "Logistic Regression": "LR",
    "MLP": "MLP",
    "Random Forest": "RF",
    "SVM": "SVM",
    "kNN": "kNN",
}
means["ModelShort"] = means["Model"].map(alias)

acc = np.zeros((len(classifiers_order), len(descriptors_order)))
f1  = np.zeros_like(acc)
for _, r in means.iterrows():
    i = classifiers_order.index(r["ModelShort"])
    j = descriptors_order.index(r["Descriptor"])
    acc[i, j] = r["Accuracy_mean"]
    f1[i, j]  = r["F1_mean"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
for ax, mat, panel, label in zip(
    axes, [acc, f1], ["(a)", "(b)"],
    ["Mean test accuracy", "Mean test F1-score"],
):
    vmin, vmax = mat.min(), mat.max()
    im = ax.imshow(
        mat, aspect="auto", cmap="viridis",
        vmin=max(0.40, vmin - 0.005), vmax=min(0.75, vmax + 0.005),
    )
    ax.set_xticks(range(len(descriptors_order)))
    ax.set_xticklabels(descriptors_order, fontsize=15, fontweight="bold")
    ax.set_yticks(range(len(classifiers_order)))
    ax.set_yticklabels(classifiers_order, fontsize=15, fontweight="bold")
    ax.set_xlabel("Descriptor", fontsize=16, fontweight="bold")
    ax.set_ylabel("Model", fontsize=16, fontweight="bold")
    for i in range(len(classifiers_order)):
        for j in range(len(descriptors_order)):
            v = mat[i, j]
            color = "white" if v < (vmin + vmax) / 2 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=15, fontweight="bold", color=color)
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(label, fontsize=15, fontweight="bold")
    cb.ax.tick_params(labelsize=13)
    ax.text(-0.55, -0.55, panel, fontsize=18, fontweight="bold",
            transform=ax.transData)

plt.savefig(os.path.join(FIG, "figure3_benchmark.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIG, "figure3_benchmark.png"), bbox_inches="tight", dpi=200)
plt.close()
print("Figure 3 saved.")

# ---------- Figure 4 ----------
imp = pd.read_csv(os.path.join(CV, "feature_importance_mordred_rf.csv"))
top = imp.head(20).iloc[::-1]

fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
ax.barh(range(len(top)), top["Importance"], color="steelblue", edgecolor="black")
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top["Descriptor"], fontsize=11)
ax.set_xlabel("Importance", fontsize=12, fontweight="bold")
ax.set_title(
    "Top Feature Importances (Mordred + Random Forest)",
    fontsize=13, fontweight="bold",
)
ax.tick_params(axis="x", labelsize=11)
ax.grid(True, axis="x", alpha=0.3)
ax.set_axisbelow(True)

plt.savefig(os.path.join(FIG, "figure4_importance.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIG, "figure4_importance.png"), bbox_inches="tight", dpi=200)
plt.close()
print("Figure 4 saved.")
