#!/usr/bin/env python3
"""
Nested 5x5 stratified CV for all 25 (classifier, descriptor) combinations,
plus final retraining of every combination on the full dataset.

Pipeline = SimpleImputer(mean) + StandardScaler + classifier, wrapped in
sklearn.pipeline.Pipeline. Hyperparameters are tuned inside each outer
fold using a 5-fold GridSearchCV with the `clf__` prefix.

Reads:
  results/descriptors/*.npy
  results/descriptors/labels.npy
  results/descriptors/ids.npy

Writes (results/cv/):
  model_performance_summary.csv         per-fold metrics
  model_performance_summary_means.csv   per-(model,descriptor) mean +/- std
  hyperparameter_summary.csv            representative best params
  confusion_matrices.csv                summed over outer folds
  predictions.csv                       per-entry out-of-fold predictions
  feature_importance_mordred_rf.csv     RF/Mordred feature importance
                                        (model retrained on full dataset)

Writes (models/):
  <model>_<descriptor>_full.joblib      every (model, descriptor) combination,
                                        retrained on the full dataset with
                                        the representative best params

Run from the repository root:
  python src/run_cv.py

Total runtime is around 10 minutes on a single core.
"""
import os
import json
import time
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESC = os.path.join(ROOT, "results", "descriptors")
OUT  = os.path.join(ROOT, "results", "cv")
MOD  = os.path.join(ROOT, "models")
os.makedirs(OUT, exist_ok=True)
os.makedirs(MOD, exist_ok=True)

y   = np.load(os.path.join(DESC, "labels.npy"))
ids = np.load(os.path.join(DESC, "ids.npy"))
print(f"N={len(y)}  class counts: {Counter(y.tolist())}")

descriptors_to_load = {
    "RDKit":   os.path.join(DESC, "rdkit_descriptors.npy"),
    "Mordred": os.path.join(DESC, "mordred_descriptors.npy"),
    "Morgan":  os.path.join(DESC, "morgan_fingerprints.npy"),
    "Avalon":  os.path.join(DESC, "avalon_fingerprints.npy"),
    "MACCS":   os.path.join(DESC, "maccs_keys.npy"),
}
X_all = {n: np.load(p).astype(np.float64) for n, p in descriptors_to_load.items()}
for n, X in X_all.items():
    print(f"  {n:8s}: shape {X.shape}")

# Drop all-NaN columns for descriptor sets that have them (Mordred mostly)
for name in ["Mordred", "RDKit"]:
    before = X_all[name].shape[1]
    keep = ~np.all(np.isnan(X_all[name]), axis=0)
    X_all[name] = X_all[name][:, keep]
    print(f"  {name}: dropped all-NaN columns: {before} -> {X_all[name].shape[1]}")

# ---------------------------------------------------------------------------
# Classifier configurations: (factory, grid)
# ---------------------------------------------------------------------------
def cfg_lr():
    return (
        LogisticRegression(max_iter=2000, random_state=0),
        {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    )
def cfg_svm():
    return (
        SVC(kernel="rbf", random_state=0),
        {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", "auto"]},
    )
def cfg_rf():
    return (
        RandomForestClassifier(random_state=0, n_jobs=1),
        {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 5, 10]},
    )
def cfg_knn():
    return (
        KNeighborsClassifier(),
        {"clf__n_neighbors": [3, 5, 7]},
    )
def cfg_mlp():
    return (
        MLPClassifier(max_iter=1000, random_state=0),
        {
            "clf__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
            "clf__alpha": [1e-4, 1e-3],
        },
    )

CLASSIFIERS = {
    "Logistic Regression": cfg_lr,
    "SVM":                 cfg_svm,
    "Random Forest":       cfg_rf,
    "kNN":                 cfg_knn,
    "MLP":                 cfg_mlp,
}

OUTER = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
INNER = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)

rows_fold, conf_rows, hyper_rows, pred_rows = [], [], [], []

t_all = time.time()
for desc_name, X in X_all.items():
    for clf_name, factory in CLASSIFIERS.items():
        tstart = time.time()
        per_fold_acc, per_fold_prec, per_fold_rec, per_fold_f1 = [], [], [], []
        cm_total = np.zeros((3, 3), dtype=int)
        best_params_per_fold = []
        for fold_i, (train_idx, test_idx) in enumerate(OUTER.split(X, y), 1):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            estimator, grid = factory()
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler",  StandardScaler()),
                ("clf",     estimator),
            ])
            gs = GridSearchCV(
                pipe, grid, cv=INNER, scoring="accuracy",
                n_jobs=-1, refit=True, error_score=np.nan,
            )
            gs.fit(X_tr, y_tr)
            best = gs.best_estimator_
            best_params_per_fold.append(gs.best_params_)
            y_pred = best.predict(X_te)
            acc = accuracy_score(y_te, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_te, y_pred, average="weighted", zero_division=0,
            )
            cm_total += confusion_matrix(y_te, y_pred, labels=[0, 1, 2])
            per_fold_acc.append(acc); per_fold_prec.append(prec)
            per_fold_rec.append(rec); per_fold_f1.append(f1)
            rows_fold.append({
                "Model": clf_name, "Descriptor": desc_name, "Fold": fold_i,
                "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
                "Params": json.dumps(gs.best_params_),
            })
            for k, idx_global in enumerate(test_idx):
                pred_rows.append({
                    "ID": int(ids[idx_global]),
                    "True_Class": int(y[idx_global]),
                    "Predicted_Class": int(y_pred[k]),
                    "Fold": fold_i, "Model": clf_name, "Descriptor": desc_name,
                })
        # Representative best params = mode across outer folds
        common_str = Counter(
            [json.dumps(p, sort_keys=True) for p in best_params_per_fold]
        ).most_common(1)[0][0]
        common = json.loads(common_str)
        hyper_rows.append({
            "Model": clf_name, "Descriptor": desc_name,
            "Representative_Best_Params": common_str,
        })
        for true_cls in range(3):
            conf_rows.append({
                "Model": clf_name, "Descriptor": desc_name,
                "True_class": true_cls,
                "Pred 0": int(cm_total[true_cls, 0]),
                "Pred 1": int(cm_total[true_cls, 1]),
                "Pred 2": int(cm_total[true_cls, 2]),
            })
        # Retrain on full dataset with representative hyperparameters and save
        est_factory, _ = factory()
        final_est, _ = factory()
        final_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler",  StandardScaler()),
            ("clf",     final_est),
        ])
        final_pipe.set_params(**common)
        final_pipe.fit(X, y)
        clf_slug = clf_name.lower().replace(" ", "_")
        joblib.dump(
            final_pipe,
            os.path.join(MOD, f"{clf_slug}_{desc_name.lower()}_full.joblib"),
        )
        accs = np.array(per_fold_acc); f1s = np.array(per_fold_f1)
        print(
            f"  {clf_name:20s} {desc_name:8s}  "
            f"acc={accs.mean():.3f}+/-{accs.std():.3f}  "
            f"f1={f1s.mean():.3f}+/-{f1s.std():.3f}  ({time.time()-tstart:.1f}s)"
        )

print(f"\nTotal CV time: {time.time()-t_all:.1f}s")

# Save outputs
fold_df = pd.DataFrame(rows_fold)
fold_df.to_csv(os.path.join(OUT, "model_performance_summary.csv"), index=False)

mean_rows = []
for (m, d), grp in fold_df.groupby(["Model", "Descriptor"]):
    mean_rows.append({
        "Model": m, "Descriptor": d,
        "Accuracy_mean":  grp["Accuracy"].mean(),
        "Accuracy_std":   grp["Accuracy"].std(),
        "Precision_mean": grp["Precision"].mean(),
        "Precision_std":  grp["Precision"].std(),
        "Recall_mean":    grp["Recall"].mean(),
        "Recall_std":     grp["Recall"].std(),
        "F1_mean":        grp["F1"].mean(),
        "F1_std":         grp["F1"].std(),
    })
mean_df = pd.DataFrame(mean_rows)
mean_df.to_csv(os.path.join(OUT, "model_performance_summary_means.csv"), index=False)

pd.DataFrame(conf_rows ).to_csv(os.path.join(OUT, "confusion_matrices.csv"),   index=False)
pd.DataFrame(hyper_rows).to_csv(os.path.join(OUT, "hyperparameter_summary.csv"), index=False)
pd.DataFrame(pred_rows ).to_csv(os.path.join(OUT, "predictions.csv"),          index=False)

# RF / Mordred feature importance from the full-data refit model
print("\nExtracting RF/Mordred feature importance ...")
rfm_path = os.path.join(MOD, "random_forest_mordred_full.joblib")
rfm = joblib.load(rfm_path)
imp = rfm.named_steps["clf"].feature_importances_
with open(os.path.join(DESC, "mordred_names.txt")) as f:
    all_names = [l.strip() for l in f.readlines()]
X_m = np.load(os.path.join(DESC, "mordred_descriptors.npy")).astype(np.float64)
keep = ~np.all(np.isnan(X_m), axis=0)
names = [n for n, k in zip(all_names, keep) if k]
imp_df = pd.DataFrame({"Descriptor": names, "Importance": imp}).sort_values(
    "Importance", ascending=False
).reset_index(drop=True)
imp_df.to_csv(os.path.join(OUT, "feature_importance_mordred_rf.csv"), index=False)
print(imp_df.head(20).to_string(index=False))
print("\nDone.")
