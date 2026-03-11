# -*- coding: utf-8 -*-
import os
import json
import csv
import pickle
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, AllChem
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from mordred import Calculator, descriptors

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# =============================
# Utility
# =============================
def safe_json_dumps(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def ensure_clean_file(path):
    if os.path.exists(path):
        os.remove(path)


def canonical_descriptor_name(descriptor_name_or_file):
    """
    Convert descriptor csv filename to clean descriptor name if needed.
    """
    mapping = {
        "rdkit_descriptors.csv": "RDKit",
        "mordred_descriptors.csv": "Mordred",
        "morgan_descriptors.csv": "Morgan",
        "avalon_descriptors.csv": "Avalon",
        "maccs_descriptors.csv": "MACCS",
        "RDKit": "RDKit",
        "Mordred": "Mordred",
        "Morgan": "Morgan",
        "Avalon": "Avalon",
        "MACCS": "MACCS",
    }
    return mapping.get(descriptor_name_or_file, descriptor_name_or_file)


def canonical_model_name(model_name):
    mapping = {
        "LogisticRegression": "Logistic Regression",
        "SVM": "SVM",
        "RandomForest": "Random Forest",
        "kNN": "kNN",
        "MLP": "MLP",
    }
    return mapping.get(model_name, model_name)


def representative_params(series):
    """
    Choose the most frequent hyperparameter setting string across folds.
    If tie, choose lexicographically smallest for stability.
    """
    counts = Counter(series.dropna().astype(str).tolist())
    if not counts:
        return ""
    max_count = max(counts.values())
    candidates = sorted([k for k, v in counts.items() if v == max_count])
    return candidates[0]


# =============================
# Descriptor calculation
# =============================
def calculate_and_save_descriptors(smiles_list, output_file, descriptor_type):
    descriptor_data = []

    if descriptor_type == "RDKit":
        descriptor_names = [desc[0] for desc in Descriptors.descList]
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                values = [Descriptors.__dict__[name](mol) for name in descriptor_names]
            else:
                values = [np.nan] * len(descriptor_names)
            descriptor_data.append(values)
        header = descriptor_names

    elif descriptor_type == "Mordred":
        calc = Calculator(descriptors, ignore_3D=True)
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                try:
                    values = list(calc(mol).asdict().values())
                except Exception as e:
                    print(f"Error calculating Mordred descriptors for SMILES {smiles}: {e}")
                    values = [np.nan] * len(calc.descriptors)
            else:
                values = [np.nan] * len(calc.descriptors)
            descriptor_data.append(values)
        header = [str(d) for d in calc.descriptors]

    elif descriptor_type == "Morgan":
        generator = AllChem.GetMorganGenerator(radius=2, fpSize=512)
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                values = list(generator.GetFingerprint(mol))
            else:
                values = [np.nan] * 512
            descriptor_data.append(values)
        header = [f"Morgan_{i}" for i in range(512)]

    elif descriptor_type == "Avalon":
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                values = list(GetAvalonFP(mol, nBits=512))
            else:
                values = [np.nan] * 512
            descriptor_data.append(values)
        header = [f"Avalon_{i}" for i in range(512)]

    elif descriptor_type == "MACCS":
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                values = list(MACCSkeys.GenMACCSKeys(mol))
            else:
                values = [np.nan] * 167
            descriptor_data.append(values)
        header = [f"MACCS_{i}" for i in range(167)]

    else:
        raise ValueError(f"Unsupported descriptor type: {descriptor_type}")

    df = pd.DataFrame(descriptor_data, columns=header)
    df.to_csv(output_file, index=False)


# =============================
# Load input data
# =============================
INPUT_CSV = "CS_Booklet-2.csv"

try:
    smiles_data = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"Error: File '{INPUT_CSV}' not found.")
    raise SystemExit(1)

required_columns = ["SMILES", "ExperimentResult"]
for col in required_columns:
    if col not in smiles_data.columns:
        print(f"Error: '{col}' column not found in '{INPUT_CSV}'.")
        raise SystemExit(1)

smiles_list = smiles_data["SMILES"].astype(str)
y_all = smiles_data["ExperimentResult"]

if "ID" in smiles_data.columns:
    sample_ids = smiles_data["ID"]
else:
    sample_ids = pd.Series(np.arange(1, len(smiles_data) + 1), name="ID")

# Optional reference column for downstream traceability
reference_col = smiles_data["Reference(s)"] if "Reference(s)" in smiles_data.columns else pd.Series([""] * len(smiles_data))


# =============================
# Save descriptors
# =============================
descriptor_files = {
    "RDKit": "rdkit_descriptors.csv",
    "Mordred": "mordred_descriptors.csv",
    "Morgan": "morgan_descriptors.csv",
    "Avalon": "avalon_descriptors.csv",
    "MACCS": "maccs_descriptors.csv",
}

for desc_type, output_file in descriptor_files.items():
    print(f"Calculating {desc_type} descriptors...")
    calculate_and_save_descriptors(smiles_list, output_file, desc_type)


# =============================
# Output files
# =============================
summary_file = "model_performance_summary.csv"
means_file = "model_performance_summary_means.csv"
hyperparam_file = "hyperparameter_summary.csv"
predictions_file = "predictions_detailed.csv"
confmat_file = "confusion_matrices_summary.csv"
best_model_file = "best_model_summary.txt"

for f in [summary_file, means_file, hyperparam_file, predictions_file, confmat_file, best_model_file]:
    ensure_clean_file(f)


# =============================
# Model training / Nested CV
# =============================
def train_and_evaluate_models(descriptor_name, descriptor_file):
    data = pd.read_csv(descriptor_file)
    X_raw = data.select_dtypes(include=[np.number])

    # preprocessing
    X_imputed = SimpleImputer(strategy="mean").fit_transform(X_raw)
    X = StandardScaler().fit_transform(X_imputed)
    y = y_all.reset_index(drop=True)

    models_and_grids = {
        "LogisticRegression": (
            LogisticRegression(max_iter=3000, solver="lbfgs", random_state=42),
            {"C": [0.01, 0.1, 1, 10]},
        ),
        "SVM": (
            SVC(probability=False),
            {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
        ),
        "kNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7]},
        ),
        "MLP": (
            MLPClassifier(max_iter=3000, random_state=42),
            {"hidden_layer_sizes": [(50,), (100,), (100, 50)], "alpha": [0.0001, 0.001]},
        ),
    }

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    summary_rows = []
    prediction_rows = []
    confmat_rows = []

    for raw_model_name, (base_model, param_grid) in models_and_grids.items():
        model_name = canonical_model_name(raw_model_name)
        print(f"\n=== {model_name} on {descriptor_name} ===")

        fold_num = 1

        for train_idx, test_idx in outer_cv.split(X, y):
            X_trainval, X_test = X[train_idx], X[test_idx]
            y_trainval, y_test = y.iloc[train_idx], y.iloc[test_idx]

            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid = GridSearchCV(
                estimator=clone(base_model),
                param_grid=param_grid,
                cv=inner_cv,
                scoring="accuracy",
                n_jobs=-1,
            )
            grid.fit(X_trainval, y_trainval)
            best_model = grid.best_estimator_

            y_pred = best_model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            # save model
            model_file = f"{raw_model_name}_{descriptor_name}_fold{fold_num}.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(best_model, f)

            best_params_str = safe_json_dumps(grid.best_params_)

            summary_rows.append({
                "Model": model_name,
                "Descriptor": descriptor_name,
                "Fold": fold_num,
                "Test_Accuracy": acc,
                "Test_Precision": prec,
                "Test_Recall": rec,
                "Test_F1": f1,
                "Best_Params": best_params_str,
            })

            # per-sample predictions (needed for Table S4)
            test_indices = list(test_idx)
            for local_i, global_idx in enumerate(test_indices):
                prediction_rows.append({
                    "ID": sample_ids.iloc[global_idx],
                    "SMILES": smiles_list.iloc[global_idx],
                    "Reference(s)": reference_col.iloc[global_idx],
                    "True_Class": int(y.iloc[global_idx]),
                    "Predicted_Class": int(y_pred[local_i]),
                    "Fold": fold_num,
                    "Model": model_name,
                    "Descriptor": descriptor_name,
                })

            # confusion matrix rows
            labels = [0, 1, 2]
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            for i_true, true_cls in enumerate(labels):
                for i_pred, pred_cls in enumerate(labels):
                    confmat_rows.append({
                        "Model": model_name,
                        "Descriptor": descriptor_name,
                        "Fold": fold_num,
                        "True_Class": true_cls,
                        "Predicted_Class": pred_cls,
                        "Count": int(cm[i_true, i_pred]),
                    })

            fold_num += 1

    return summary_rows, prediction_rows, confmat_rows


all_summary_rows = []
all_prediction_rows = []
all_confmat_rows = []

for desc_type, desc_file in descriptor_files.items():
    summary_rows, prediction_rows, confmat_rows = train_and_evaluate_models(desc_type, desc_file)
    all_summary_rows.extend(summary_rows)
    all_prediction_rows.extend(prediction_rows)
    all_confmat_rows.extend(confmat_rows)

# Save detailed summary
summary_df = pd.DataFrame(all_summary_rows)
summary_df.to_csv(summary_file, index=False)

# Save detailed predictions
predictions_df = pd.DataFrame(all_prediction_rows)
predictions_df.to_csv(predictions_file, index=False)

# Save raw confusion matrix counts
confmat_df = pd.DataFrame(all_confmat_rows)

# Sum confusion matrices over folds for each model/descriptor pair
confmat_summary_df = (
    confmat_df.groupby(["Model", "Descriptor", "True_Class", "Predicted_Class"], as_index=False)["Count"]
    .sum()
    .sort_values(["Model", "Descriptor", "True_Class", "Predicted_Class"])
)
confmat_summary_df.to_csv(confmat_file, index=False)

# =============================
# Aggregated performance table for Table S2
# =============================
means_df = (
    summary_df.groupby(["Model", "Descriptor"], as_index=False)
    .agg(
        Accuracy_Mean=("Test_Accuracy", "mean"),
        Accuracy_SD=("Test_Accuracy", "std"),
        Precision_Mean=("Test_Precision", "mean"),
        Precision_SD=("Test_Precision", "std"),
        Recall_Mean=("Test_Recall", "mean"),
        Recall_SD=("Test_Recall", "std"),
        F1_Mean=("Test_F1", "mean"),
        F1_SD=("Test_F1", "std"),
        Representative_Best_Params=("Best_Params", representative_params),
    )
    .sort_values(["Model", "Descriptor"])
)

# mean ± std 文字列も作っておく
means_df["Accuracy"] = means_df.apply(lambda r: f"{r['Accuracy_Mean']:.3f} ± {r['Accuracy_SD']:.3f}", axis=1)
means_df["Precision"] = means_df.apply(lambda r: f"{r['Precision_Mean']:.3f} ± {r['Precision_SD']:.3f}", axis=1)
means_df["Recall"] = means_df.apply(lambda r: f"{r['Recall_Mean']:.3f} ± {r['Recall_SD']:.3f}", axis=1)
means_df["F1 score"] = means_df.apply(lambda r: f"{r['F1_Mean']:.3f} ± {r['F1_SD']:.3f}", axis=1)

means_df.to_csv(means_file, index=False)

# =============================
# Hyperparameter summary for Table S3
# =============================
hyperparam_df = (
    summary_df.groupby(["Model", "Descriptor"], as_index=False)
    .agg(
        Representative_Best_Params=("Best_Params", representative_params),
        Fold1_Best_Params=("Best_Params", lambda s: s.iloc[0] if len(s) > 0 else ""),
        Fold2_Best_Params=("Best_Params", lambda s: s.iloc[1] if len(s) > 1 else ""),
        Fold3_Best_Params=("Best_Params", lambda s: s.iloc[2] if len(s) > 2 else ""),
        Fold4_Best_Params=("Best_Params", lambda s: s.iloc[3] if len(s) > 3 else ""),
        Fold5_Best_Params=("Best_Params", lambda s: s.iloc[4] if len(s) > 4 else ""),
    )
    .sort_values(["Model", "Descriptor"])
)
hyperparam_df.to_csv(hyperparam_file, index=False)

# =============================
# Best model summary
# =============================
best_row = means_df.sort_values(
    by=["Accuracy_Mean", "F1_Mean", "Precision_Mean", "Recall_Mean"],
    ascending=False
).iloc[0]

with open(best_model_file, "w", encoding="utf-8", newline="\n") as f:
    f.write("Best model summary\n")
    f.write("==================\n")
    f.write(f"Model: {best_row['Model']}\n")
    f.write(f"Descriptor: {best_row['Descriptor']}\n")
    f.write(f"Accuracy: {best_row['Accuracy']}\n")
    f.write(f"Precision: {best_row['Precision']}\n")
    f.write(f"Recall: {best_row['Recall']}\n")
    f.write(f"F1 score: {best_row['F1 score']}\n")
    f.write(f"Representative best params: {best_row['Representative_Best_Params']}\n")

print("\nDone.")
print("Generated files:")
print(f"  - {summary_file}")
print(f"  - {means_file}")
print(f"  - {hyperparam_file}")
print(f"  - {predictions_file}")
print(f"  - {confmat_file}")
print(f"  - {best_model_file}")
