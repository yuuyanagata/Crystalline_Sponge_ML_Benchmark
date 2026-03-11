# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# ============================
DESC_FILE = "rdkit_descriptors.csv"
LABEL_FILE = "CS_Booklet-2.csv"
TARGET_COL = "ExperimentResult"

OUTPUT_CSV = "feature_importance_rdkit_rf.csv"
OUTPUT_PNG = "feature_importance_rdkit_rf_top20.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 500

# ============================
print("Loading data ...")
desc_df = pd.read_csv(DESC_FILE)
label_df = pd.read_csv(LABEL_FILE)

if TARGET_COL not in label_df.columns:
    raise KeyError(f"'{TARGET_COL}' column was not found in {LABEL_FILE}")

y = label_df[TARGET_COL]

# ============================
X_raw = desc_df.select_dtypes(include=[np.number]).copy()

if X_raw.shape[1] == 0:
    raise ValueError("No numeric descriptor columns were found in rdkit_descriptors.csv")

print(f"Descriptor matrix shape before imputation: {X_raw.shape}")

all_nan_cols = X_raw.columns[X_raw.isna().all()].tolist()
if all_nan_cols:
    print(f"Removing all-NaN columns: {len(all_nan_cols)}")
    X_raw = X_raw.drop(columns=all_nan_cols)

imputer = SimpleImputer(strategy="mean")
X_imp = imputer.fit_transform(X_raw)
X = pd.DataFrame(X_imp, columns=X_raw.columns)

print(f"Descriptor matrix shape after imputation: {X.shape}")
print(f"Target size: {len(y)}")

# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Train shape: {X_train.shape}")
print(f"Test shape : {X_test.shape}")

# ============================
print("Training Random Forest ...")
rf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

train_acc = rf.score(X_train, y_train)
test_acc = rf.score(X_test, y_test)

print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy : {test_acc:.4f}")

# ============================
importances = rf.feature_importances_

fi_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values("Importance", ascending=False)

fi_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"Saved feature importance table to: {OUTPUT_CSV}")

# ============================
top_n = min(20, len(fi_df))
top_df = fi_df.head(top_n).copy()

plt.figure(figsize=(10, max(6, top_n * 0.35)))

cmap = plt.cm.viridis
colors = cmap(np.linspace(0.15, 0.9, top_n))

plt.barh(top_df["Feature"], top_df["Importance"], color=colors)
plt.gca().invert_yaxis()

plt.title("Top Feature Importances (RDKit + Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved figure to: {OUTPUT_PNG}")

# ============================
print("\nTop feature importances:")
print(fi_df.head(20).to_string(index=False))

print("\nDone.")