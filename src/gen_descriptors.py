#!/usr/bin/env python3
"""
Generate the five descriptor matrices used in the benchmark.

Reads:
  data/CS_Booklet_v5.csv   (the cleaned, 279-entry modelled dataset)

Writes (to results/descriptors/):
  rdkit_descriptors.npy            (RDKit physicochemical, ~217 cols)
  mordred_descriptors.npy          (Mordred 2D-only, 1613 cols; all-NaN cols
                                    kept here, dropped by run_cv.py)
  morgan_fingerprints.npy          (Morgan, radius=2, 2048 bits)
  avalon_fingerprints.npy          (Avalon, 512 bits)
  maccs_keys.npy                   (MACCS, 167 bits)
  labels.npy                       (length 279)
  ids.npy                          (length 279)
  rdkit_names.txt
  mordred_names.txt

Run from the repository root:
  python src/gen_descriptors.py
"""
import os
import time
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit.Avalon import pyAvalonTools
from mordred import Calculator, descriptors as mordred_descriptors

RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "descriptors")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(ROOT, "data", "CS_Booklet_v5.csv"))
df["SMILES"] = df["SMILES"].str.strip()
df["ExperimentResult"] = df["ExperimentResult"].astype(int)
print(f"Entries: {len(df)}")
print(f"Class distribution: {df['ExperimentResult'].value_counts().sort_index().to_dict()}")

# If SMILES is masked ('<industrial-confidential>'), Chem.MolFromSmiles
# returns None. Use the cached descriptor files in that case.
mols = [Chem.MolFromSmiles(s) for s in df["SMILES"]]
n_bad = sum(1 for m in mols if m is None)
if n_bad:
    print(f"WARNING: {n_bad} SMILES could not be parsed (likely masked rows).")
    print("This script needs all SMILES to be valid. Use the cached descriptor")
    print("files in results/descriptors/ to reproduce the benchmark instead.")
    raise SystemExit(1)

np.save(os.path.join(OUT, "labels.npy"), df["ExperimentResult"].to_numpy())
np.save(os.path.join(OUT, "ids.npy"), df["ID"].to_numpy())

print("\nRDKit descriptors ...")
descriptor_list = [(name, fn) for name, fn in Descriptors._descList]
rdkit_names = [n for n, _ in descriptor_list]
X_rdkit = np.full((len(mols), len(rdkit_names)), np.nan, dtype=np.float64)
for i, m in enumerate(mols):
    for j, (_, fn) in enumerate(descriptor_list):
        try:
            X_rdkit[i, j] = fn(m)
        except Exception:
            pass
print(f"  shape: {X_rdkit.shape}  (NaN count {int(np.isnan(X_rdkit).sum())})")
np.save(os.path.join(OUT, "rdkit_descriptors.npy"), X_rdkit)
with open(os.path.join(OUT, "rdkit_names.txt"), "w") as f:
    f.write("\n".join(rdkit_names))

print("\nMordred descriptors (this takes a few minutes) ...")
calc = Calculator(mordred_descriptors, ignore_3D=True)
t0 = time.time()
mord_df = calc.pandas(mols, quiet=True, nproc=1)
print(f"  elapsed: {time.time()-t0:.1f}s   shape: {mord_df.shape}")
mord_np = mord_df.to_numpy()
mord_clean = np.full(mord_np.shape, np.nan, dtype=np.float64)
for i in range(mord_np.shape[0]):
    for j in range(mord_np.shape[1]):
        try:
            fv = float(mord_np[i, j])
            if np.isfinite(fv):
                mord_clean[i, j] = fv
        except Exception:
            pass
print(f"  finite cells: {(~np.isnan(mord_clean)).sum()}/{mord_clean.size}")
np.save(os.path.join(OUT, "mordred_descriptors.npy"), mord_clean)
with open(os.path.join(OUT, "mordred_names.txt"), "w") as f:
    f.write("\n".join(map(str, mord_df.columns)))

print("\nMorgan fingerprints (radius=2, 2048 bits) ...")
X_morgan = np.zeros((len(mols), 2048), dtype=np.uint8)
for i, m in enumerate(mols):
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
    arr = np.zeros((2048,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    X_morgan[i] = arr
np.save(os.path.join(OUT, "morgan_fingerprints.npy"), X_morgan)
print(f"  shape: {X_morgan.shape}")

print("\nAvalon fingerprints (512 bits) ...")
X_avalon = np.zeros((len(mols), 512), dtype=np.uint8)
for i, m in enumerate(mols):
    fp = pyAvalonTools.GetAvalonFP(m, nBits=512)
    arr = np.zeros((512,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    X_avalon[i] = arr
np.save(os.path.join(OUT, "avalon_fingerprints.npy"), X_avalon)
print(f"  shape: {X_avalon.shape}")

print("\nMACCS keys (167 bits) ...")
X_maccs = np.zeros((len(mols), 167), dtype=np.uint8)
for i, m in enumerate(mols):
    fp = MACCSkeys.GenMACCSKeys(m)
    arr = np.zeros((167,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    X_maccs[i] = arr
np.save(os.path.join(OUT, "maccs_keys.npy"), X_maccs)
print(f"  shape: {X_maccs.shape}")

print("\nDone.")
