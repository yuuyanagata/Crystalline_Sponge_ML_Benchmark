#!/usr/bin/env python3
"""
Predict the crystalline sponge class (0/1/2) for new SMILES strings,
using the best-performing random forest + Mordred descriptor pipeline
retrained on the full 279-entry modelled dataset.

Class codes:
  0 = unsuccessful CS analysis
  1 = structure solved with ZnI2-tpt only
  2 = structure solved with ZnCl2-tpt (or with both hosts in independent
      submissions; ZnCl2-tpt is the experimentally preferred host)

Usage:
  python src/predict.py "CCO" "c1ccccc1O"
  python src/predict.py --input path/to/smiles.txt

The text file should contain one SMILES per line.
"""
import argparse
import os
import sys
import joblib
import numpy as np
from rdkit import Chem, RDLogger
from mordred import Calculator, descriptors as mordred_descriptors

RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "random_forest_mordred_full.joblib")

CLASS_NAMES = {
    0: "unsuccessful CS analysis",
    1: "ZnI2-tpt only",
    2: "ZnCl2-tpt (preferred)",
}

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("smiles", nargs="*", help="SMILES strings to classify")
    p.add_argument("--input", "-i", default=None,
                   help="text file with one SMILES per line")
    return p.parse_args()

def main():
    args = parse_args()
    smiles_list = list(args.smiles)
    if args.input:
        with open(args.input) as f:
            smiles_list.extend(line.strip() for line in f if line.strip())
    if not smiles_list:
        print("No SMILES given.", file=sys.stderr)
        sys.exit(1)

    mols = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            print(f"WARNING: failed to parse SMILES {smi!r}", file=sys.stderr)
        mols.append(m)

    # Compute Mordred 2D descriptors
    calc = Calculator(mordred_descriptors, ignore_3D=True)
    mord_df = calc.pandas(mols, quiet=True, nproc=1)
    mord_np = mord_df.to_numpy()
    X = np.full(mord_np.shape, np.nan, dtype=np.float64)
    for i in range(mord_np.shape[0]):
        for j in range(mord_np.shape[1]):
            try:
                v = float(mord_np[i, j])
                if np.isfinite(v):
                    X[i, j] = v
            except Exception:
                pass

    # Match training-time column subset (drop all-NaN cols as in run_cv.py)
    # The saved pipeline was fitted on the training-time column subset, so we
    # need to apply the same drop. We assume the training matrix retained
    # 1513 columns of the 1613 Mordred features; we reproduce this by loading
    # the cached descriptor file's mask.
    cache = os.path.join(ROOT, "results", "descriptors", "mordred_descriptors.npy")
    keep = ~np.all(np.isnan(np.load(cache)), axis=0)
    X = X[:, keep]

    pipe = joblib.load(MODEL)
    pred = pipe.predict(X)

    print(f"\n{'SMILES':<60s} {'class':>6s}  meaning")
    print("-" * 100)
    for smi, c in zip(smiles_list, pred):
        print(f"{smi:<60s} {int(c):>6d}  {CLASS_NAMES[int(c)]}")

if __name__ == "__main__":
    main()
