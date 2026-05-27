#!/usr/bin/env python3
"""
Constitutional-duplicate consolidation that produced CS_Booklet_v5.csv from
the 300-entry raw dataset CS_Booklet_raw.csv.

Rules:
  (1) Group entries by isomeric canonical SMILES. Entries that share an
      isomeric canonical SMILES represent the SAME molecule and should
      appear only once in the modelled dataset.
  (2) Within each group, the entry with the smallest ID is retained as
      the representative; the other IDs are removed.
  (3) When labels within a group disagree, the conflict is resolved in
      favour of class 2 (ZnCl2-tpt). ZnCl2-tpt is the experimentally
      preferred host owing to its higher framework stability and easier
      handling under routine laboratory conditions, so a compound that
      can be analysed by either host is operationally assigned to class 2.

Reads:
  data/CS_Booklet_raw.csv   (300 entries with the raw class labels)

Writes:
  data/CS_Booklet_v5.csv    (279 entries; the modelled set)

Run from the repository root:
  python src/consolidate.py
"""
import os
import pandas as pd
from collections import defaultdict, Counter
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data", "CS_Booklet_raw.csv"))
df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
df["SMILES"] = df["SMILES"].str.strip()
df["ExperimentResult"] = df["ExperimentResult"].astype(int)

def iso(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, isomericSmiles=True, canonical=True) if m else None

df["iso"] = df["SMILES"].apply(iso)

print("Before consolidation:")
print(f"  N = {len(df)}")
print(f"  class distribution: {Counter(df['ExperimentResult'])}")

groups = defaultdict(list)
for _, r in df.iterrows():
    groups[r["iso"]].append((r["ID"], r["ExperimentResult"]))
dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
print(f"\nConstitutional-duplicate groups: {len(dup_groups)}")
print(f"  total entries in such groups: {sum(len(v) for v in dup_groups.values())}")

keep_id_per_group = {}
final_class_per_group = {}
removed_ids = []
print("\nConsolidation log:")
for iso_smi, members in sorted(dup_groups.items(),
                               key=lambda kv: min(m[0] for m in kv[1])):
    classes = {c for _, c in members}
    final_cls = 2 if 2 in classes else (1 if 1 in classes else 0)
    keep_id = min(m[0] for m in members)
    keep_id_per_group[iso_smi] = keep_id
    final_class_per_group[iso_smi] = final_cls
    for mid, _ in members:
        if mid != keep_id:
            removed_ids.append(mid)
    note = " <- conflict resolved to class 2" if len(classes) > 1 else ""
    members_str = ", ".join(f"{m}({c})" for m, c in sorted(members))
    print(f"  keep ID {keep_id:4d}, final class {final_cls}  | {members_str}{note}")

new_rows = []
seen = set()
for _, r in df.iterrows():
    iso_smi = r["iso"]
    if iso_smi in dup_groups:
        if r["ID"] != keep_id_per_group[iso_smi]:
            continue
        if iso_smi in seen:
            continue
        seen.add(iso_smi)
        new_rows.append({
            "ID": r["ID"], "SMILES": r["SMILES"],
            "ExperimentResult": final_class_per_group[iso_smi],
        })
    else:
        new_rows.append({
            "ID": r["ID"], "SMILES": r["SMILES"],
            "ExperimentResult": r["ExperimentResult"],
        })

new_df = pd.DataFrame(new_rows)
print("\nAfter consolidation:")
print(f"  N = {len(new_df)}")
print(f"  class distribution: {Counter(new_df['ExperimentResult'])}")
print(f"  removed {len(removed_ids)} duplicate rows")
print(f"  removed IDs: {sorted(removed_ids)}")

out = os.path.join(ROOT, "data", "CS_Booklet_v5.csv")
new_df[["ID", "SMILES", "ExperimentResult"]].to_csv(out, index=False)
print(f"\nSaved: {out}")
