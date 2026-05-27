# Crystalline Sponge ML Benchmark

Molecular descriptor-based prediction of host–guest compatibility in the crystalline sponge (CS) method.

This repository contains the code, dataset, descriptor matrices, trained models, and per-fold cross-validation outputs for the manuscript:

> *Molecular descriptor-based prediction of host–guest compatibility in the crystalline sponge method*
> Y. Nagata et al., submitted to *RSC Advances*, 2026.

The work formulates CS host selection as a three-class classification task:

| Class | Meaning |
|-----|---------|
| 0 | Unsuccessful CS analysis |
| 1 | Structure solved with ZnI₂–tpt only |
| 2 | Structure solved with ZnCl₂–tpt, or with both hosts in independent submissions |

ZnCl₂–tpt is the experimentally preferred host owing to its higher framework stability and easier handling under routine laboratory conditions; therefore, when a guest molecule had been reported as compatible with both hosts in independent submissions, the conflict was resolved in favour of class 2 during data preparation.

## Headline results

Nested 5×5 stratified cross-validation on 279 entries, five molecular representations × five classifiers (25 combinations):

| Configuration | Mean test accuracy | Weighted mean test F1 |
|---|---|---|
| **Random Forest / Mordred** | **0.674 ± 0.043** | **0.669 ± 0.043** |
| Random Forest / RDKit | 0.674 ± 0.036 | 0.665 ± 0.037 |
| SVM / MACCS | 0.656 ± 0.088 | 0.637 ± 0.097 |
| Logistic Regression / Mordred | 0.646 ± 0.068 | 0.644 ± 0.067 |
| Random Forest / MACCS | 0.645 ± 0.061 | 0.639 ± 0.064 |

Trivial majority-class baseline: 113 / 279 = 0.405.

## Repository layout

```
.
├── data/
│   ├── CS_Booklet_raw.csv   raw curated dataset (300 entries before consolidation)
│   └── CS_Booklet_v5.csv    modelled dataset (279 entries; the input to gen_descriptors.py)
├── src/
│   ├── consolidate.py       deduplication and label-conflict resolution
│   ├── gen_descriptors.py   compute RDKit, Mordred, Morgan, Avalon, MACCS matrices
│   ├── run_cv.py            nested 5×5 stratified CV for all 25 combinations
│   ├── regen_figures.py     reproduce manuscript Figures 3 and 4
│   └── predict.py           inference on new SMILES with the best model
├── results/
│   ├── descriptors/         cached descriptor matrices (numpy arrays + names)
│   ├── cv/                  per-fold and aggregated CV outputs
│   └── figures/             benchmark heatmap (Fig. 3) and feature importance (Fig. 4)
├── models/                  every (model, descriptor) combination retrained on
│                            the full 279-entry dataset (25 .joblib files)
├── requirements.txt
├── LICENSE                  MIT (code) / CC-BY-4.0 (data and outputs)
└── README.md
```

## Quick start

```bash
# Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt

# Reproduce the benchmark from the cached descriptor matrices in
# results/descriptors/ (the only path that works in the public release,
# because industrial SMILES are masked).
python src/run_cv.py              # ~10 minutes, also retrains and saves models
python src/regen_figures.py       # produces Figures 3 and 4

# Inference: predict the CS class for new compounds (works because the
# trained model and Mordred pipeline are fully released).
python src/predict.py "CCO" "c1ccc(O)cc1"
```

In the public release, `src/gen_descriptors.py` and `src/consolidate.py`
cannot run end-to-end because the SMILES strings for the 127 industrial
entries have been masked. Use the cached descriptor matrices instead
(see the Dataset section below). For end-users running inference on
their own compounds, `predict.py` only needs the trained model and
works without any access to the training SMILES.

The benchmark uses two fixed random seeds (`OUTER=42`, `INNER=43`) and a deterministic per-classifier `random_state=0`, so reproducibility across machines is reliable up to BLAS-level numerical differences.

## Dataset

The modelled set of 279 entries was derived from a curated set of 300 raw entries: 169 from literature reports and 131 from collaborative studies with industrial partners (covering internal ID range ≥ 181). Of the 180 literature-derived compounds listed in Table S1 of the manuscript ESI, **eleven (Table S1 IDs 23, 33, 59, 66, 70, 126, 157, 167, 169, 170, 175) are SMILES-level duplicates of other compounds already present in the assembled raw dataset under different internal ID numbers**, with identical SMILES strings and identical class labels; these eleven entries were absorbed into their existing counterparts at the dataset-assembly stage. The literature-derived portion of the raw dataset therefore contains 169 entries. During the constitutional-duplicate consolidation step encoded in `src/consolidate.py`, 18 further isomeric-SMILES groups consolidated 39 raw entries into 18 representative entries, removing 21 duplicate rows; nine of those groups carried inconsistent class labels, all resolved in favour of class 2 (see the consolidation log printed by the script).

The five remaining mixed-class groups in the modelled set are genuine stereoisomer sets (11 entries, 3.9% of the dataset) and represent an irreducible 2D-descriptor error floor of approximately 4%; see Table S6 of the manuscript ESI.

### Public release note — industrial SMILES masked

The 131 raw entries (127 in the modelled set) contributed by industrial partners are subject to confidentiality agreements. In this public release, their SMILES strings have been replaced with the placeholder `<industrial-confidential>` in `data/CS_Booklet_raw.csv` and `data/CS_Booklet_v5.csv`. Class labels and entry IDs are kept intact.

Because of this, `src/gen_descriptors.py` and `src/consolidate.py` **cannot** reprocess the industrial entries from SMILES. To reproduce the benchmark, use the cached descriptor matrices in `results/descriptors/`, which include all 279 entries as numerical features and are sufficient for `src/run_cv.py`, `src/regen_figures.py`, and downstream analyses. Requests for the original industrial SMILES can be directed to the corresponding author and will be considered on a case-by-case basis in consultation with the relevant partner organizations.

## Trained models

`models/` contains every (algorithm, descriptor) combination retrained on the full 279-entry dataset using the representative best hyperparameters from cross-validation. Each model is a `sklearn.pipeline.Pipeline` containing a `SimpleImputer(mean)`, a `StandardScaler`, and the classifier, so it can be loaded with `joblib.load(...)` and called directly on new feature vectors:

```python
import joblib
pipe = joblib.load("models/random_forest_mordred_full.joblib")
pred = pipe.predict(X_new_mordred)        # shape (n_samples, n_features)
proba = pipe.predict_proba(X_new_mordred)
```

The `predict.py` helper script automates the SMILES → Mordred descriptors → prediction pipeline for the best model.

## Citation

If you use this code, dataset, or trained models, please cite:

```
Y. Nagata et al., "Molecular descriptor-based prediction of host-guest
compatibility in the crystalline sponge method", RSC Advances (2026).
```

## License

- **Code (`src/`)** — MIT License (see `LICENSE`)
- **Data and outputs (`data/`, `results/`, `models/`)** — CC-BY-4.0
