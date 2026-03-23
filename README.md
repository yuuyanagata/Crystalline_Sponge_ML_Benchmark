# Crystalline Sponge ML Benchmark

Machine learning benchmark for predicting the success of crystalline sponge analysis using molecular descriptors.

This script calculates multiple molecular descriptors from SMILES strings and evaluates several machine learning models using nested cross-validation. In addition to the fold-wise models generated during cross-validation, the modified version also retrains each model and descriptor combination on the full dataset and saves a single final model as a `.pkl` file.

The workflow was developed for analyzing host–guest compatibility in crystalline sponge systems such as ZnCl2–tpt and ZnI2–tpt.

## Features

- Descriptor calculation from SMILES
  - RDKit descriptors
  - Mordred descriptors
  - Morgan fingerprints
  - Avalon fingerprints
  - MACCS keys

- Machine learning models
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - k-Nearest Neighbors (kNN)
  - Multi-Layer Perceptron (MLP)

- Nested cross-validation (5×5)
- Automatic hyperparameter optimization
- Fold-wise model export during outer cross-validation
- Final retraining on the full dataset for all model and descriptor combinations
- Detailed output tables suitable for scientific publications

## Input

The script expects a CSV file:

`CS_Booklet.csv`

Required columns:

| Column | Description |
|------|-------------|
| SMILES | Molecular SMILES string |
| ExperimentResult | Class label |

Class labels:

| Value | Meaning |
|------|---------|
| 0 | Analysis failed |
| 1 | Successful with ZnI2-tpt |
| 2 | Successful with ZnCl2-tpt |

Optional columns:

- ID
- Reference(s)

## Output Files

The script automatically generates the following summary files:

- `model_performance_summary.csv`
- `model_performance_summary_means.csv`
- `hyperparameter_summary.csv`
- `predictions_detailed.csv`
- `confusion_matrices_summary.csv`
- `best_model_summary.txt`

Descriptor files are also generated:

- `rdkit_descriptors.csv`
- `mordred_descriptors.csv`
- `morgan_descriptors.csv`
- `avalon_descriptors.csv`
- `maccs_descriptors.csv`

Fold-wise models generated during nested cross-validation are saved as:

- `<Model>_<Descriptor>_fold1.pkl`
- `<Model>_<Descriptor>_fold2.pkl`
- `<Model>_<Descriptor>_fold3.pkl`
- `<Model>_<Descriptor>_fold4.pkl`
- `<Model>_<Descriptor>_fold5.pkl`

Examples:

- `SVM_MACCS_fold1.pkl`
- `RandomForest_Morgan_fold3.pkl`

Final models retrained on the full dataset are saved for all model and descriptor combinations as:

- `final_<Model>_<Descriptor>.pkl`

Examples:

- `final_SVM_MACCS.pkl`
- `final_LogisticRegression_RDKit.pkl`
- `final_RandomForest_Mordred.pkl`

These final `.pkl` files contain a scikit-learn `Pipeline` with the following steps:

1. `SimpleImputer(strategy="mean")`
2. `StandardScaler()`
3. Trained classifier with the representative hyperparameters selected from nested cross-validation

## Requirements

Python packages:

- numpy
- pandas
- scikit-learn
- rdkit
- mordred

Example installation:

```bash
pip install numpy pandas scikit-learn mordred rdkit
```

## Usage

Run the modified script:

```bash
python 5_Algorithms_Cl_and_I_modified.py
```

The program will:

1. Calculate descriptors
2. Train ML models
3. Perform nested cross-validation
4. Save fold-wise evaluation models
5. Retrain final models on the full dataset for all model and descriptor combinations
6. Output performance tables and summary files

## Notes

The fold-wise `.pkl` files correspond to models obtained in individual outer folds of nested cross-validation. These files are useful for evaluation traceability.

The `final_*.pkl` files are the recommended models for prediction on new compounds because they are retrained on the full dataset and include the preprocessing pipeline.

## License

This project is licensed under the MIT License.
