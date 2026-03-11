# Crystalline Sponge ML Benchmark

Machine learning benchmark for predicting the success of crystalline sponge analysis using molecular descriptors.

This script calculates multiple molecular descriptors from SMILES strings and evaluates several machine learning models using nested cross-validation.

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
- Detailed output tables suitable for scientific publications

## Input

The script expects a CSV file:

CS_Booklet-2.csv

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

The script automatically generates:

- model_performance_summary.csv
- model_performance_summary_means.csv
- hyperparameter_summary.csv
- predictions_detailed.csv
- confusion_matrices_summary.csv
- best_model_summary.txt

Descriptor files are also generated:

- rdkit_descriptors.csv
- mordred_descriptors.csv
- morgan_descriptors.csv
- avalon_descriptors.csv
- maccs_descriptors.csv

## Requirements

Python packages:

numpy  
pandas  
scikit-learn  
rdkit  
mordred  

Example installation:

pip install numpy pandas scikit-learn mordred rdkit

## Usage

Run the script:

python 5_Algorithms_Cl_and_I.py

The program will:

1. Calculate descriptors
2. Train ML models
3. Perform nested cross-validation
4. Output performance tables

## License

This project is licensed under the MIT License.
