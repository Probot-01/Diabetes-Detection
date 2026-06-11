# Voice-Based Diabetes Detection

An end-to-end Machine Learning pipeline designed to detect diabetes directly from vocal biomarkers (MFCCs, LPCs, ZCR, etc.) alongside optional biological demographics. 

This project is built explicitly to execute sequentially inside Google Colab environments.

## Overview

The core objective of this repository is to train and evaluate robust classification models (Random Forest, XGBoost, SVM) on voice data. A significant portion of this pipeline is dedicated to extreme data hygiene—ensuring zero train/test data leakage from SMOTE resampling, systematically proving the absence of confounding variables (like `is_synthetic` markers or Blood Sugar Levels), and explicitly isolating demographics (Age, Gender, BMI) to measure the raw predictive power of the voice features alone.

## Data Versions (A, B, C)

To properly understand the impact of confounding demographic variables, the dataset is split into three distinct versions prior to model training:
- **Version A (Voice Only):** Drops Age, BMI, and Gender. Measures pure vocal predictive power.
- **Version B (Voice + BMI/Gender):** Drops Age (a massive confounder), but retains BMI and Gender.
- **Version C (All Features):** Contains all raw features including Age (⚠️ known confound).

## Pipeline Architecture

The scripts should be copied and run in Google Colab sequentially:

### Phase 1: Data Ingestion & Preprocessing
- `colab_01_data_inspection.py` - Initial CSV load, class balance verification, and EDA.
- `colab_02_preprocessing.py` - Drops direct medical indicators (`BSL`) to prevent cheating, cleans datasets.
- `colab_03_eda.py` - Deeper exploratory data analysis.
- `colab_04_data_preparation.py` - Train/Test splits, applies SMOTE for imbalance, and executes `StandardScaler`. Outputs initial `.npy` arrays.
- `colab_04b_feature_name_check.py` / `colab_setup.py` - Environment and variable setup.

### Phase 2: Feature Isolation & Cleansing
- `create_versions.py` - Initially partitions the data into Versions A, B, and C.
- `colab_07_data_fixes.py` - The heavy-lifting script that purges the `is_synthetic` column and rigorously deletes duplicate SMOTE leakage rows from the training matrices.
- `colab_08_rebuild_versions.py` - Cleanly reconstructs Versions A, B, and C directly from the safely scrubbed data arrays to guarantee flawless alignments.

### Phase 3: Diagnostics & Validation
- `colab_11_gender_confound.py` - Cross-tabulates label classes to detect if gender was inadvertently skewing predictions >20%.
- `colab_06_data_validation.py` - Validates the absence of forbidden columns, missing values, and calculates train/test row intersection.
- `colab_09_final_diagnostic.py` - The "Final Flight Check". Ensures matrix shapes align flawlessly right before training.

### Phase 4: Training & Explainability
- `colab_05_model_training.py` - Dynamically ingests the Version A, B, and C matrices and trains an RF, XGB, and SVM on each. Prints comparison tables (AUC-ROC, Sensitivity, F1).
- `colab_10_post_training.py` - Re-trains the ultimate winning model (XGBoost Version B), saves it to `best_model.pkl`, and spins up `shap.TreeExplainer()` to generate visual intelligence graphs mapping precisely *why* the model made its diagnoses.

## Setup
Upload your original `diabetes_voice_dataset.csv` directly into the base file directory of your Google Colab instance, and begin executing the pipeline.

## Outputs
- **`.npy` arrays**: Standardized and cleansed matrices.
- **`best_model.pkl`**: The fully trained, serialized XGBoost model.
- **`shap_*.png`**: High-resolution SHAP visual diagrams explaining algorithmic rationale.
