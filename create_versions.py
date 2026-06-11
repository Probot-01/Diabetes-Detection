import numpy as np
import pandas as pd
import os

def find_index(features, keyword):
    keyword = keyword.lower()
    for i, name in enumerate(features):
        if keyword in str(name).lower():
            return i
    return None

def create_versions():
    print("STEP 1 - Loading files...")
    required_files = ['X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy', 'feature_names.csv']
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Warning: Missing files {missing_files}. The script may fail if these are required.")

    try:
        X_train = np.load('X_train.npy', allow_pickle=True)
        X_test = np.load('X_test.npy', allow_pickle=True)
        y_train = np.load('y_train.npy', allow_pickle=True)
        y_test = np.load('y_test.npy', allow_pickle=True)
        
        # Load feature names dynamically based on how it was saved
        df_feats = pd.read_csv('feature_names.csv')
        if 'feature_name' in df_feats.columns:
            feature_names = df_feats['feature_name'].tolist()
        else:
            # Fallback for files without a header row
            df_feats = pd.read_csv('feature_names.csv', header=None)
            if df_feats.shape[1] == 1:
                feature_names = df_feats.values.flatten().tolist()
            elif df_feats.shape[0] == 1:
                feature_names = df_feats.iloc[0].tolist()
            else:
                feature_names = df_feats.iloc[:, 0].tolist()
            
        print(f"Loaded {len(feature_names)} features.")
        print(f"Feature names list: {feature_names}\n")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print("STEP 2 - Finding column indices for Age, BMI, Gender...")
    age_idx = find_index(feature_names, 'age')
    bmi_idx = find_index(feature_names, 'bmi')
    
    gender_idx = find_index(feature_names, 'gender')
    if gender_idx is None:
        # Fallback in case it's named 'sex'
        gender_idx = find_index(feature_names, 'sex')

    print(f"Age index: {age_idx}")
    print(f"BMI index: {bmi_idx}")
    print(f"Gender index: {gender_idx}\n")

    # Indices to drop
    indices_all_metadata = [idx for idx in [age_idx, bmi_idx, gender_idx] if idx is not None]
    indices_age_only = [idx for idx in [age_idx] if idx is not None]

    print("STEP 3 - Version A (Voice only):")
    if len(indices_all_metadata) > 0:
        X_train_A = np.delete(X_train, indices_all_metadata, axis=1)
        X_test_A = np.delete(X_test, indices_all_metadata, axis=1)
    else:
        print("Warning: Age, BMI, and Gender not found. Version A will be identical to original.")
        X_train_A = X_train.copy()
        X_test_A = X_test.copy()
        
    np.save('X_train_A.npy', X_train_A)
    np.save('X_test_A.npy', X_test_A)
    print("Dropped Age, BMI, Gender from X_train and X_test.")
    print(f"Saved X_train_A.npy and X_test_A.npy. Shape: {X_train_A.shape}\n")

    print("STEP 4 - Version B (Voice + BMI + Gender, NO Age):")
    if len(indices_age_only) > 0:
        X_train_B = np.delete(X_train, indices_age_only, axis=1)
        X_test_B = np.delete(X_test, indices_age_only, axis=1)
    else:
        print("Warning: Age not found. Version B will be identical to original.")
        X_train_B = X_train.copy()
        X_test_B = X_test.copy()
        
    np.save('X_train_B.npy', X_train_B)
    np.save('X_test_B.npy', X_test_B)
    print("Dropped only Age from X_train and X_test.")
    print(f"Saved X_train_B.npy and X_test_B.npy. Shape: {X_train_B.shape}\n")

    print("STEP 5 - Version C is already X_train.npy and X_test.npy.")
    X_train_C = X_train
    print("No action needed.")
    print(f"Shape: {X_train_C.shape}\n")

    print("STEP 6 - Summary:")
    feats_A = X_train_A.shape[1] if X_train_A.ndim > 1 else 1
    feats_B = X_train_B.shape[1] if X_train_B.ndim > 1 else 1
    feats_C = X_train_C.shape[1] if X_train_C.ndim > 1 else 1
    
    print("╔══════════════════════════════════════════╗")
    print("║          VERSION SUMMARY                 ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║ Version A — Voice only      : {feats_A:<3} feats  ║")
    print(f"║ Version B — Voice+BMI/Gender: {feats_B:<3} feats  ║")
    print(f"║ Version C — All + Age       : {feats_C:<3} feats  ║")
    print("╚══════════════════════════════════════════╝")

if __name__ == "__main__":
    create_versions()
