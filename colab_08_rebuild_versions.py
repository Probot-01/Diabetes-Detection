import numpy as np
import pandas as pd

def main():
    print("=" * 60)
    print("🔄  DIAGNOSE AND REBUILD FEATURE VERSIONS")
    print("=" * 60)
    
    # ---------------------------------------------------------
    # STEP 1 - Diagnose & Align
    # ---------------------------------------------------------
    print("\nSTEP 1 - Diagnose & Align:")
    
    # Load arrays first so we know the exact target shape
    try:
        X_train_C = np.load('X_train.npy', allow_pickle=True).astype(np.float32)
        X_test_C  = np.load('X_test.npy', allow_pickle=True).astype(np.float32)
        y_train   = np.load('y_train_C.npy', allow_pickle=True)
        y_test    = np.load('y_test.npy', allow_pickle=True)
        target_size = X_train_C.shape[1]
    except Exception as e:
        print(f"Error loading arrays: {e}")
        return
        
    try:
        # Load feature names dynamically
        df_feats = pd.read_csv('feature_names.csv')
        if 'feature_name' in df_feats.columns:
            features = df_feats['feature_name'].tolist()
        else:
            df_feats = pd.read_csv('feature_names.csv', header=None)
            features = df_feats.iloc[:, 0].tolist()
            
        # CRITICAL ALIGNMENT FIX
        # Clean out known stray headers and previously dropped columns
        features = [f for f in features if str(f).strip().lower() not in [
            'feature_name', '0', 'unnamed: 0', 'is_synthetic', '', 'nan'
        ]]
        
        # If the list is STILL larger than our matrix, trim stray values from the front
        while len(features) > target_size:
            features = features[1:]
            
    except Exception as e:
        print(f"Failed to load feature_names.csv: {e}")
        return
        
    print(f"Total features loaded from CSV: {len(features)}\n")
    
    targets = ['AGE(in years)', 'GENDER', 'BMI', 'is_synthetic']
    indices = {}
    
    for t in targets:
        idx = None
        
        # 1. First, look for an exact case-insensitive match
        for i, f in enumerate(features):
            if str(f).lower() == str(t).lower():
                idx = i
                break
                
        # 2. Second, do a robust substring match if exact fails
        if idx is None:
            for i, f in enumerate(features):
                f_lower = str(f).lower()
                # Ensure we don't accidentally match words like 'average' or 'package' for age
                if t == 'AGE(in years)':
                    if f_lower == 'age' or 'age(' in f_lower or f_lower.startswith('age_') or f_lower.endswith('_age'):
                        idx = i
                        break
                # Generic substring match for everything else
                elif str(t).lower() in f_lower:
                    idx = i
                    break
                    
        indices[t] = idx
        if idx is not None:
            print(f"  {t:<15} : FOUND at index {idx} -> '{features[idx]}'")
        else:
            print(f"  {t:<15} : NOT FOUND")

    # ---------------------------------------------------------
    # STEP 2 - Rebuild all three versions
    # ---------------------------------------------------------
    print("\nSTEP 2 - Rebuild all three versions from scratch:")
    try:
        # Load the cleaned Version C base models
        X_train_C = np.load('X_train.npy', allow_pickle=True).astype(np.float32)
        X_test_C  = np.load('X_test.npy', allow_pickle=True).astype(np.float32)
        y_train   = np.load('y_train_C.npy', allow_pickle=True)
        y_test    = np.load('y_test.npy', allow_pickle=True)
        
        print(f"Base Version C loaded. X_train={X_train_C.shape}, X_test={X_test_C.shape}")
        
        idx_age    = indices.get('AGE(in years)')
        idx_gender = indices.get('GENDER')
        idx_bmi    = indices.get('BMI')
        
        # Determine valid indices to drop
        drop_for_A = [i for i in [idx_age, idx_gender, idx_bmi] if i is not None]
        drop_for_B = [i for i in [idx_age] if i is not None]
        
        # --- VERSION A ---
        print("\n--- Rebuilding Version A (Drop Age, BMI, Gender) ---")
        if len(drop_for_A) > 0:
            X_train_A = np.delete(X_train_C, drop_for_A, axis=1)
            X_test_A  = np.delete(X_test_C, drop_for_A, axis=1)
        else:
            print("WARNING: Could not find indices to drop. Version A will match C.")
            X_train_A, X_test_A = X_train_C.copy(), X_test_C.copy()
            
        y_train_A = y_train.copy()
        
        np.save('X_train_A.npy', X_train_A)
        np.save('X_test_A.npy', X_test_A)
        np.save('y_train_A.npy', y_train_A)
        print(f"Saved Version A: X_train={X_train_A.shape}, y_train={y_train_A.shape}")

        # --- VERSION B ---
        print("\n--- Rebuilding Version B (Drop Age only) ---")
        if len(drop_for_B) > 0:
            X_train_B = np.delete(X_train_C, drop_for_B, axis=1)
            X_test_B  = np.delete(X_test_C, drop_for_B, axis=1)
        else:
            print("WARNING: Could not find Age index to drop. Version B will match C.")
            X_train_B, X_test_B = X_train_C.copy(), X_test_C.copy()
            
        y_train_B = y_train.copy()
        
        np.save('X_train_B.npy', X_train_B)
        np.save('X_test_B.npy', X_test_B)
        np.save('y_train_B.npy', y_train_B)
        print(f"Saved Version B: X_train={X_train_B.shape}, y_train={y_train_B.shape}")

        # --- VERSION C ---
        print("\n--- Rebuilding Version C (Keep All) ---")
        # X_train.npy and X_test.npy are already C (and already lack leakage/is_synthetic)
        # We simply re-save y_train_C.npy for explicit file presence matching A and B
        np.save('y_train_C.npy', y_train)
        print("Version C is already X_train.npy and X_test.npy. Copied y_train_C.npy.")
        print(f"Saved Version C: X_train={X_train_C.shape}, y_train={y_train.shape}")
        
    except Exception as e:
        print(f"Error in STEP 2: {e}")
        return

    # ---------------------------------------------------------
    # STEP 3 - Verify
    # ---------------------------------------------------------
    print("\nSTEP 3 - Verify Shapes:")
    feats_A = X_train_A.shape[1]
    feats_B = X_train_B.shape[1]
    feats_C = X_train_C.shape[1]
    
    print(f"Version A features : {feats_A}")
    print(f"Version B features : {feats_B}")
    print(f"Version C features : {feats_C}\n")
    
    b_minus_a = feats_B - feats_A
    c_minus_b = feats_C - feats_B
    c_minus_a = feats_C - feats_A
    
    check_1 = (b_minus_a == 2)
    check_2 = (c_minus_b == 1)
    check_3 = (c_minus_a == 3)
    
    print(f"B - A = {b_minus_a} (Expected: 2) -> {'PASS' if check_1 else 'FAIL'}")
    print(f"C - B = {c_minus_b} (Expected: 1) -> {'PASS' if check_2 else 'FAIL'}")
    print(f"C - A = {c_minus_a} (Expected: 3) -> {'PASS' if check_3 else 'FAIL'}")
    
    if check_1 and check_2 and check_3:
        print("\n✅ OVERALL VERIFICATION: PASS")
    else:
        print("\n❌ OVERALL VERIFICATION: FAIL")

if __name__ == "__main__":
    main()
