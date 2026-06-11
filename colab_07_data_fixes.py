import numpy as np
import pandas as pd
import os

def main():
    print("=" * 60)
    print("🛠️  DATA FIXES PROTOCOL")
    print("=" * 60)
    
    fixes_status = {
        "is_synthetic": False,
        "leakage": False
    }

    # ==========================================
    # PROBLEM 1 - Remove is_synthetic
    # ==========================================
    print("\n[PROBLEM 1] Removing 'is_synthetic' column from all versions...")
    
    try:
        # 1. Read feature list
        df_feats = pd.read_csv('feature_names.csv')
        if 'feature_name' in df_feats.columns:
            features_C = df_feats['feature_name'].tolist()
        else:
            df_feats = pd.read_csv('feature_names.csv', header=None)
            features_C = df_feats.iloc[:, 0].tolist()
            
        def get_idx(feats, name):
            for i, f in enumerate(feats):
                if name.lower() in str(f).lower():
                    return i
            return None
            
        idx_C = get_idx(features_C, 'is_synthetic')
        
        # Load arrays (force float32 to match previous pipeline conventions)
        X_train_A = np.load('X_train_A.npy', allow_pickle=True).astype(np.float32)
        X_test_A  = np.load('X_test_A.npy', allow_pickle=True).astype(np.float32)
        X_train_B = np.load('X_train_B.npy', allow_pickle=True).astype(np.float32)
        X_test_B  = np.load('X_test_B.npy', allow_pickle=True).astype(np.float32)
        X_train_C = np.load('X_train.npy', allow_pickle=True).astype(np.float32)
        X_test_C  = np.load('X_test.npy', allow_pickle=True).astype(np.float32)
        
        if idx_C is not None:
            # Shift the index for A and B since they previously dropped other metadata columns
            dropped_for_A = [i for i, f in enumerate(features_C) if any(d.lower() in str(f).lower() for d in ['age', 'bmi', 'gender', 'sex'])]
            idx_A = idx_C - sum(1 for i in dropped_for_A if i < idx_C)
            
            dropped_for_B = [i for i, f in enumerate(features_C) if 'age' in str(f).lower()]
            idx_B = idx_C - sum(1 for i in dropped_for_B if i < idx_C)
            
            # Remove from A
            if idx_A < X_train_A.shape[1]:
                X_train_A = np.delete(X_train_A, idx_A, axis=1)
                X_test_A = np.delete(X_test_A, idx_A, axis=1)
                np.save('X_train_A.npy', X_train_A)
                np.save('X_test_A.npy', X_test_A)
                
            # Remove from B
            if idx_B < X_train_B.shape[1]:
                X_train_B = np.delete(X_train_B, idx_B, axis=1)
                X_test_B = np.delete(X_test_B, idx_B, axis=1)
                np.save('X_train_B.npy', X_train_B)
                np.save('X_test_B.npy', X_test_B)
                
            # Remove from C
            if idx_C < X_train_C.shape[1]:
                X_train_C = np.delete(X_train_C, idx_C, axis=1)
                X_test_C = np.delete(X_test_C, idx_C, axis=1)
                np.save('X_train.npy', X_train_C)
                np.save('X_test.npy', X_test_C)
                
            print(f"is_synthetic removed. New shapes: A={X_train_A.shape[1]} B={X_train_B.shape[1]} C={X_train_C.shape[1]}")
            fixes_status["is_synthetic"] = True
            
            # Remove it from feature_names.csv completely
            feat_list_clean = [f for f in features_C if 'is_synthetic' not in str(f).lower()]
            if 'feature_name' in df_feats.columns:
                pd.DataFrame({'feature_name': feat_list_clean}).to_csv('feature_names.csv', index=False)
            else:
                pd.DataFrame(feat_list_clean).to_csv('feature_names.csv', index=False, header=False)
            print("is_synthetic permanently dropped from feature_names.csv")
            
        else:
            print("is_synthetic column not found in feature list. It may have already been removed.")
            fixes_status["is_synthetic"] = True # Passed state if already clean
    except Exception as e:
        print(f"Error in Problem 1: {e}")

    # ==========================================
    # PROBLEM 2 - Fix Data Leakage
    # ==========================================
    print("\n[PROBLEM 2] Fixing data leakage (removing matching rows from train)...")
    
    def remove_leakage(train_file, test_file, y_train_source, y_train_out, version_name):
        # Load matrices
        X_tr = np.load(train_file, allow_pickle=True)
        X_te = np.load(test_file, allow_pickle=True)
        y_tr = np.load(y_train_source, allow_pickle=True)
        
        # Convert to pandas DataFrames
        df_train = pd.DataFrame(X_tr)
        df_test_unique = pd.DataFrame(X_te).drop_duplicates()
        
        # Find rows in X_train that are identical to any row in X_test
        merged = df_train.merge(df_test_unique, on=list(df_train.columns), how='left', indicator=True)
        
        # 'left_only' means the row was NOT found in df_test_unique
        to_keep = merged['_merge'] == 'left_only'
        
        removed_count = (~to_keep).sum()
        
        # Filter both X_train and y_train down to the clean rows
        X_tr_clean = X_tr[to_keep.values]
        y_tr_clean = y_tr[to_keep.values]
        
        # Save back to disk
        np.save(train_file, X_tr_clean)
        np.save(y_train_out, y_tr_clean)
        
        print(f"{version_name}: Removed {removed_count} duplicate rows. New train shape: {X_tr_clean.shape}")
        return removed_count, X_tr_clean.shape
        
    try:
        # Fix each version, reading original y_train.npy and saving versioned y_train outputs.
        rem_A, shape_A = remove_leakage('X_train_A.npy', 'X_test_A.npy', 'y_train.npy', 'y_train_A.npy', 'Version A')
        rem_B, shape_B = remove_leakage('X_train_B.npy', 'X_test_B.npy', 'y_train.npy', 'y_train_B.npy', 'Version B')
        rem_C, shape_C = remove_leakage('X_train.npy',   'X_test.npy',   'y_train.npy', 'y_train_C.npy', 'Version C')
        
        fixes_status["leakage"] = True
    except Exception as e:
        print(f"Error in Problem 2: {e}")

    # ==========================================
    # STEP 3 - Verify Fixes
    # ==========================================
    print("\n[STEP 3] Verifying fixes...")
    
    # 3.1 Verify is_synthetic is completely gone
    try:
        df_feats = pd.read_csv('feature_names.csv')
        feat_list = df_feats['feature_name'].tolist() if 'feature_name' in df_feats.columns else df_feats.iloc[:, 0].tolist()
        has_synthetic = any('is_synthetic' in str(f).lower() for f in feat_list)
        
        if not has_synthetic:
            print("PASS: is_synthetic not in feature list.")
            is_synth_pass = True
        else:
            print("FAIL: is_synthetic still in feature list.")
            is_synth_pass = False
    except:
        is_synth_pass = False

    # 3.2 Verify zero duplicate rows remain
    def check_dupes(train_file, test_file):
        X_tr = np.load(train_file, allow_pickle=True)
        X_te = np.load(test_file, allow_pickle=True)
        # Hash row matching for rapid duplicate counting
        s_tr = set(map(tuple, X_tr))
        s_te = set(map(tuple, X_te))
        return len(s_tr.intersection(s_te))

    try:
        dup_A = check_dupes('X_train_A.npy', 'X_test_A.npy')
        dup_B = check_dupes('X_train_B.npy', 'X_test_B.npy')
        dup_C = check_dupes('X_train.npy', 'X_test.npy')
        
        if dup_A == 0 and dup_B == 0 and dup_C == 0:
            print("PASS: Zero duplicate rows between train and test.")
            leakage_pass = True
        else:
            print(f"FAIL: Duplicates found -> A={dup_A}, B={dup_B}, C={dup_C}")
            leakage_pass = False
    except:
        leakage_pass = False

    # 3.3 Print validated shapes
    print("\nNew File Shapes:")
    print(f"X_train_A: {shape_A}, X_test_A: {np.load('X_test_A.npy', allow_pickle=True).shape}")
    print(f"X_train_B: {shape_B}, X_test_B: {np.load('X_test_B.npy', allow_pickle=True).shape}")
    print(f"X_train_C: {shape_C}, X_test_C: {np.load('X_test.npy', allow_pickle=True).shape}")
    
    # ==========================================
    # FINAL SUMMARY REPORT
    # ==========================================
    print("\n" + "╔" + "═"*38 + "╗")
    print("║" + "FIXES APPLIED".center(38) + "║")
    print("╠" + "═"*38 + "╣")
    
    icon_synth = "✅" if is_synth_pass else "❌"
    icon_leak = "✅" if leakage_pass else "❌"
    
    def print_summary_row(key, val):
        row = f" {key:<23}: {val} "
        pad = 38 - len(row)
        if pad < 0: pad = 0
        print("║" + row + " "*pad + "║")
        
    print_summary_row("is_synthetic removed", icon_synth)
    print_summary_row("Leakage rows removed", icon_leak)
    print_summary_row("Version A train shape", str(shape_A))
    print_summary_row("Version B train shape", str(shape_B))
    print_summary_row("Version C train shape", str(shape_C))
    
    print("╚" + "═"*38 + "╝")

if __name__ == "__main__":
    main()
