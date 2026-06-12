import numpy as np
import pandas as pd

def main():
    print("=" * 65)
    print("🚀 FINAL PRE-TRAINING FLIGHT CHECK & DIAGNOSTIC 🚀")
    print("=" * 65)

    checks = []
    
    def check_condition(name, condition, details=""):
        status = "✅ PASS" if condition else "❌ FAIL"
        checks.append((name, condition))
        print(f"{status} | {name:<35} | {details}")
        return condition

    # ---------------------------------------------------------
    # 1. FILE LOADING & SHAPES
    # ---------------------------------------------------------
    print("\n--- 1. FILE LOADING ---")
    try:
        # Load exactly what the training script will consume
        X_tr_A = np.load('X_train_A.npy', allow_pickle=True)
        X_te_A = np.load('X_test_A.npy', allow_pickle=True)
        y_tr_A = np.load('y_train_A.npy', allow_pickle=True)
        
        X_tr_B = np.load('X_train_B.npy', allow_pickle=True)
        X_te_B = np.load('X_test_B.npy', allow_pickle=True)
        y_tr_B = np.load('y_train_B.npy', allow_pickle=True)
        
        # Version C uses the base train/test files
        X_tr_C = np.load('X_train.npy', allow_pickle=True)
        X_te_C = np.load('X_test.npy', allow_pickle=True)
        y_tr_C = np.load('y_train_C.npy', allow_pickle=True)
        
        y_test = np.load('y_test.npy', allow_pickle=True)
        
        check_condition("Files Loaded Successfully", True, "All 10 .npy files located")
    except Exception as e:
        check_condition("Files Loaded Successfully", False, str(e))
        return

    # ---------------------------------------------------------
    # 2. DIMENSIONAL ALIGNMENT
    # ---------------------------------------------------------
    print("\n--- 2. DIMENSIONAL ALIGNMENT (CRITICAL FOR ML) ---")
    
    # Feature columns must match between Train and Test
    check_condition("Ver A Features Match (Train=Test)", X_tr_A.shape[1] == X_te_A.shape[1], f"Both have {X_tr_A.shape[1]} cols")
    check_condition("Ver B Features Match (Train=Test)", X_tr_B.shape[1] == X_te_B.shape[1], f"Both have {X_tr_B.shape[1]} cols")
    check_condition("Ver C Features Match (Train=Test)", X_tr_C.shape[1] == X_te_C.shape[1], f"Both have {X_tr_C.shape[1]} cols")
    
    # Rows must match exactly with Labels
    check_condition("Ver A Labels Match (X=y rows)", X_tr_A.shape[0] == y_tr_A.shape[0], f"{X_tr_A.shape[0]} rows aligned")
    check_condition("Ver B Labels Match (X=y rows)", X_tr_B.shape[0] == y_tr_B.shape[0], f"{X_tr_B.shape[0]} rows aligned")
    check_condition("Ver C Labels Match (X=y rows)", X_tr_C.shape[0] == y_tr_C.shape[0], f"{X_tr_C.shape[0]} rows aligned")
    
    # Test set rows must match universal y_test
    check_condition("Test Labels Match (X=y rows)", X_te_A.shape[0] == y_test.shape[0], f"{X_te_A.shape[0]} rows aligned")

    # ---------------------------------------------------------
    # 3. DATA QUALITY (NaNs / Infs)
    # ---------------------------------------------------------
    print("\n--- 3. DATA QUALITY (CORRUPTION CHECK) ---")
    def has_corruption(arr):
        arr = arr.astype(np.float32)
        return np.isnan(arr).any() or np.isinf(arr).any()
        
    check_condition("Ver A Data Clean", not has_corruption(X_tr_A) and not has_corruption(X_te_A), "No NaNs/Infs detected")
    check_condition("Ver B Data Clean", not has_corruption(X_tr_B) and not has_corruption(X_te_B), "No NaNs/Infs detected")
    check_condition("Ver C Data Clean", not has_corruption(X_tr_C) and not has_corruption(X_te_C), "No NaNs/Infs detected")

    # ---------------------------------------------------------
    # 4. TRAIN/TEST LEAKAGE
    # ---------------------------------------------------------
    print("\n--- 4. DATA LEAKAGE (RIGOROUS CHECK) ---")
    def get_duplicates(tr, te):
        tr_set = set(map(tuple, tr.astype(np.float32)))
        te_set = set(map(tuple, te.astype(np.float32)))
        return len(tr_set.intersection(te_set))
    
    dup_A = get_duplicates(X_tr_A, X_te_A)
    dup_B = get_duplicates(X_tr_B, X_te_B)
    dup_C = get_duplicates(X_tr_C, X_te_C)
    
    check_condition("Ver A Zero Leakage", dup_A == 0, f"{dup_A} duplicate rows found")
    check_condition("Ver B Zero Leakage", dup_B == 0, f"{dup_B} duplicate rows found")
    check_condition("Ver C Zero Leakage", dup_C == 0, f"{dup_C} duplicate rows found")

    # ---------------------------------------------------------
    # 5. RELATIVE FEATURE SHAPES
    # ---------------------------------------------------------
    print("\n--- 5. RELATIVE VERSION SHAPES ---")
    f_A = X_tr_A.shape[1]
    f_B = X_tr_B.shape[1]
    f_C = X_tr_C.shape[1]
    check_condition("Version A < Version B", f_A < f_B, f"A:{f_A} < B:{f_B}")
    check_condition("Version B < Version C", f_B < f_C, f"B:{f_B} < C:{f_C}")
    check_condition("Version C - A = exactly 3", (f_C - f_A) == 3, f"{f_C} - {f_A} = {f_C - f_A}")

    # ---------------------------------------------------------
    # 6. LABEL INTEGRITY (Binary Classification)
    # ---------------------------------------------------------
    print("\n--- 6. TARGET LABEL INTEGRITY ---")
    def check_labels(y):
        return set(np.unique(y)).issubset({0, 1})

    check_condition("y_train_A is Binary", check_labels(y_tr_A), "Only 0s and 1s")
    check_condition("y_train_B is Binary", check_labels(y_tr_B), "Only 0s and 1s")
    check_condition("y_train_C is Binary", check_labels(y_tr_C), "Only 0s and 1s")
    check_condition("y_test is Binary",    check_labels(y_test), "Only 0s and 1s")

    # ---------------------------------------------------------
    # FINAL VERDICT
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("                    FINAL VERDICT                    ")
    print("=" * 65)
    
    all_passed = all(status for name, status in checks)
    if all_passed:
        print("🟢 ALL SYSTEMS GO! Data matrices are mathematically flawless.")
        print("🟢 Proceed immediately to model training (colab_05_model_training.py).")
    else:
        print("🔴 ABORT TRAINING! One or more integrity checks failed.")
        print("🔴 Review the logs above to identify the mismatch.")

if __name__ == "__main__":
    main()
