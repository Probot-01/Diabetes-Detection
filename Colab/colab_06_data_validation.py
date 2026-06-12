import numpy as np
import pandas as pd

RULER = "=" * 60

def main():
    print(RULER)
    print("🩺 PRE-TRAINING DATA VALIDATION PROTOCOL")
    print(RULER)
    
    # Track the success of each step to build the final report
    checks = {
        "Forbidden columns check": True,
        "Missing / NaN values": True,
        "Label integrity": True,
        "SMOTE balance check": True,
        "Feature scaling check": True,
        "Version shape logic": True,
        "Train/test leakage check": True,
        "Zero variance features": True
    }
    
    # ---------------------------------------------------------
    # STEP 1 - Load all files
    # ---------------------------------------------------------
    print("\nSTEP 1 - Loading all files")
    try:
        # Load datasets as floats to easily check for NaNs/Infs
        X_train_A = np.load('X_train_A.npy', allow_pickle=True).astype(np.float32)
        X_test_A  = np.load('X_test_A.npy', allow_pickle=True).astype(np.float32)
        
        X_train_B = np.load('X_train_B.npy', allow_pickle=True).astype(np.float32)
        X_test_B  = np.load('X_test_B.npy', allow_pickle=True).astype(np.float32)
        
        X_train_C = np.load('X_train.npy', allow_pickle=True).astype(np.float32)
        X_test_C  = np.load('X_test.npy', allow_pickle=True).astype(np.float32)
        
        y_train = np.load('y_train.npy', allow_pickle=True)
        y_test  = np.load('y_test.npy', allow_pickle=True)
        
        # Load feature names
        df_feats = pd.read_csv('feature_names.csv')
        if 'feature_name' in df_feats.columns:
            features_C = df_feats['feature_name'].tolist()
        else:
            df_feats = pd.read_csv('feature_names.csv', header=None)
            features_C = df_feats.iloc[:, 0].tolist()
            
        print(f"X_train_A: {X_train_A.shape}, X_test_A: {X_test_A.shape}")
        print(f"X_train_B: {X_train_B.shape}, X_test_B: {X_test_B.shape}")
        print(f"X_train_C: {X_train_C.shape}, X_test_C: {X_test_C.shape}")
        print(f"y_train  : {y_train.shape}, y_test  : {y_test.shape}")
    except Exception as e:
        print(f"Failed to load files: {e}")
        return

    # Derive feature lists for A and B using exactly the logic used to create them
    def drop_feats_like_before(feats, to_drop):
        res = []
        for f in feats:
            if not any(d.lower() in str(f).lower() for d in to_drop):
                res.append(f)
        return res
        
    features_A = drop_feats_like_before(features_C, ['age', 'bmi', 'gender', 'sex'])
    features_B = drop_feats_like_before(features_C, ['age'])

    # ---------------------------------------------------------
    # STEP 2 - Forbidden columns
    # ---------------------------------------------------------
    # WHY THIS MATTERS: A forbidden column (like 'BSL') teaches the model the answer directly 
    # (data leakage). 'age' is a confounding variable we explicitly want to exclude from A & B.
    print("\nSTEP 2 - Checking for forbidden columns")
    def check_forbidden(feats, name):
        forbidden_exact = ['bsl', 'label', 'age']
        forbidden_sub = ['is_synthetic', 'blood_sugar']
        
        found = []
        for f in feats:
            f_lower = str(f).lower()
            if f_lower in forbidden_exact or any(bad in f_lower for bad in forbidden_sub):
                found.append(f)
                
        if found:
            print(f"[{name}] FAIL: found {found}")
            return False
        print(f"[{name}] PASS: no forbidden columns found.")
        return True

    pass_A = check_forbidden(features_A, "Version A")
    pass_B = check_forbidden(features_B, "Version B")
    pass_C = check_forbidden(features_C, "Version C")  # This will likely fail due to 'Age', expected.
    
    if not (pass_A and pass_B and pass_C):
        checks["Forbidden columns check"] = False

    # ---------------------------------------------------------
    # STEP 3 - Missing / Corrupt Values
    # ---------------------------------------------------------
    # WHY THIS MATTERS: Machine learning models crash immediately if they encounter NaNs or Infinity. 
    # All-zero columns provide absolutely zero useful patterns.
    print("\nSTEP 3 - Checking for missing or corrupt values")
    def check_corrupt(arr, name):
        nans = np.isnan(arr).sum()
        infs = np.isinf(arr).sum()
        all_zeros = np.sum(np.all(arr == 0, axis=0))
        if nans > 0 or infs > 0 or all_zeros > 0:
            print(f"[{name}] FAIL: {nans} NaNs, {infs} Infs, {all_zeros} all-zero columns.")
            return False
        print(f"[{name}] PASS: No missing or corrupt values.")
        return True

    results_3 = [
        check_corrupt(X_train_A, "X_train_A"), check_corrupt(X_test_A, "X_test_A"),
        check_corrupt(X_train_B, "X_train_B"), check_corrupt(X_test_B, "X_test_B"),
        check_corrupt(X_train_C, "X_train_C"), check_corrupt(X_test_C, "X_test_C")
    ]
    if not all(results_3):
        checks["Missing / NaN values"] = False

    # ---------------------------------------------------------
    # STEP 4 - Label Integrity
    # ---------------------------------------------------------
    # WHY THIS MATTERS: Binary classification targets MUST be exactly two classes: 0 and 1.
    print("\nSTEP 4 - Checking label integrity")
    def check_labels(y, name):
        u = np.unique(y)
        if set(u).issubset({0, 1}):
            print(f"[{name}] PASS: contains only {u}")
            return True
        print(f"[{name}] FAIL: invalid values {u}")
        return False
        
    l1 = check_labels(y_train, "y_train")
    l2 = check_labels(y_test, "y_test")
    if not (l1 and l2):
        checks["Label integrity"] = False

    # ---------------------------------------------------------
    # STEP 5 - SMOTE Balance Check
    # ---------------------------------------------------------
    # WHY THIS MATTERS: If y_train isn't 50/50, the model will just guess the majority class. 
    # If y_test isn't 75/25, our final evaluation is medically unrealistic.
    print("\nSTEP 5 - Checking SMOTE application")
    _, counts_tr = np.unique(y_train, return_counts=True)
    if len(counts_tr) == 2 and (0.45 <= counts_tr[0]/sum(counts_tr) <= 0.55):
        print(f"[y_train] PASS: roughly 50/50 balance (Class 0: {counts_tr[0]}, Class 1: {counts_tr[1]})")
        s1 = True
    else:
        print(f"[y_train] FAIL: Not balanced. Counts: {counts_tr}")
        s1 = False

    _, counts_te = np.unique(y_test, return_counts=True)
    if len(counts_te) == 2:
        ratio = max(counts_te) / sum(counts_te)
        if 0.65 <= ratio <= 0.85:
            print(f"[y_test]  PASS: roughly 75/25 original balance (Ratio max class {ratio:.2f})")
            s2 = True
        else:
            print(f"[y_test]  FAIL: Ratio is {ratio:.2f}, not original distribution.")
            s2 = False
    else:
        print("[y_test] FAIL: Not 2 classes.")
        s2 = False
        
    if not (s1 and s2):
        checks["SMOTE balance check"] = False

    # ---------------------------------------------------------
    # STEP 6 - Feature Scaling
    # ---------------------------------------------------------
    # WHY THIS MATTERS: Scaling (mean=0, std=1) allows models to weigh features equally instead 
    # of prioritizing columns with naturally larger numerical values.
    print("\nSTEP 6 - Checking feature scaling")
    def check_scaling(arr, name):
        m = np.mean(arr)
        s = np.std(arr)
        if abs(m) < 0.1 and abs(s - 1.0) < 0.1:
            print(f"[{name}] PASS: Mean = {m:.4f}, Std = {s:.4f}")
            return True
        print(f"[{name}] FAIL: Mean = {m:.4f}, Std = {s:.4f}")
        return False

    sc1 = check_scaling(X_train_A, "X_train_A")
    sc2 = check_scaling(X_train_B, "X_train_B")
    sc3 = check_scaling(X_train_C, "X_train_C")
    if not (sc1 and sc2 and sc3):
        checks["Feature scaling check"] = False

    # ---------------------------------------------------------
    # STEP 7 - Version Shape Logic
    # ---------------------------------------------------------
    # WHY THIS MATTERS: Ensures our "create_versions.py" script accurately stripped out metadata correctly.
    print("\nSTEP 7 - Checking version shapes")
    sh1 = X_train_A.shape[1] < X_train_B.shape[1]
    sh2 = X_train_B.shape[1] < X_train_C.shape[1]
    sh3 = X_train_C.shape[1] == X_train_A.shape[1] + 3
    
    if sh1 and sh2 and sh3:
        print("PASS: Version A < B < C, and C has exactly 3 more features than A.")
    else:
        print(f"FAIL: Logic broken. A:{X_train_A.shape[1]}, B:{X_train_B.shape[1]}, C:{X_train_C.shape[1]}")
        checks["Version shape logic"] = False

    # ---------------------------------------------------------
    # STEP 8 - Data Leakage Check
    # ---------------------------------------------------------
    # WHY THIS MATTERS: If any row from X_test was accidentally mixed into X_train, 
    # the model "memorizes" the test data instead of actually learning, invalidating evaluation scores.
    print("\nSTEP 8 - Checking train/test data leakage")
    def check_leakage(tr, te, name):
        # Convert rows into tuples so they can be hashed and checked in a set instantly
        tr_set = set(map(tuple, tr))
        te_set = set(map(tuple, te))
        intersect = tr_set.intersection(te_set)
        count = len(intersect)
        if count == 0:
            print(f"[{name}] PASS: 0 duplicate rows found.")
            return True
        print(f"[{name}] FAIL: {count} identical rows found in both sets (Leakage!).")
        return False
        
    lk1 = check_leakage(X_train_A, X_test_A, "Version A")
    lk2 = check_leakage(X_train_B, X_test_B, "Version B")
    lk3 = check_leakage(X_train_C, X_test_C, "Version C")
    if not (lk1 and lk2 and lk3):
        checks["Train/test leakage check"] = False

    # ---------------------------------------------------------
    # STEP 9 - Zero Variance Features
    # ---------------------------------------------------------
    # WHY THIS MATTERS: A feature column filled with entirely the same number (variance = 0) 
    # adds no discriminatory power to the model and just wastes memory/compute.
    print("\nSTEP 9 - Checking for zero-variance features")
    def check_variance(arr, feats, name):
        stds = np.std(arr, axis=0)
        zeros = np.where(stds == 0)[0]
        if len(zeros) == 0:
            print(f"[{name}] PASS: No zero-variance features.")
            return True
        bad_names = [feats[i] for i in zeros if i < len(feats)]
        print(f"[{name}] WARN: Zero-variance found in: {bad_names}")
        return False
        
    v1 = check_variance(X_train_A, features_A, "Version A")
    v2 = check_variance(X_train_B, features_B, "Version B")
    v3 = check_variance(X_train_C, features_C, "Version C")
    if not (v1 and v2 and v3):
        checks["Zero variance features"] = False

    # ---------------------------------------------------------
    # STEP 10 - Final Report
    # ---------------------------------------------------------
    print("\nSTEP 10 - Final validation report")
    print("╔" + "═"*50 + "╗")
    print("║" + "PRE-TRAINING VALIDATION REPORT".center(50) + "║")
    print("╠" + "═"*50 + "╣")
    
    for key, passed in checks.items():
        icon = "✅" if passed else "❌"
        row_content = f" {key:<26} : {icon} "
        # 50 total internal width minus length of our string ensures right border aligns perfectly
        padding = 50 - len(row_content)
        print("║" + row_content + " "*padding + "║")
        
    print("╚" + "═"*50 + "╝\n")
    
    if all(checks.values()):
        print("✅ DATA IS CLEAN. SAFE TO TRAIN.")
    else:
        print("❌ PROBLEMS FOUND. DO NOT TRAIN YET. FIX ABOVE ISSUES.")

if __name__ == "__main__":
    main()
