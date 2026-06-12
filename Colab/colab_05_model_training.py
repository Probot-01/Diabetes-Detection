import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
import joblib

def main():
    print("STEP 1 - Load all data:")
    
    # Load feature versions and their specific cleaned labels
    X_train_A = np.load('X_train_A.npy', allow_pickle=True).astype(np.float32)
    X_test_A  = np.load('X_test_A.npy', allow_pickle=True).astype(np.float32)
    y_train_A = np.load('y_train_A.npy', allow_pickle=True)
    
    X_train_B = np.load('X_train_B.npy', allow_pickle=True).astype(np.float32)
    X_test_B  = np.load('X_test_B.npy', allow_pickle=True).astype(np.float32)
    y_train_B = np.load('y_train_B.npy', allow_pickle=True)
    
    X_train_C = np.load('X_train.npy', allow_pickle=True).astype(np.float32)
    X_test_C  = np.load('X_test.npy', allow_pickle=True).astype(np.float32)
    y_train_C = np.load('y_train_C.npy', allow_pickle=True)
    
    # y_test is identical for all versions since test rows were unmodified
    y_test = np.load('y_test.npy', allow_pickle=True)
    
    print("Data loaded successfully.\n")
    
    def train_and_evaluate(X_train, X_test, y_train, y_test, version_name):
        # STEP 2 - Define models
        models = {
            "RF": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
            "XGB": XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=3, eval_metric='logloss', use_label_encoder=False),
            "SVM": SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            print(f"--- Training {name} ---")
            # Train the model
            model.fit(X_train, y_train)
            
            # Predict and calculate probabilities
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test)
                
            # Basic Metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            f1 = f1_score(y_test, y_pred)
            
            # Confusion Matrix components
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Sensitivity = out of all real diabetic patients, how many did we correctly catch? 
            # Higher is better for medical screening. It tells us our true positive rate.
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Specificity = out of all real non-diabetic patients, how many were correctly cleared?
            # It tells us our true negative rate.
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            print(f"Accuracy    : {acc:.4f}")
            print(f"AUC-ROC     : {auc:.4f}")
            print(f"Sensitivity : {sensitivity:.4f}")
            print(f"Specificity : {specificity:.4f}")
            print(f"F1 Score    : {f1:.4f}")
            
            print("Confusion Matrix:")
            print(f"[{tn:3d} {fp:3d}]")
            print(f"[{fn:3d} {tp:3d}]\n")
            
            results[name] = {
                "model_obj": model,
                "AUC-ROC": auc,
                "Sensitivity": sensitivity,
                "F1": f1
            }
        return results

    # STEP 3 - Run train_and_evaluate on all three versions
    print("===== VERSION A: VOICE ONLY =====")
    res_A = train_and_evaluate(X_train_A, X_test_A, y_train_A, y_test, "A: Voice only")
    
    print("===== VERSION B: VOICE + BMI/GENDER =====")
    res_B = train_and_evaluate(X_train_B, X_test_B, y_train_B, y_test, "B: Voice+BMI/Sex")
    
    print("===== VERSION C: ALL FEATURES + AGE (CONFOUND) =====")
    res_C = train_and_evaluate(X_train_C, X_test_C, y_train_C, y_test, "C: All + Age ⚠️")
    
    # STEP 4 - Print a final comparison table
    print("STEP 4 - Final Comparison Table:\n")
    print("╔" + "═"*19 + "╦" + "═"*8 + "╦" + "═"*9 + "╦" + "═"*13 + "╦" + "═"*10 + "╗")
    print("║ Version           ║ Model  ║ AUC-ROC ║ Sensitivity ║    F1    ║")
    print("╠" + "═"*19 + "╬" + "═"*8 + "╬" + "═"*9 + "╬" + "═"*13 + "╬" + "═"*10 + "╣")
    
    def print_row(version, model_name, metrics):
        auc = f"{metrics['AUC-ROC']:.4f}"
        sens = f"{metrics['Sensitivity']:.4f}"
        f1 = f"{metrics['F1']:.4f}"
        
        # Exact width formatting to align with the table headers perfectly
        c1 = f" {version:<18}"
        c2 = f" {model_name:<7}"
        c3 = f"  {auc:<7}"
        c4 = f"    {sens:<9}"
        c5 = f"   {f1:<7}"
        
        print(f"║{c1}║{c2}║{c3}║{c4}║{c5}║")

    for m in ["RF", "XGB", "SVM"]:
        print_row("A: Voice only", m, res_A[m])
    for m in ["RF", "XGB", "SVM"]:
        print_row("B: Voice+BMI/Sex", m, res_B[m])
    for m in ["RF", "XGB", "SVM"]:
        print_row("C: All + Age ⚠️", m, res_C[m])
        
    print("╚" + "═"*19 + "╩" + "═"*8 + "╩" + "═"*9 + "╩" + "═"*13 + "╩" + "═"*10 + "╝\n")

    # STEP 5 - Save the best model
    print("STEP 5 - Save the best model:")
    
    best_auc = -1
    best_model_info = None
    
    # ONLY consider Version A and Version B (not C) for the final model
    candidates = [
        ("A", "RF", res_A["RF"]), ("A", "XGB", res_A["XGB"]), ("A", "SVM", res_A["SVM"]),
        ("B", "RF", res_B["RF"]), ("B", "XGB", res_B["XGB"]), ("B", "SVM", res_B["SVM"])
    ]
    
    for ver, m_name, metrics in candidates:
        if metrics["AUC-ROC"] > best_auc:
            best_auc = metrics["AUC-ROC"]
            best_model_info = (ver, m_name, metrics["model_obj"])
            
    best_ver, best_m_name, best_model_obj = best_model_info
    
    joblib.dump(best_model_obj, "best_model.pkl")
    print(f"Best model: Version {best_ver} + {best_m_name} with AUC = {best_auc:.4f}")

if __name__ == "__main__":
    main()
