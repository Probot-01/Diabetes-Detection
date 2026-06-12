import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score, matthews_corrcoef
import os

def main():
    print("=" * 60)
    print("🛡️  PRE-DEPLOYMENT MODEL RIGOROUS TESTING")
    print("=" * 60)
    
    # ---------------------------------------------------------
    # LOAD ASSETS
    # ---------------------------------------------------------
    # The models were saved in the root directory in colab_12, 
    # but the user might have moved them to voice_diabetes_app/models.
    # We will safely check both paths.
    def get_path(filename):
        if os.path.exists(filename): return filename
        alt_path = os.path.join("voice_diabetes_app", "models", filename)
        if os.path.exists(alt_path): return alt_path
        return filename

    try:
        model1 = joblib.load(get_path('model1_voice_only.pkl'))
        model2 = joblib.load(get_path('model2_voice_demo.pkl'))
        scaler_A = joblib.load(get_path('scaler_A.pkl'))
        scaler_B = joblib.load(get_path('scaler_B.pkl'))
        
        # Test arrays from the original colab directory
        X_test_A = np.load('X_test_A.npy', allow_pickle=True).astype(np.float32)
        X_test_B = np.load('X_test_B.npy', allow_pickle=True).astype(np.float32)
        y_test = np.load('y_test.npy', allow_pickle=True).astype(int)
        
    except Exception as e:
        print(f"Failed to load assets: {e}")
        return

    # ---------------------------------------------------------
    # TEST 1 - Basic sanity check
    # ---------------------------------------------------------
    print("\n--- TEST 1: Basic Sanity Check ---")
    try:
        pred_1 = model1.predict(X_test_A)
        prob_1 = model1.predict_proba(X_test_A)
        pred_2 = model2.predict(X_test_B)
        prob_2 = model2.predict_proba(X_test_B)
        
        print(f"Model 1 predict_proba shape: {prob_1.shape}")
        print(f"Model 2 predict_proba shape: {prob_2.shape}")
        print("✅ PASS")
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return
        
    # Helper for TEST 2 and 3
    def evaluate(model, X, y_true):
        prob = model.predict_proba(X)[:, 1]
        pred = model.predict(X)
        
        acc = accuracy_score(y_true, pred)
        auc = roc_auc_score(y_true, prob)
        f1 = f1_score(y_true, pred)
        mcc = matthews_corrcoef(y_true, pred)
        
        cm = confusion_matrix(y_true, pred)
        tn, fp, fn, tp = cm.ravel()
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return prob, pred, acc, auc, f1, mcc, cm, tn, fp, fn, tp, sens, spec

    p1_prob, p1_pred, a1, auc1, f1_1, mcc1, cm1, tn1, fp1, fn1, tp1, sens1, spec1 = evaluate(model1, X_test_A, y_test)
    p2_prob, p2_pred, a2, auc2, f1_2, mcc2, cm2, tn2, fp2, fn2, tp2, sens2, spec2 = evaluate(model2, X_test_B, y_test)

    # ---------------------------------------------------------
    # TEST 2 - Performance metrics
    # ---------------------------------------------------------
    print("\n--- TEST 2: Performance Metrics ---")
    print(f"Model 1: Acc={a1:.3f}, AUC={auc1:.3f}, Sens={sens1:.3f}, Spec={spec1:.3f}, F1={f1_1:.3f}, MCC={mcc1:.3f}")
    if auc1 > 0.75: print("Model 1: ✅ PASS") 
    else: print("Model 1: ❌ FAIL")
    
    print(f"Model 2: Acc={a2:.3f}, AUC={auc2:.3f}, Sens={sens2:.3f}, Spec={spec2:.3f}, F1={f1_2:.3f}, MCC={mcc2:.3f}")
    if auc2 > 0.75: print("Model 2: ✅ PASS") 
    else: print("Model 2: ❌ FAIL")

    # ---------------------------------------------------------
    # TEST 3 - Confusion matrix
    # ---------------------------------------------------------
    print("\n--- TEST 3: Confusion Matrix ---")
    def print_cm(name, cm, tn, fp, fn, tp):
        print(f"\n{name}:")
        print("              Predicted")
        print("              NEG    POS")
        print(f"Actual NEG   [{tn:^4}] [{fp:^4}]")
        print(f"Actual POS   [{fn:^4}] [{tp:^4}]")
        print(f"  True Positives  (caught diabetics)    : {tp}")
        print(f"  False Negatives (missed diabetics)    : {fn}")
        print(f"  False Positives (wrong alarms)        : {fp}")
        print(f"  True Negatives  (correct healthy)     : {tn}")
        
        if (fn + tp) > 0 and fn / (fn + tp) > 0.20:
            print(f"  ⚠️ WARNING: False Negatives exceed 20% of actual positives! ({fn / (fn + tp)*100:.1f}% missed)")

    print_cm("Model 1", cm1, tn1, fp1, fn1, tp1)
    print_cm("Model 2", cm2, tn2, fp2, fn2, tp2)

    # ---------------------------------------------------------
    # TEST 4 - Probability distribution check
    # ---------------------------------------------------------
    print("\n--- TEST 4: Probability Distribution Check ---")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(p1_prob[y_test == 0], bins=20, alpha=0.5, label='Actual NEG', color='green')
    plt.hist(p1_prob[y_test == 1], bins=20, alpha=0.5, label='Actual POS', color='red')
    plt.title('Model 1: Probability Distribution')
    plt.xlabel('Predicted Probability of Diabetes')
    plt.ylabel('Count')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(p2_prob[y_test == 0], bins=20, alpha=0.5, label='Actual NEG', color='green')
    plt.hist(p2_prob[y_test == 1], bins=20, alpha=0.5, label='Actual POS', color='red')
    plt.title('Model 2: Probability Distribution')
    plt.xlabel('Predicted Probability of Diabetes')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("model_prob_distribution.png")
    print("Saved distribution plot to model_prob_distribution.png")

    # ---------------------------------------------------------
    # TEST 5 - Threshold analysis
    # ---------------------------------------------------------
    print("\n--- TEST 5: Threshold Analysis ---")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    def analyze_thresholds(name, prob, y_true):
        print(f"\n{name} Threshold Analysis:")
        best_f1 = -1
        best_t = 0.5
        for t in thresholds:
            t_pred = (prob >= t).astype(int)
            t_cm = confusion_matrix(y_true, t_pred)
            if t_cm.size == 4:
                t_tn, t_fp, t_fn, t_tp = t_cm.ravel()
            else:
                t_tn, t_fp, t_fn, t_tp = 0,0,0,0
            t_sens = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0
            t_spec = t_tn / (t_tn + t_fp) if (t_tn + t_fp) > 0 else 0
            t_f1 = f1_score(y_true, t_pred)
            if t_f1 > best_f1:
                best_f1 = t_f1
                best_t = t
            print(f"  Thresh {t:.1f} | Sens: {t_sens:.3f} | Spec: {t_spec:.3f} | F1: {t_f1:.3f}")
        return best_t

    best_t_1 = analyze_thresholds("Model 1", p1_prob, y_test)
    best_t_2 = analyze_thresholds("Model 2", p2_prob, y_test)

    # ---------------------------------------------------------
    # TEST 6 - Edge case tests
    # ---------------------------------------------------------
    print("\n--- TEST 6: Edge Case Tests ---")
    def analyze_edges(name, prob, y_true):
        print(f"\n{name} Edge Cases:")
        sorted_idx = np.argsort(prob)
        
        lowest = sorted_idx[:5]
        print("  5 Most Confident NON-DIABETIC:")
        for i in lowest: print(f"    Sample {i:<4} | Prob: {prob[i]:.4f} | Actual: {y_true[i]}")
            
        highest = sorted_idx[-5:]
        print("  5 Most Confident DIABETIC:")
        for i in reversed(highest): print(f"    Sample {i:<4} | Prob: {prob[i]:.4f} | Actual: {y_true[i]}")
            
        dist_to_half = np.abs(prob - 0.5)
        uncertain = np.argsort(dist_to_half)[:5]
        print("  5 Most UNCERTAIN:")
        for i in uncertain: print(f"    Sample {i:<4} | Prob: {prob[i]:.4f} | Actual: {y_true[i]}")

    analyze_edges("Model 1", p1_prob, y_test)
    analyze_edges("Model 2", p2_prob, y_test)

    # ---------------------------------------------------------
    # TEST 7 - Final report
    # ---------------------------------------------------------
    print("\n--- TEST 7: Final Report ---")
    print("╔══════════════════════════════════════════════╗")
    print("║           MODEL TEST REPORT                  ║")
    print("╠══════════════════════════════════════════════╣")
    print("║ MODEL 1 — Voice Only                         ║")
    print(f"║   AUC-ROC       : {auc1:<27.4f}║")
    print(f"║   Sensitivity   : {sens1:<27.4f}║")
    print(f"║   Best threshold: {best_t_1:<27.1f}║")
    print(f"║   Missed diabetics (FN): {fn1:<20}║")
    status1 = "✅ PASS" if auc1 > 0.75 else "❌ FAIL"
    print(f"║   Status        : {status1:<27}║")
    print("╠══════════════════════════════════════════════╣")
    print("║ MODEL 2 — Voice + BMI/Gender                 ║")
    print(f"║   AUC-ROC       : {auc2:<27.4f}║")
    print(f"║   Sensitivity   : {sens2:<27.4f}║")
    print(f"║   Best threshold: {best_t_2:<27.1f}║")
    print(f"║   Missed diabetics (FN): {fn2:<20}║")
    status2 = "✅ PASS" if auc2 > 0.75 else "❌ FAIL"
    print(f"║   Status        : {status2:<27}║")
    print("╚══════════════════════════════════════════════╝\n")

    if "✅ PASS" in status1 and "✅ PASS" in status2:
        print("✅ MODELS VERIFIED. SAFE FOR DEPLOYMENT.")
    else:
        print("❌ DO NOT DEPLOY. REVIEW ABOVE ISSUES.")

if __name__ == "__main__":
    main()
