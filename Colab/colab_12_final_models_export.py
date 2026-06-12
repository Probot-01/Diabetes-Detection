import numpy as np
import pandas as pd
import joblib
import copy
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score

def main():
    print("=" * 60)
    print("🚀 EXPORTING FINAL PRODUCTION MODELS")
    print("=" * 60)

    # ---------------------------------------------------------
    # 0. Load Base Assets (Scaler & Features)
    # ---------------------------------------------------------
    try:
        y_test = np.load('y_test.npy', allow_pickle=True).astype(int)
        
        # Load the universal scaler (fitted on 270+ features)
        base_scaler = joblib.load('scaler.pkl')
        
        # Load universal feature names
        df_feats = pd.read_csv('feature_names.csv')
        if 'feature_name' in df_feats.columns:
            features = df_feats['feature_name'].tolist()
        else:
            df_feats = pd.read_csv('feature_names.csv', header=None)
            features = df_feats.iloc[:, 0].tolist()
            
        features = [f for f in features if str(f).strip().lower() not in [
            'feature_name', '0', 'unnamed: 0', 'is_synthetic', '', 'nan'
        ]]
        
        # Align feature count dynamically based on the original base matrix
        target_size = np.load('X_train.npy', allow_pickle=True).shape[1]
        while len(features) > target_size:
            features = features[1:]
            
    except Exception as e:
        print(f"Error loading base files: {e}")
        return

    # Calculate exact indices that were dropped in Version A and B
    drop_idx_A = [i for i, f in enumerate(features) if any(d in str(f).lower() for d in ['age', 'bmi', 'gender', 'sex'])]
    drop_idx_B = [i for i, f in enumerate(features) if 'age(in years)' in str(f).lower() or str(f).lower() == 'age' or 'age(' in str(f).lower() or str(f).lower().startswith('age_') or str(f).lower().endswith('_age')]

    # ---------------------------------------------------------
    # 1. MODEL 1 — Voice Only (Version A)
    # ---------------------------------------------------------
    print("\n--- Training MODEL 1 (Voice Only) ---")
    X_train_A = np.load('X_train_A.npy', allow_pickle=True).astype(np.float32)
    y_train_A = np.load('y_train_A.npy', allow_pickle=True).astype(int)
    X_test_A  = np.load('X_test_A.npy', allow_pickle=True).astype(np.float32)
    
    # Train
    model_A = XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=3, eval_metric='logloss', use_label_encoder=False)
    model_A.fit(X_train_A, y_train_A)
    joblib.dump(model_A, "model1_voice_only.pkl")
    
    # Construct exact Scaler for A
    # Slicing the internal metric arrays so the scaler accepts the smaller footprint
    scaler_A = copy.deepcopy(base_scaler)
    if hasattr(scaler_A, 'mean_'):
        scaler_A.mean_ = np.delete(scaler_A.mean_, drop_idx_A)
        scaler_A.scale_ = np.delete(scaler_A.scale_, drop_idx_A)
        scaler_A.var_ = np.delete(scaler_A.var_, drop_idx_A)
    scaler_A.n_features_in_ = X_train_A.shape[1]
    joblib.dump(scaler_A, "scaler_A.pkl")
    
    # Construct exact Feature List for A
    features_A = [f for i, f in enumerate(features) if i not in drop_idx_A]
    if len(features_A) > X_train_A.shape[1]: features_A = features_A[:X_train_A.shape[1]]
    pd.DataFrame(features_A, columns=['feature_name']).to_csv('feature_names_A.csv', index=False)
    
    print("Model 1 artifacts saved: model1_voice_only.pkl | scaler_A.pkl | feature_names_A.csv")

    # ---------------------------------------------------------
    # 2. MODEL 2 — Voice + Gender + BMI (Version B)
    # ---------------------------------------------------------
    print("\n--- Training MODEL 2 (Voice + Demo) ---")
    X_train_B = np.load('X_train_B.npy', allow_pickle=True).astype(np.float32)
    y_train_B = np.load('y_train_B.npy', allow_pickle=True).astype(int)
    X_test_B  = np.load('X_test_B.npy', allow_pickle=True).astype(np.float32)
    
    # Train
    model_B = XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=3, eval_metric='logloss', use_label_encoder=False)
    model_B.fit(X_train_B, y_train_B)
    joblib.dump(model_B, "model2_voice_demo.pkl")
    
    # Construct exact Scaler for B
    scaler_B = copy.deepcopy(base_scaler)
    if hasattr(scaler_B, 'mean_'):
        scaler_B.mean_ = np.delete(scaler_B.mean_, drop_idx_B)
        scaler_B.scale_ = np.delete(scaler_B.scale_, drop_idx_B)
        scaler_B.var_ = np.delete(scaler_B.var_, drop_idx_B)
    scaler_B.n_features_in_ = X_train_B.shape[1]
    joblib.dump(scaler_B, "scaler_B.pkl")
    
    # Construct exact Feature List for B
    features_B = [f for i, f in enumerate(features) if i not in drop_idx_B]
    if len(features_B) > X_train_B.shape[1]: features_B = features_B[:X_train_B.shape[1]]
    pd.DataFrame(features_B, columns=['feature_name']).to_csv('feature_names_B.csv', index=False)

    print("Model 2 artifacts saved: model2_voice_demo.pkl | scaler_B.pkl | feature_names_B.csv")

    # ---------------------------------------------------------
    # 3. VERIFICATION & SUMMARY
    # ---------------------------------------------------------
    def get_metrics(model, X):
        y_pred = model.predict(X)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
        else:
            y_prob = model.decision_function(X)
        
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        return auc, sens, f1
        
    auc_A, sens_A, f1_A = get_metrics(model_A, X_test_A)
    auc_B, sens_B, f1_B = get_metrics(model_B, X_test_B)
    
    print("\n╔══════════════════════════════════════════════╗")
    print("║              FINAL MODEL SUMMARY             ║")
    print("╠══════════════════════════════════════════════╣")
    print("║ Model 1 — Voice Only                         ║")
    print(f"║   AUC-ROC     : {auc_A:<29.4f}║")
    print(f"║   Sensitivity : {sens_A:<29.4f}║")
    print(f"║   F1-Score    : {f1_A:<29.4f}║")
    print(f"║   Input features : {X_train_A.shape[1]:<26}║")
    print("║   Saved as    : model1_voice_only.pkl        ║")
    print("╠══════════════════════════════════════════════╣")
    print("║ Model 2 — Voice + Gender + BMI               ║")
    print(f"║   AUC-ROC     : {auc_B:<29.4f}║")
    print(f"║   Sensitivity : {sens_B:<29.4f}║")
    print(f"║   F1-Score    : {f1_B:<29.4f}║")
    print(f"║   Input features : {X_train_B.shape[1]:<26}║")
    print("║   Saved as    : model2_voice_demo.pkl        ║")
    print("╚══════════════════════════════════════════════╝\n")
    print("Both models ready for deployment.")

if __name__ == "__main__":
    main()
