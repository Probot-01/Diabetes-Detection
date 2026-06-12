import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, recall_score
import shap
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("🚀 TRAINING NEW MODELS (3, 4, 5M, 5F)")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. LOAD ASSETS
    # ---------------------------------------------------------
    try:
        df = pd.read_csv('dataset.csv')
        
        X_train_A = np.load('X_train_A.npy')
        y_train_A = np.load('y_train_A.npy')
        X_test_A = np.load('X_test_A.npy')
        y_test = np.load('y_test.npy')
        
        # We also load X_train_B. 
        # Why? Because X_train_A is already SMOTE-balanced (e.g. 1500 rows). 
        # The original training split only has 800 rows. We cannot append an 800-row BMI column to a 1500-row Voice array.
        # X_train_B contains the exact same 1500 SMOTE-balanced rows, but includes the scaled BMI and Gender columns.
        X_train_B = np.load('X_train_B.npy')
        X_test_B = np.load('X_test_B.npy')
        scaler_B = joblib.load('scaler_B.pkl')
        
        feats_A = pd.read_csv('feature_names_A.csv').iloc[:, 0].tolist()
        feats_B = pd.read_csv('feature_names_B.csv').iloc[:, 0].tolist()
        
    except Exception as e:
        print(f"File load error: {e}")
        return

    # Extract the BMI scaler logic explicitly
    scaler_bmi = StandardScaler()
    scaler_bmi.mean_ = np.array([scaler_B.mean_[-1]])
    scaler_bmi.scale_ = np.array([scaler_B.scale_[-1]])
    scaler_bmi.var_ = np.array([scaler_B.var_[-1]])
    scaler_bmi.n_features_in_ = 1
    joblib.dump(scaler_bmi, 'scaler_bmi.pkl')
    print("✅ Saved scaler_bmi.pkl")

    # ---------------------------------------------------------
    # MODEL 3 — Voice + BMI only
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("MODEL 3 — Voice + BMI only (no gender)")
    print("="*40)
    
    # Drop Gender (index 267) from X_train_B to get exactly Voice + BMI on the SMOTEd dataset
    X_train_3 = np.delete(X_train_B, 267, axis=1)
    X_test_3 = np.delete(X_test_B, 267, axis=1)
    print(f"Final shape: {X_train_3.shape}")
    
    model3 = XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=3, eval_metric='logloss')
    model3.fit(X_train_3, y_train_A)
    joblib.dump(model3, 'model3_voice_bmi.pkl')
    
    auc_3 = roc_auc_score(y_test, model3.predict_proba(X_test_3)[:, 1])
    sens_3 = recall_score(y_test, model3.predict(X_test_3))
    print(f"AUC: {auc_3:.4f} | Sensitivity: {sens_3:.4f}")
    
    feats_3 = feats_B.copy()
    feats_3.pop(267)
    pd.DataFrame(feats_3, columns=['feature_name']).to_csv('feature_names_C3.csv', index=False)
    
    # ---------------------------------------------------------
    # MODEL 4 — Voice + Gender only
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("MODEL 4 — Voice + Gender only (no BMI)")
    print("="*40)
    
    # Drop BMI (index 268) from X_train_B
    X_train_4 = np.delete(X_train_B, 268, axis=1)
    X_test_4 = np.delete(X_test_B, 268, axis=1)
    print(f"Final shape: {X_train_4.shape}")
    
    model4 = XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=3, eval_metric='logloss')
    model4.fit(X_train_4, y_train_A)
    joblib.dump(model4, 'model4_voice_gender.pkl')
    
    auc_4 = roc_auc_score(y_test, model4.predict_proba(X_test_4)[:, 1])
    sens_4 = recall_score(y_test, model4.predict(X_test_4))
    print(f"AUC: {auc_4:.4f} | Sensitivity: {sens_4:.4f}")
    
    feats_4 = feats_B.copy()
    feats_4.pop(268)
    pd.DataFrame(feats_4, columns=['feature_name']).to_csv('feature_names_C4.csv', index=False)

    # ---------------------------------------------------------
    # RAW DATASET PREP FOR 5M / 5F
    # ---------------------------------------------------------
    # Dynamically find exact case-sensitive column names
    gender_col = next((c for c in df.columns if c.lower() in ['gender', 'sex']), 'Gender')
    label_col = next((c for c in df.columns if c.lower() in ['label', 'class']), 'label')
    
    print("\n" + "="*40)
    print("GENDER DISTRIBUTION (Raw Dataset)")
    print("="*40)
    
    m_val, f_val = None, None
    for g_val in df[gender_col].unique():
        sub = df[df[gender_col] == g_val]
        diabetic_rate = sub[label_col].mean()
        print(f"Gender '{g_val}': {len(sub)} rows, Diabetic Rate: {diabetic_rate*100:.1f}%")
        
        # Auto-detect which integer or string represents male/female
        if isinstance(g_val, str) and 'm' in g_val.lower() and 'f' not in g_val.lower():
            m_val = g_val
        elif isinstance(g_val, str) and 'f' in g_val.lower():
            f_val = g_val
        elif g_val == 1:
            m_val = 1
        elif g_val == 0:
            f_val = 0
            
    if m_val is None: m_val = 1
    if f_val is None: f_val = 0

    voice_cols = [c for c in df.columns if c in feats_A]

    # ---------------------------------------------------------
    # MODEL 5M — Male patients only
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("MODEL 5M — Male patients only")
    print("="*40)
    
    df_m = df[df[gender_col] == m_val]
    X_m = df_m[voice_cols].values
    y_m = df_m[label_col].values
    
    X_train_m_raw, X_test_m_raw, y_train_m, y_test_m = train_test_split(X_m, y_m, test_size=0.2, stratify=y_m, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train_m_sm, y_train_m_sm = smote.fit_resample(X_train_m_raw, y_train_m)
    
    scaler_m = StandardScaler()
    X_train_m_scaled = scaler_m.fit_transform(X_train_m_sm)
    X_test_m_scaled = scaler_m.transform(X_test_m_raw)
    joblib.dump(scaler_m, 'scaler_male.pkl')
    
    model5m = XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=3, eval_metric='logloss', use_label_encoder=False)
    model5m.fit(X_train_m_scaled, y_train_m_sm)
    joblib.dump(model5m, 'model5M_male.pkl')
    
    auc_5m = roc_auc_score(y_test_m, model5m.predict_proba(X_test_m_scaled)[:, 1])
    sens_5m = recall_score(y_test_m, model5m.predict(X_test_m_scaled))
    print(f"Male Dataset Size: {len(df_m)} | Train: {len(y_train_m_sm)} (SMOTE) | Test: {len(y_test_m)}")
    print(f"AUC: {auc_5m:.4f} | Sensitivity: {sens_5m:.4f}")

    # ---------------------------------------------------------
    # MODEL 5F — Female patients only
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("MODEL 5F — Female patients only")
    print("="*40)
    
    df_f = df[df[gender_col] == f_val]
    X_f = df_f[voice_cols].values
    y_f = df_f[label_col].values
    
    X_train_f_raw, X_test_f_raw, y_train_f, y_test_f = train_test_split(X_f, y_f, test_size=0.2, stratify=y_f, random_state=42)
    
    X_train_f_sm, y_train_f_sm = smote.fit_resample(X_train_f_raw, y_train_f)
    
    scaler_f = StandardScaler()
    X_train_f_scaled = scaler_f.fit_transform(X_train_f_sm)
    X_test_f_scaled = scaler_f.transform(X_test_f_raw)
    joblib.dump(scaler_f, 'scaler_female.pkl')
    
    model5f = XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=3, eval_metric='logloss', use_label_encoder=False)
    model5f.fit(X_train_f_scaled, y_train_f_sm)
    joblib.dump(model5f, 'model5F_female.pkl')
    
    auc_5f = roc_auc_score(y_test_f, model5f.predict_proba(X_test_f_scaled)[:, 1])
    sens_5f = recall_score(y_test_f, model5f.predict(X_test_f_scaled))
    print(f"Female Dataset Size: {len(df_f)} | Train: {len(y_train_f_sm)} (SMOTE) | Test: {len(y_test_f)}")
    print(f"AUC: {auc_5f:.4f} | Sensitivity: {sens_5f:.4f}")

    # ---------------------------------------------------------
    # SHAP ANALYSIS
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("SHAP ANALYSIS")
    print("="*40)
    
    explainer_3 = shap.TreeExplainer(model3)
    shap_vals_3 = explainer_3.shap_values(X_test_3)
    mean_abs_shap_3 = np.abs(shap_vals_3).mean(axis=0)
    
    shap_df_3 = pd.DataFrame({'Feature': feats_3, 'Importance': mean_abs_shap_3})
    shap_df_3 = shap_df_3.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    print("\nTop 10 Features for Model 3 (Voice + BMI):")
    for i in range(10):
        print(f"  {i+1}. {shap_df_3['Feature'][i]:<15} - {shap_df_3['Importance'][i]:.4f}")
        
    bmi_imp = shap_df_3[shap_df_3['Feature'].str.contains('bmi', case=False)]['Importance'].values[0]
    top_voice_imp = shap_df_3[~shap_df_3['Feature'].str.contains('bmi', case=False)]['Importance'].values[0]
    
    print(f"\nBMI importance: {bmi_imp:.4f}")
    print(f"Top voice feature importance: {top_voice_imp:.4f}")
    print(f"BMI dominance ratio: {bmi_imp / top_voice_imp:.2f}x")
    
    explainer_5m = shap.TreeExplainer(model5m)
    shap_vals_5m = explainer_5m.shap_values(X_test_m_scaled)
    mean_abs_shap_5m = np.abs(shap_vals_5m).mean(axis=0)
    
    shap_df_5m = pd.DataFrame({'Feature': voice_cols, 'Importance': mean_abs_shap_5m})
    shap_df_5m = shap_df_5m.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    print("\nTop 5 Features for Model 5M (Male Only - Pure Voice Signal):")
    for i in range(5):
        print(f"  {i+1}. {shap_df_5m['Feature'][i]:<15} - {shap_df_5m['Importance'][i]:.4f}")

    # ---------------------------------------------------------
    # FINAL COMPARISON TABLE
    # ---------------------------------------------------------
    try:
        model1 = joblib.load('model1_voice_only.pkl')
        auc_1 = roc_auc_score(y_test, model1.predict_proba(X_test_A)[:, 1])
        sens_1 = recall_score(y_test, model1.predict(X_test_A))
        
        model2 = joblib.load('model2_voice_demo.pkl')
        auc_2 = roc_auc_score(y_test, model2.predict_proba(X_test_B)[:, 1])
        sens_2 = recall_score(y_test, model2.predict(X_test_B))
        
        explainer_2 = shap.TreeExplainer(model2)
        shap_df_2 = pd.DataFrame({'Feature': feats_B, 'Importance': np.abs(explainer_2.shap_values(X_test_B)).mean(axis=0)})
        gender_imp_2 = shap_df_2[shap_df_2['Feature'].str.contains('gender|sex', case=False)]['Importance'].values[0]
    except:
        auc_1, sens_1, auc_2, sens_2, gender_imp_2 = 0, 0, 0, 0, 0
    
    explainer_4 = shap.TreeExplainer(model4)
    shap_df_4 = pd.DataFrame({'Feature': feats_4, 'Importance': np.abs(explainer_4.shap_values(X_test_4)).mean(axis=0)})
    gender_imp_4 = shap_df_4[shap_df_4['Feature'].str.contains('gender|sex', case=False)]['Importance'].values[0]
    
    print("\n" + "╔" + "═"*62 + "╗")
    print("║                  ALL MODELS COMPARISON                       ║")
    print("╠" + "═"*14 + "╦" + "═"*26 + "╦" + "═"*9 + "╦" + "═"*10 + "╣")
    print(f"║ {'Model':<12} ║ {'Features':<24} ║ {'AUC-ROC':<7} ║ {'Sensit.':<8} ║")
    print("╠" + "═"*14 + "╬" + "═"*26 + "╬" + "═"*9 + "╬" + "═"*10 + "╣")
    print(f"║ {'Model 1':<12} ║ {'Voice only':<24} ║ {auc_1:<7.4f} ║ {sens_1:<8.4f} ║")
    print(f"║ {'Model 2':<12} ║ {'Voice + BMI + Gender':<24} ║ {auc_2:<7.4f} ║ {sens_2:<8.4f} ║")
    print(f"║ {'Model 3':<12} ║ {'Voice + BMI only':<24} ║ {auc_3:<7.4f} ║ {sens_3:<8.4f} ║")
    print(f"║ {'Model 4':<12} ║ {'Voice + Gender only':<24} ║ {auc_4:<7.4f} ║ {sens_4:<8.4f} ║")
    print(f"║ {'Model 5M':<12} ║ {'Male patients only':<24} ║ {auc_5m:<7.4f} ║ {sens_5m:<8.4f} ║")
    print(f"║ {'Model 5F':<12} ║ {'Female patients only':<24} ║ {auc_5f:<7.4f} ║ {sens_5f:<8.4f} ║")
    print("╚" + "═"*14 + "╩" + "═"*26 + "╩" + "═"*9 + "╩" + "═"*10 + "╝")
    
    print("\nGENDER DOMINANCE ANALYSIS:")
    print(f" Model 2 gender importance  : {gender_imp_2:.2f}")
    print(f" Model 3 BMI importance     : {bmi_imp:.2f}")
    print(f" Model 4 gender importance  : {gender_imp_4:.2f}")
    print(f" Model 5M top feature       : {shap_df_5m['Feature'][0]} ({shap_df_5m['Importance'][0]:.2f})")
    
    print("\nRECOMMENDATION:")
    print(" If Model 3 AUC is close to Model 2 -> use Model 3, drop gender")
    print(" If Model 5M/5F AUC > Model 1 -> use stratified approach")

if __name__ == "__main__":
    main()
