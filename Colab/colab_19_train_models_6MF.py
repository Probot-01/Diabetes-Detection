import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print("TRAINING MODELS 6M AND 6F (Voice + BMI, Gender Stratified)")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. LOAD ASSETS
    # ---------------------------------------------------------
    # Load the original raw dataset. We need actual non-SMOTE rows
    # so we can map real BMI values to each patient.
    try:
        df = pd.read_csv('dataset.csv')
        scaler_A   = joblib.load('scaler_A.pkl')    # 267-feature voice scaler
        scaler_bmi = joblib.load('scaler_bmi.pkl')  # 1-feature BMI scaler
    except Exception as e:
        print(f"Failed to load assets: {e}")
        return

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # ---------------------------------------------------------
    # 2. FIND EXACT COLUMN NAMES (case-insensitive)
    # ---------------------------------------------------------
    # The CSV might have 'GENDER', 'Gender', 'gender' etc.
    # We search case-insensitively to avoid KeyErrors.
    gender_col = next((c for c in df.columns if c.lower() in ['gender', 'sex']), None)
    label_col  = next((c for c in df.columns if c.lower() in ['label', 'class']), None)
    bmi_col    = next((c for c in df.columns if c.lower() == 'bmi'), None)

    if not gender_col or not label_col or not bmi_col:
        print(f"Could not find required columns!")
        print(f"  gender_col={gender_col}, label_col={label_col}, bmi_col={bmi_col}")
        return

    print(f"\nUsing columns: Gender='{gender_col}', Label='{label_col}', BMI='{bmi_col}'")

    # ---------------------------------------------------------
    # 3. FIND VOICE FEATURE COLUMNS
    # ---------------------------------------------------------
    # Load the list of 267 voice feature names from the CSV we saved earlier.
    # These are the column names that exist in both dataset.csv and feature_names_A.csv.
    feats_A = pd.read_csv('feature_names_A.csv').iloc[:, 0].tolist()
    feats_A = [str(f).strip() for f in feats_A]

    # Only keep features that actually exist in the dataset
    voice_cols = [c for c in feats_A if c in df.columns]
    print(f"Voice feature columns found in dataset: {len(voice_cols)} (expected 267)")

    if len(voice_cols) < 200:
        print("Too few voice columns found. Check that dataset.csv contains feature columns.")
        return

    # ---------------------------------------------------------
    # 4. AUTO-DETECT MALE / FEMALE VALUES
    # ---------------------------------------------------------
    print("\nGender distribution in dataset:")
    m_val, f_val = None, None
    for g_val in df[gender_col].unique():
        sub = df[df[gender_col] == g_val]
        rate = sub[label_col].mean()
        print(f"  Gender '{g_val}': {len(sub)} rows, Diabetic Rate: {rate*100:.1f}%")
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
    print(f"\nDetected: Male='{m_val}', Female='{f_val}'")

    # ---------------------------------------------------------
    # XGBoost settings — same hyperparameters for all models
    # scale_pos_weight=3 compensates for the class imbalance
    # (roughly 3x more non-diabetic than diabetic patients)
    # ---------------------------------------------------------
    XGB_PARAMS = dict(
        n_estimators=100,
        random_state=42,
        scale_pos_weight=3,
        eval_metric='logloss',
        use_label_encoder=False
    )

    smote = SMOTE(random_state=42)
    results = {}

    # ==========================================================
    # MODEL 6M — Male patients: Voice + BMI
    # ==========================================================
    print("\n" + "="*40)
    print("MODEL 6M — Male + BMI")
    print("="*40)

    df_m = df[df[gender_col] == m_val].copy()
    print(f"Male subset size: {len(df_m)}")
    print(f"Class distribution: {df_m[label_col].value_counts().to_dict()}")

    # Extract voice features + BMI as a 268-column raw array
    X_m_voice = df_m[voice_cols].values          # shape (N, 267)
    X_m_bmi   = df_m[[bmi_col]].values           # shape (N, 1)
    y_m        = df_m[label_col].values

    # Stratified 80/20 split — preserves diabetic/non-diabetic ratio
    X_m_voice_tr, X_m_voice_te, X_m_bmi_tr, X_m_bmi_te, y_m_tr, y_m_te = \
        train_test_split(X_m_voice, X_m_bmi, y_m,
                         test_size=0.2, stratify=y_m, random_state=42)

    # SMOTE only on training split to avoid data leakage
    # We temporarily concatenate voice + BMI before SMOTE so both are balanced together
    X_m_combined_tr = np.hstack([X_m_voice_tr, X_m_bmi_tr])
    X_m_combined_sm, y_m_sm = smote.fit_resample(X_m_combined_tr, y_m_tr)
    print(f"After SMOTE: {X_m_combined_sm.shape[0]} training samples")

    # Split back into voice and BMI after SMOTE
    X_m_voice_sm = X_m_combined_sm[:, :len(voice_cols)]
    X_m_bmi_sm   = X_m_combined_sm[:, len(voice_cols):]

    # Fit a NEW voice scaler on male-only SMOTE training data
    # (Different from scaler_A which was fitted on ALL patients)
    scaler_maleB = StandardScaler()
    X_m_voice_tr_scaled = scaler_maleB.fit_transform(X_m_voice_sm)
    X_m_voice_te_scaled  = scaler_maleB.transform(X_m_voice_te)
    joblib.dump(scaler_maleB, 'scaler_maleB.pkl')
    print("Saved -> scaler_maleB.pkl")

    # Scale BMI using the shared scaler_bmi (already fitted on all patients)
    X_m_bmi_tr_scaled = scaler_bmi.transform(X_m_bmi_sm)
    X_m_bmi_te_scaled  = scaler_bmi.transform(X_m_bmi_te)

    # Concatenate voice + BMI → (N, 268)
    X_m_train_full = np.hstack([X_m_voice_tr_scaled, X_m_bmi_tr_scaled])
    X_m_test_full  = np.hstack([X_m_voice_te_scaled,  X_m_bmi_te_scaled])
    print(f"Train shape: {X_m_train_full.shape} | Test shape: {X_m_test_full.shape}")

    model6M = XGBClassifier(**XGB_PARAMS)
    model6M.fit(X_m_train_full, y_m_sm)
    joblib.dump(model6M, 'model6M_male_bmi.pkl')
    print("Saved -> model6M_male_bmi.pkl")

    auc_6m  = roc_auc_score(y_m_te, model6M.predict_proba(X_m_test_full)[:, 1])
    sens_6m = recall_score(y_m_te, model6M.predict(X_m_test_full))
    f1_6m   = f1_score(y_m_te, model6M.predict(X_m_test_full))
    print(f"AUC: {auc_6m:.4f} | Sensitivity: {sens_6m:.4f} | F1: {f1_6m:.4f}")
    results['Model 6M'] = (auc_6m, sens_6m, 'Male + BMI')

    # ==========================================================
    # MODEL 6F — Female patients: Voice + BMI
    # ==========================================================
    print("\n" + "="*40)
    print("MODEL 6F — Female + BMI")
    print("="*40)

    df_f = df[df[gender_col] == f_val].copy()
    print(f"Female subset size: {len(df_f)}")
    print(f"Class distribution: {df_f[label_col].value_counts().to_dict()}")

    X_f_voice = df_f[voice_cols].values
    X_f_bmi   = df_f[[bmi_col]].values
    y_f        = df_f[label_col].values

    X_f_voice_tr, X_f_voice_te, X_f_bmi_tr, X_f_bmi_te, y_f_tr, y_f_te = \
        train_test_split(X_f_voice, X_f_bmi, y_f,
                         test_size=0.2, stratify=y_f, random_state=42)

    X_f_combined_tr = np.hstack([X_f_voice_tr, X_f_bmi_tr])
    X_f_combined_sm, y_f_sm = smote.fit_resample(X_f_combined_tr, y_f_tr)
    print(f"After SMOTE: {X_f_combined_sm.shape[0]} training samples")

    X_f_voice_sm = X_f_combined_sm[:, :len(voice_cols)]
    X_f_bmi_sm   = X_f_combined_sm[:, len(voice_cols):]

    scaler_femaleB = StandardScaler()
    X_f_voice_tr_scaled = scaler_femaleB.fit_transform(X_f_voice_sm)
    X_f_voice_te_scaled  = scaler_femaleB.transform(X_f_voice_te)
    joblib.dump(scaler_femaleB, 'scaler_femaleB.pkl')
    print("Saved -> scaler_femaleB.pkl")

    X_f_bmi_tr_scaled = scaler_bmi.transform(X_f_bmi_sm)
    X_f_bmi_te_scaled  = scaler_bmi.transform(X_f_bmi_te)

    X_f_train_full = np.hstack([X_f_voice_tr_scaled, X_f_bmi_tr_scaled])
    X_f_test_full  = np.hstack([X_f_voice_te_scaled,  X_f_bmi_te_scaled])
    print(f"Train shape: {X_f_train_full.shape} | Test shape: {X_f_test_full.shape}")

    model6F = XGBClassifier(**XGB_PARAMS)
    model6F.fit(X_f_train_full, y_f_sm)
    joblib.dump(model6F, 'model6F_female_bmi.pkl')
    print("Saved -> model6F_female_bmi.pkl")

    auc_6f  = roc_auc_score(y_f_te, model6F.predict_proba(X_f_test_full)[:, 1])
    sens_6f = recall_score(y_f_te, model6F.predict(X_f_test_full))
    f1_6f   = f1_score(y_f_te, model6F.predict(X_f_test_full))
    print(f"AUC: {auc_6f:.4f} | Sensitivity: {sens_6f:.4f} | F1: {f1_6f:.4f}")
    results['Model 6F'] = (auc_6f, sens_6f, 'Female + BMI')

    # ---------------------------------------------------------
    # FINAL COMPARISON TABLE
    # ---------------------------------------------------------
    print("\n" + "╔" + "═"*14 + "╦" + "═"*26 + "╦" + "═"*9 + "╦" + "═"*10 + "╗")
    print(f"║ {'Model':<12} ║ {'Features':<24} ║ {'AUC-ROC':<7} ║ {'Sensit.':<8} ║")
    print("╠" + "═"*14 + "╬" + "═"*26 + "╬" + "═"*9 + "╬" + "═"*10 + "╣")
    print(f"║ {'Model 1':<12} ║ {'Voice only':<24} ║ {'0.8044':<7} ║ {'0.6375':<8} ║")
    print(f"║ {'Model 3':<12} ║ {'Voice + BMI':<24} ║ {'0.7870':<7} ║ {'0.6375':<8} ║")
    print(f"║ {'Model 5M':<12} ║ {'Male only':<24} ║ {'0.8841':<7} ║ {'0.6379':<8} ║")
    print(f"║ {'Model 5F':<12} ║ {'Female only':<24} ║ {'0.7278':<7} ║ {'0.4091':<8} ║")
    print(f"║ {'Model 6M':<12} ║ {'Male + BMI':<24} ║ {auc_6m:<7.4f} ║ {sens_6m:<8.4f} ║")
    print(f"║ {'Model 6F':<12} ║ {'Female + BMI':<24} ║ {auc_6f:<7.4f} ║ {sens_6f:<8.4f} ║")
    print("╚" + "═"*14 + "╩" + "═"*26 + "╩" + "═"*9 + "╩" + "═"*10 + "╝")

    print("\nFiles to download from Colab:")
    print("  model6M_male_bmi.pkl")
    print("  model6F_female_bmi.pkl")
    print("  scaler_maleB.pkl")
    print("  scaler_femaleB.pkl")

if __name__ == "__main__":
    main()
