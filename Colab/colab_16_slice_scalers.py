import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os

def get_path(filename):
    if os.path.exists(filename): return filename
    alt = os.path.join("voice_diabetes_app", "models", filename)
    if os.path.exists(alt): return alt
    return filename

def main():
    print("=" * 60)
    print("✂️  SLICING ORIGINAL SCALER BY FEATURE INDICES")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. LOAD ASSETS
    # ---------------------------------------------------------
    try:
        scaler = joblib.load('scaler.pkl')
        
        # Load universal 272 features
        df_all = pd.read_csv('feature_names.csv')
        feats_all = df_all['feature_name'].tolist() if 'feature_name' in df_all.columns else df_all.iloc[:, 0].tolist()
        
        # Load specific version subset features
        df_A = pd.read_csv(get_path('feature_names_A.csv'))
        feats_A = df_A['feature_name'].tolist() if 'feature_name' in df_A.columns else df_A.iloc[:, 0].tolist()
        
        df_B = pd.read_csv(get_path('feature_names_B.csv'))
        feats_B = df_B['feature_name'].tolist() if 'feature_name' in df_B.columns else df_B.iloc[:, 0].tolist()
        
    except Exception as e:
        print(f"Failed to load required assets: {e}")
        return

    print("--- 1. LOADING CONFIRMATION ---")
    print(f"Original scaler mean_ shape : {scaler.mean_.shape}")
    print(f"Original features length    : {len(feats_all)}")
    print(f"Version A features length   : {len(feats_A)}")
    print(f"Version B features length   : {len(feats_B)}")

    # Sanitize feature names to ensure perfect string matching
    feats_all = [str(f).strip() for f in feats_all]
    feats_A = [str(f).strip() for f in feats_A]
    feats_B = [str(f).strip() for f in feats_B]

    # ---------------------------------------------------------
    # 2. SLICING SCALER A
    # ---------------------------------------------------------
    print("\n--- 2. SLICING SCALER A ---")
    indices_A = []
    for f in feats_A:
        if f in feats_all:
            indices_A.append(feats_all.index(f))
        else:
            print(f"Warning: {f} not found in master feature list!")
            
    print(f"Found {len(indices_A)} matching column indices for Version A.")
    if len(indices_A) != 267:
        print(f"⚠️  WARNING: Expected 267 indices, found {len(indices_A)}")
        
    scaler_A_new = StandardScaler()
    scaler_A_new.mean_ = scaler.mean_[indices_A]
    scaler_A_new.scale_ = scaler.scale_[indices_A]
    scaler_A_new.var_ = scaler.var_[indices_A]
    scaler_A_new.n_features_in_ = len(indices_A)
    scaler_A_new.n_samples_seen_ = scaler.n_samples_seen_
    
    joblib.dump(scaler_A_new, 'scaler_A_sliced.pkl')
    print("Saved -> scaler_A_sliced.pkl")

    # ---------------------------------------------------------
    # 3. SLICING SCALER B
    # ---------------------------------------------------------
    print("\n--- 3. SLICING SCALER B ---")
    indices_B = []
    for f in feats_B:
        if f in feats_all:
            indices_B.append(feats_all.index(f))
        else:
            print(f"Warning: {f} not found in master feature list!")
            
    print(f"Found {len(indices_B)} matching column indices for Version B.")
    if len(indices_B) != 269:
        print(f"⚠️  WARNING: Expected 269 indices, found {len(indices_B)}")
        
    scaler_B_new = StandardScaler()
    scaler_B_new.mean_ = scaler.mean_[indices_B]
    scaler_B_new.scale_ = scaler.scale_[indices_B]
    scaler_B_new.var_ = scaler.var_[indices_B]
    scaler_B_new.n_features_in_ = len(indices_B)
    scaler_B_new.n_samples_seen_ = scaler.n_samples_seen_
    
    joblib.dump(scaler_B_new, 'scaler_B_sliced.pkl')
    print("Saved -> scaler_B_sliced.pkl")

    # ---------------------------------------------------------
    # 4. FINAL VERIFICATION
    # ---------------------------------------------------------
    print("\n--- 4. VERIFICATION ---")
    print(f"scaler_A_new.mean_.shape -> {scaler_A_new.mean_.shape}")
    print(f"scaler_B_new.mean_.shape -> {scaler_B_new.mean_.shape}")

    # Test that the mathematical transform actually succeeds without throwing a dimension error
    dummy_A = np.zeros((1, len(indices_A)))
    dummy_B = np.zeros((1, len(indices_B)))
    
    try:
        scaler_A_new.transform(dummy_A)
        scaler_B_new.transform(dummy_B)
        print("✅ Transform test passed for both scalers.")
    except Exception as e:
        print(f"❌ Transform test failed: {e}")
        
    print("\n" + "=" * 60)
    print("Download scaler_A_sliced.pkl and scaler_B_sliced.pkl")
    print("Place them in your local models/ folder")
    print("Then update app.py to load the new filenames and restart.")
    print("=" * 60)

if __name__ == "__main__":
    main()
