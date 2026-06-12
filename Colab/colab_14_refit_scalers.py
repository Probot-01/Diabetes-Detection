import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def main():
    print("=" * 60)
    print("🔄 REFITTING SCALERS ON SUB-DATASETS")
    print("=" * 60)
    
    # ---------------------------------------------------------
    # 1. LOAD ARRAYS
    # ---------------------------------------------------------
    try:
        X_train_A = np.load('X_train_A.npy', allow_pickle=True).astype(np.float32)
        X_train_B = np.load('X_train_B.npy', allow_pickle=True).astype(np.float32)
    except Exception as e:
        print(f"Failed to load arrays: {e}")
        return

    print(f"X_train_A shape: {X_train_A.shape}")
    print(f"X_train_B shape: {X_train_B.shape}")
    
    print(f"\nX_train_A features: {X_train_A.shape[1]}")
    print(f"X_train_B features: {X_train_B.shape[1]}\n")
    
    # ---------------------------------------------------------
    # 2. FIT NEW SCALERS
    # ---------------------------------------------------------
    scaler_A = StandardScaler()
    scaler_A.fit(X_train_A)
    joblib.dump(scaler_A, 'scaler_A.pkl')
    
    scaler_B = StandardScaler()
    scaler_B.fit(X_train_B)
    joblib.dump(scaler_B, 'scaler_B.pkl')
    
    # ---------------------------------------------------------
    # 3. VERIFY
    # ---------------------------------------------------------
    print("--- VERIFICATION ---")
    print(f"scaler_A.mean_.shape  -> {scaler_A.mean_.shape}  (Expected: {X_train_A.shape[1]})")
    print(f"scaler_B.mean_.shape  -> {scaler_B.mean_.shape}  (Expected: {X_train_B.shape[1]})\n")
    
    # Hard assertions
    assert scaler_A.mean_.shape[0] == X_train_A.shape[1], "Mismatch in Scaler A!"
    assert scaler_B.mean_.shape[0] == X_train_B.shape[1], "Mismatch in Scaler B!"
    
    print(f"scaler_A fitted on {scaler_A.mean_.shape[0]} features")
    print(f"scaler_B fitted on {scaler_B.mean_.shape[0]} features")
    
    print("\n" + "!" * 60)
    print("CRITICAL WARNING:")
    print("If your X_train_A/B arrays contain data that was ALREADY SCALED")
    print("in an earlier step, fitting a new StandardScaler on them is dangerous!")
    print("Your web app extracts RAW unscaled audio features. If it feeds RAW")
    print("features into a scaler that was trained on PRE-SCALED features,")
    print("the math will be completely broken and predictions will fail.")
    print("!" * 60)
    
    print("\nDownload both and replace in your models/ folder")

if __name__ == "__main__":
    main()
