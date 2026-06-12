import numpy as np

def main():
    print("=" * 60)
    print("🔍 DATA SCALING DIAGNOSTIC")
    print("=" * 60)
    
    try:
        X_train_A = np.load('X_train_A.npy', allow_pickle=True).astype(np.float32)
    except Exception as e:
        print(f"Failed to load X_train_A.npy: {e}")
        return

    # Calculate overall metrics across the entire 2D matrix
    overall_min = np.min(X_train_A)
    overall_max = np.max(X_train_A)
    overall_mean = np.mean(X_train_A)
    overall_std = np.std(X_train_A)

    print(f"Overall min value : {overall_min:.4f}")
    print(f"Overall max value : {overall_max:.4f}")
    print(f"Overall mean      : {overall_mean:.4f}")
    print(f"Overall std       : {overall_std:.4f}")
    
    print("\n" + "-" * 60)
    print("INTERPRETATION:")
    
    # Heuristic to check if standard scaled (mean ~ 0, std ~ 1)
    # Scaled data rarely has a max value above 20, whereas raw audio often exceeds 100+
    if abs(overall_mean) < 0.5 and abs(overall_std - 1.0) < 0.5 and abs(overall_max) < 50:
        print("⚠️ DATA IS ALREADY SCALED. Do not fit new scaler.")
    else:
        print("✅ DATA IS RAW. Safe to fit new scaler.")

if __name__ == "__main__":
    main()
