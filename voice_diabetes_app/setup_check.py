import os
import sys

def check_imports():
    print("--- 1. Library Import Check ---")
    libraries = {
        "gradio": "gradio",
        "librosa": "librosa",
        "numpy": "numpy",
        "joblib": "joblib",
        "praat-parselmouth": "parselmouth", # The python import name differs from pip name
        "scikit-learn": "sklearn",          # The python import name differs from pip name
        "scipy": "scipy",
        "pandas": "pandas",
        "xgboost": "xgboost"
    }
    
    all_passed = True
    for package, module_name in libraries.items():
        try:
            __import__(module_name)
            print(f"✅ PASS: {package} is installed")
        except ImportError:
            print(f"❌ FAIL: {package} is missing (run: pip install {package})")
            all_passed = False
            
    return all_passed

def check_models():
    print("\n--- 2. Model Files Check ---")
    expected_files = [
        "model1_voice_only.pkl",
        "model2_voice_demo.pkl",
        "scaler_A.pkl",
        "scaler_B.pkl",
        "feature_names_A.csv",
        "feature_names_B.csv"
    ]
    
    all_passed = True
    for f in expected_files:
        path = os.path.join("models", f)
        if os.path.exists(path):
            print(f"✅ PASS: {f} found in models/")
        else:
            print(f"❌ FAIL: {f} is missing from models/")
            all_passed = False
            
    return all_passed

def main():
    print("=" * 50)
    print("🔍 DIABETES WEB APP SETUP CHECK")
    print("=" * 50)
    
    imports_ok = check_imports()
    models_ok = check_models()
    
    print("\n" + "=" * 50)
    if imports_ok and models_ok:
        print("✅ ALL GOOD — run: python app.py")
    else:
        print("❌ FIX ABOVE ISSUES FIRST")
    print("=" * 50)

if __name__ == "__main__":
    main()
