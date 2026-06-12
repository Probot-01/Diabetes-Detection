import pandas as pd
import os

def get_path(filename):
    if os.path.exists(filename): return filename
    alt = os.path.join("voice_diabetes_app", "models", filename)
    if os.path.exists(alt): return alt
    return filename

def main():
    print("=" * 60)
    print("--- MODEL 2 FEATURE ORDER DIAGNOSTIC ---")
    print("=" * 60)

    try:
        df = pd.read_csv(get_path('feature_names_B.csv'))
        if 'feature_name' in df.columns:
            features = df['feature_name'].tolist()
        else:
            features = df.iloc[:, 0].tolist()
    except Exception as e:
        print(f"Failed to load feature_names_B.csv: {e}")
        return

    features = [str(f).strip() for f in features]

    print(f"\n1. Total count: {len(features)}")
    
    print("\n2. First 10 feature names:")
    for i, f in enumerate(features[:10]):
        print(f"   {i:<3}: {f}")
        
    print("\n3. Last 10 feature names:")
    for i in range(max(0, len(features)-10), len(features)):
        print(f"   {i:<3}: {features[i]}")

    print("\n4. Exact position of JITTER and SHIMMER:")
    found_jitter = False
    found_shimmer = False
    for i, f in enumerate(features):
        if 'jitter' in f.lower():
            print(f"   JITTER  found at index: {i} ({f})")
            found_jitter = True
        if 'shimmer' in f.lower():
            print(f"   SHIMMER found at index: {i} ({f})")
            found_shimmer = True
    if not found_jitter: print("   JITTER not found!")
    if not found_shimmer: print("   SHIMMER not found!")

    print("\n5. Exact position of GENDER and BMI:")
    found_gender = False
    found_bmi = False
    for i, f in enumerate(features):
        if 'gender' in f.lower() or 'sex' in f.lower():
            print(f"   GENDER found at index: {i} ({f})")
            found_gender = True
        if 'bmi' in f.lower():
            print(f"   BMI    found at index: {i} ({f})")
            found_bmi = True
    if not found_gender: print("   GENDER not found!")
    if not found_bmi: print("   BMI not found!")

    print("\n6. ALL FEATURES IN EXACT ORDER:")
    print("-" * 30)
    for i, f in enumerate(features):
        print(f"{i:<4}: {f}")

if __name__ == "__main__":
    main()
