import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# =====================================================================
# STEP 1 - LOAD ORIGINAL CSV
# =====================================================================
# Search for the CSV file in /content/ (Colab default upload location)
csv_candidates = [f for f in os.listdir('/content/') if f.endswith('.csv')]
print("CSV files found in /content/:")
for f in sorted(csv_candidates):
    size = os.path.getsize(f'/content/{f}')
    print(f"  - {f}  ({size:,} bytes)")

# Pick the LARGEST CSV — the dataset is always much bigger than feature name files
csv_candidates_sorted = sorted(
    csv_candidates,
    key=lambda f: os.path.getsize(f'/content/{f}'),
    reverse=True
)
CSV_PATH = '/content/' + csv_candidates_sorted[0]
print(f"\nAuto-selected (largest): {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# =====================================================================
# STEP 2 - RECREATE EXACT SAME SPLIT
# =====================================================================

# Find the label column (case-insensitive)
label_col = next((c for c in df.columns if c.lower() in ['label', 'class', 'diabetes']), None)

if label_col is None:
    print("\n❌ Could not auto-detect label column!")
    print("All columns in your CSV:")
    for i, c in enumerate(df.columns):
        print(f"  {i:3}: {c}")
    raise ValueError(
        "Set label_col manually below. "
        "Example: label_col = 'your_label_column_name'"
    )

print(f"\nLabel column detected: '{label_col}'")

# Drop columns that were excluded during training
COLS_TO_DROP = []

# Drop label column
COLS_TO_DROP.append(label_col)

# Drop BSL, Chroma, Tonnetz columns if present
for col in df.columns:
    if any(kw in col.upper() for kw in ['BSL', 'CHROMA', 'TONNETZ']):
        COLS_TO_DROP.append(col)
        print(f"Dropping column: {col}")

# Drop is_synthetic column if present
if 'is_synthetic' in df.columns:
    COLS_TO_DROP.append('is_synthetic')
    print("Dropping column: is_synthetic")

# Capture gender and BMI BEFORE dropping them
# These are saved separately so Models 6M/6F can use them
gender_col = next((c for c in df.columns if c.lower() in ['gender', 'sex']), None)
bmi_col    = next((c for c in df.columns if c.lower() == 'bmi'), None)
print(f"Gender column: '{gender_col}' | BMI column: '{bmi_col}'")

# Drop from X so it contains only voice features
if gender_col: COLS_TO_DROP.append(gender_col)
if bmi_col:    COLS_TO_DROP.append(bmi_col)

# Drop MFCC1 — training data started from MFCC2 (models expect 79 MFCCs, not 80)
COLS_TO_DROP.append('MFCC1')

# Drop AGE — was in the raw CSV but not used as a training feature
age_col = next((c for c in df.columns if 'age' in c.lower()), None)
if age_col:
    COLS_TO_DROP.append(age_col)
    print(f"Dropping age column: '{age_col}'")

COLS_TO_DROP = list(set(COLS_TO_DROP))  # deduplicate

# Separate features and label
y = df[label_col].values
X = df.drop(columns=COLS_TO_DROP, errors='ignore')

print(f"\nX shape after all drops: {X.shape}  (expected: (1600, 267))")
print(f"First 5 columns : {X.columns.tolist()[:5]}")
print(f"Last  5 columns : {X.columns.tolist()[-5:]}")

# Sanity check
assert X.shape[1] == 267, (
    f"Expected 267 features, got {X.shape[1]}.\n"
    f"Columns: {X.columns.tolist()}"
)
print("Feature count check PASSED: 267 ✅")


# Recreate EXACT same split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"\nSplit done:")
print(f"  Train: {X_train.shape}")
print(f"  Test : {X_test.shape}")

# =====================================================================
# STEP 3 - SAVE TEST SET ONLY (unscaled raw values)
# =====================================================================
# Save X_test as raw unscaled numpy array — 267 VOICE FEATURES ONLY
np.save('/content/X_test_raw.npy', X_test.values.astype(np.float32))
np.save('/content/y_test.npy', y_test.astype(np.int32))

# Save exact feature names in order
pd.DataFrame(X_test.columns, columns=['feature_name']).to_csv(
    '/content/feature_names_A.csv', index=False
)

# Save BMI and Gender for the test rows separately
# Models 6M/6F need BMI — app2.py reads it from here
if gender_col or bmi_col:
    meta_cols = {}
    if gender_col: meta_cols['gender'] = df[gender_col]
    if bmi_col:    meta_cols['bmi']    = df[bmi_col]
    df_meta = pd.DataFrame(meta_cols)

    # Apply the same train_test_split to meta columns using the same indices
    full_index = df.index
    _, idx_test = train_test_split(
        full_index, test_size=0.2,
        stratify=y, random_state=42
    )
    df_meta_test = df_meta.loc[idx_test].reset_index(drop=True)
    df_meta_test.to_csv('/content/X_test_meta.csv', index=False)
    print("  /content/X_test_meta.csv  (gender + BMI for test rows)")

print("\nSaved files:")
print("  /content/X_test_raw.npy    (267 voice features, unscaled)")
print("  /content/y_test.npy        (true labels 0/1)")
print("  /content/feature_names_A.csv")
if gender_col or bmi_col:
    print("  /content/X_test_meta.csv   (gender + BMI per test row)")

# =====================================================================
# STEP 4 - PRINT CONFIRMATION
# =====================================================================
X_check = np.load('/content/X_test_raw.npy')
y_check = np.load('/content/y_test.npy')

diabetic     = int(np.sum(y_check == 1))
non_diabetic = int(np.sum(y_check == 0))

print("\n" + "="*45)
print("CONFIRMATION")
print("="*45)
print(f"X_test shape        : {X_check.shape}")
print(f"y_test shape        : {y_check.shape}")
print(f"Diabetic in test    : {diabetic}")
print(f"Non-diabetic in test: {non_diabetic}")
print(f"Feature names saved : {len(X_test.columns)}")
print("="*45)
print("\nDownload all 3 files from /content/ and place in:")
print("  voice_diabetes_app/models/")
