
# ============================================================
# colab_02_preprocessing.py
# Diabetes Voice Dataset — Cleaning & Preparation Pipeline
#
# IMPORTANT: Run colab_01_data_inspection.py first so that
#            the DataFrame `df` already exists in memory.
#            This script expects `df` to be available.
# ============================================================

import numpy as np
import pandas as pd

# We'll print a ruler line often to keep output readable
RULER = "=" * 60

print(RULER)
print("🧹 Starting Data Cleaning & Preparation Pipeline")
print(RULER)


# ============================================================
# STEP 1 — Drop the BSL Column
# ============================================================
# BSL = Blood Sugar Level.
# BSL directly causes diabetes, so if we give it to the model,
# the model would simply read blood sugar and ignore the voice
# features entirely. That's "cheating" — the model would not
# actually learn anything about voice patterns.
# We must remove BSL BEFORE creating X (features).
# ------------------------------------------------------------

# Make a working copy so the original `df` stays intact
df_clean = df.copy()

if 'BSL' in df_clean.columns:
    df_clean.drop(columns=['BSL'], inplace=True)
    print("\n✅ STEP 1 — BSL column removed.")
    print(f"   Remaining columns: {df_clean.shape[1]}")
else:
    print("\n⚠️  STEP 1 — 'BSL' column not found. Skipping drop.")
    print(f"   Columns present: {list(df_clean.columns[:10])} ...")


# ============================================================
# STEP 2 — Separate Features (X) and Label (y)
# ============================================================
# In machine learning we always separate:
#   X = the INPUT columns the model will learn from (features)
#   y = the OUTPUT column the model must predict (label)
#
# Our label column is called 'label':
#   0 = Non-Diabetic
#   1 = Diabetic
#
# X contains everything EXCEPT the label column (and BSL,
# which we already removed in Step 1).
# ------------------------------------------------------------

LABEL_COL = 'label'   # 👈 Change this if your column is named differently

if LABEL_COL not in df_clean.columns:
    raise ValueError(f"Label column '{LABEL_COL}' not found! "
                     f"Available columns: {list(df_clean.columns[-10:])}")

# y = just the label column (a 1-D series of 0s and 1s)
y = df_clean[LABEL_COL].copy()

# X = every column EXCEPT the label column
X = df_clean.drop(columns=[LABEL_COL]).copy()

print(f"\n✅ STEP 2 — Features and label separated.")
print(f"   X shape (samples × features) : {X.shape}")
print(f"   y shape (samples,)           : {y.shape}")
print(f"   Unique label values          : {sorted(y.unique())}")


# ============================================================
# STEP 3 — Split X into Voice Features and Demographic Features
# ============================================================
# Our feature set has two very different types of columns:
#
#   Voice Features (304 columns):
#     Extracted from audio — MFCCs, Chroma, Tonnetz, etc.
#     These capture how a person sounds.
#
#   Demographic Features (3 columns):
#     Age, Gender, BMI — personal health/biological info.
#     These are NOT derived from the voice recording.
#
# We separate them now so we can treat them differently
# during preprocessing (e.g. scaling, encoding).
# ------------------------------------------------------------

# Define which columns are demographic
DEMOGRAPHIC_COLS = ['Age', 'Gender', 'BMI']

# Check all demographic columns actually exist
missing_demo = [c for c in DEMOGRAPHIC_COLS if c not in X.columns]
if missing_demo:
    print(f"\n⚠️  STEP 3 — Missing demographic column(s): {missing_demo}")
    DEMOGRAPHIC_COLS = [c for c in DEMOGRAPHIC_COLS if c in X.columns]

# Voice features = everything that is NOT a demographic column
voice_feature_names = [col for col in X.columns if col not in DEMOGRAPHIC_COLS]

# Build the two sub-DataFrames
voice_features      = X[voice_feature_names].copy()
demographic_features = X[DEMOGRAPHIC_COLS].copy()

print(f"\n✅ STEP 3 — Features split into two groups.")
print(f"   Voice feature columns      : {voice_features.shape[1]}")
print(f"   Demographic feature columns: {demographic_features.shape[1]}")
print(f"   Demographic columns        : {DEMOGRAPHIC_COLS}")


# ============================================================
# STEP 4 — Drop Chroma and Tonnetz from Voice Features
# ============================================================
# Chroma and Tonnetz are music-analysis features.
# They capture harmonic / tonal properties useful in music,
# but voice pathology research shows they add noise rather
# than signal for detecting medical conditions from speech.
# Removing them reduces dimensionality and keeps the model
# focused on clinically meaningful voice properties.
#
# Expected removals:
#   Chroma  → 24 columns  (names contain "chroma")
#   Tonnetz → 12 columns  (names contain "tonnetz")
# ------------------------------------------------------------

# Find all Chroma columns (case-insensitive search by name)
chroma_cols  = [col for col in voice_features.columns
                if 'chroma' in col.lower()]

# Find all Tonnetz columns
tonnetz_cols = [col for col in voice_features.columns
                if 'tonnetz' in col.lower()]

cols_to_drop = chroma_cols + tonnetz_cols

if cols_to_drop:
    voice_features.drop(columns=cols_to_drop, inplace=True)
    print(f"\n✅ STEP 4 — Chroma and Tonnetz removed.")
    print(f"   Chroma columns removed      : {len(chroma_cols)}")
    print(f"   Tonnetz columns removed     : {len(tonnetz_cols)}")
    print(f"   Remaining voice features    : {voice_features.shape[1]}")
else:
    print("\n⚠️  STEP 4 — No Chroma or Tonnetz columns detected by name.")
    print("   Check that your column names contain 'chroma' / 'tonnetz'.")
    print(f"   Sample voice column names: {list(voice_features.columns[:8])}")

print(f"   Chroma and Tonnetz removed. Remaining voice features: {voice_features.shape[1]}")


# ============================================================
# STEP 5 — Check for and Handle Missing Values
# ============================================================
# Machine learning models cannot process NaN (empty) values.
# A common strategy is to fill each missing value with the
# MEAN of that column — this is called "mean imputation".
# It's not perfect, but it's safe for a first pass.
#
# We handle voice and demographic features separately because:
#   - Voice features are all numeric → use mean
#   - Gender is text/categorical → we handle separately
# ------------------------------------------------------------

total_filled = 0

# --- Handle Voice Features (all numeric) ---
voice_nulls = voice_features.isnull().sum().sum()
if voice_nulls > 0:
    print(f"\n⚠️  STEP 5 — Found {voice_nulls} missing value(s) in voice features.")
    # Fill each column's NaN with that column's mean value
    voice_features.fillna(voice_features.mean(), inplace=True)
    total_filled += voice_nulls
    print(f"   Filled {voice_nulls} missing voice feature value(s) with column mean.")
else:
    print(f"\n✅ STEP 5 — Voice features: No missing values. ✔")

# --- Handle Demographic Features ---
for col in demographic_features.columns:
    null_count = demographic_features[col].isnull().sum()
    if null_count > 0:
        if demographic_features[col].dtype == object:
            # For text columns like Gender, use the most frequent value (mode)
            fill_val = demographic_features[col].mode()[0]
            demographic_features[col].fillna(fill_val, inplace=True)
            print(f"   Filled {null_count} missing '{col}' value(s) with mode: '{fill_val}'")
        else:
            # For numeric columns like Age and BMI, use mean
            fill_val = demographic_features[col].mean()
            demographic_features[col].fillna(fill_val, inplace=True)
            print(f"   Filled {null_count} missing '{col}' value(s) with mean: {fill_val:.2f}")
        total_filled += null_count

if total_filled == 0:
    print("   Demographic features: No missing values. ✔")
    print(f"\n   Total values filled across ALL columns: 0")
else:
    print(f"\n   Total values filled across ALL columns: {total_filled}")


# ============================================================
# STEP 6 — Combine & Save Cleaned Data
# ============================================================
# We now merge voice_features and demographic_features back
# into one final feature matrix X_final, then save:
#
#   features.npy       → the full feature matrix (numbers)
#   labels.npy         → the label array (0s and 1s)
#   feature_names.csv  → list of column names (so we remember
#                        what each column in features.npy means)
#
# .npy files are NumPy's fast binary format — much faster to
# load than CSV for large arrays.
# ------------------------------------------------------------

# Merge voice + demographic back into one feature matrix
X_final = pd.concat([voice_features, demographic_features], axis=1)

print(f"\n✅ STEP 6 — Final feature matrix assembled.")
print(f"   X_final shape : {X_final.shape}")
print(f"   y shape       : {y.shape}")

# Convert to NumPy arrays for saving
X_array = X_final.values.astype(np.float32)   # float32 is memory-efficient
y_array = y.values.astype(np.int32)           # int for class labels

# Save feature matrix
np.save("features.npy", X_array)

# Save label array
np.save("labels.npy", y_array)

# Save column names as CSV (one column name per row)
feature_names_df = pd.DataFrame({'feature_name': X_final.columns.tolist()})
feature_names_df.to_csv("feature_names.csv", index=False)

print("\n✅ Saved features.npy, labels.npy, feature_names.csv")

# --- Final Summary ---
print("\n" + RULER)
print("📦 PREPROCESSING COMPLETE — SUMMARY")
print(RULER)
print(f"   Original columns (after BSL drop) : {df_clean.shape[1] - 1}")  # -1 for label
print(f"   Chroma + Tonnetz removed          : {len(cols_to_drop)}")
print(f"   Final feature count               : {X_final.shape[1]}")
print(f"     └─ Voice features               : {voice_features.shape[1]}")
print(f"     └─ Demographic features         : {demographic_features.shape[1]}")
print(f"   Total samples                     : {X_final.shape[0]}")
print(f"   Missing values filled             : {total_filled}")
print(f"   Files saved                       : features.npy, labels.npy, feature_names.csv")
print(RULER)
print("🚀 Ready for the next step: Feature Scaling & Model Training!")
print(RULER)
