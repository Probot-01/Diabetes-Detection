# ============================================================
# colab_04_data_preparation.py
# Diabetes Voice Dataset — Balancing, Splitting & Scaling
#
# IMPORTANT: Run colab_02_preprocessing.py first!
#   This script expects:
#     features.npy  — cleaned feature matrix
#     labels.npy    — label array (0=Non-Diabetic, 1=Diabetic)
#
# What this script does:
#   1. Check class imbalance (75% vs 25%)
#   2. Split into train/test sets (keeping imbalance intact)
#   3. Apply SMOTE to ONLY the training set
#   4. Scale features using StandardScaler
#   5. Save everything ready for model training
# ============================================================

# Install imbalanced-learn (provides SMOTE)
# This library is not pre-installed in Colab, so we install it first.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imbalanced-learn"],
               check=True)

import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib                                    # for saving the scaler to disk

RULER  = "=" * 60
RULER2 = "─" * 60

print(RULER)
print("⚙️  Starting Data Preparation Pipeline")
print("   (Splitting → SMOTE → Scaling → Saving)")
print(RULER)


# ============================================================
# LOAD DATA
# ============================================================

X = np.load("features.npy")    # shape: (n_samples, n_features)
y = np.load("labels.npy")      # shape: (n_samples,)

n_samples, n_features = X.shape
print(f"\n✅ Data loaded.")
print(f"   Samples  : {n_samples}")
print(f"   Features : {n_features}")


# ============================================================
# STEP 1 — Class Distribution BEFORE Any Balancing
# ============================================================
# This is our starting point. We expect roughly 75% class 0
# (non-diabetic) and 25% class 1 (diabetic).
#
# Why does imbalance matter?
#   If you train a model on 75/25 data, the model will learn
#   that "just predict non-diabetic every time" gives 75%
#   accuracy — so it may do exactly that. This is dangerous
#   in medical settings because we NEED to detect the rare
#   diabetic class correctly.
# ------------------------------------------------------------

print(f"\n{RULER}")
print("📊 STEP 1 — BEFORE BALANCING")
print(RULER)

raw_counts = Counter(y)
for cls in sorted(raw_counts):
    count = raw_counts[cls]
    pct   = count / len(y) * 100
    label = "Non-Diabetic" if cls == 0 else "Diabetic"
    print(f"   Class {cls} ({label:14s}): {count:5d} samples  ({pct:.1f}%)")

print(f"\n   Total samples: {len(y)}")
print(f"   Imbalance ratio: {raw_counts[0]/raw_counts[1]:.1f}:1  "
      f"(non-diabetic : diabetic)")


# ============================================================
# STEP 2 — Train / Test Split (BEFORE SMOTE)
# ============================================================
# We split the dataset into:
#   Training set (80%) → the model learns from this
#   Test set     (20%) → we evaluate the model on this
#
# ⚠️  CRITICAL RULE: Split BEFORE applying SMOTE!
#
#   WHY? SMOTE creates synthetic (fake) samples by blending
#   real samples together. If we applied SMOTE BEFORE splitting,
#   some of those synthetic samples might be "copies" of real
#   test samples — and the model would essentially have already
#   seen the test data during training. This is called DATA
#   LEAKAGE and makes your evaluation results completely fake.
#
#   The rule is: TEST DATA MUST ALWAYS BE 100% REAL, UNTOUCHED.
#   SMOTE only goes near training data. Never test data.
#
# stratify=y → ensures BOTH splits keep the same 75/25 ratio.
#   Without stratify, random chance could give you 90/10 in test.
# ------------------------------------------------------------

print(f"\n{RULER}")
print("📊 STEP 2 — TRAIN / TEST SPLIT  (80% / 20%)")
print(RULER)
print("   ⚠️  We split BEFORE SMOTE to prevent data leakage.")
print("       Test data must stay real — no synthetic samples.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,       # 20% goes to test set
    random_state=42,      # fixed seed → reproducible splits every run
    stratify=y            # preserve 75/25 class ratio in both halves
)

print(f"\n   X_train shape  : {X_train.shape}  → {X_train.shape[0]} training samples")
print(f"   X_test  shape  : {X_test.shape}   → {X_test.shape[0]} test samples")
print(f"   y_train shape  : {y_train.shape}")
print(f"   y_test  shape  : {y_test.shape}")

# Show class distribution in each split (should mirror original)
train_counts = Counter(y_train)
test_counts  = Counter(y_test)
print(f"\n   Training class 0: {train_counts[0]}  |  class 1: {train_counts[1]}")
print(f"   Test     class 0: {test_counts[0]}   |  class 1: {test_counts[1]}")


# ============================================================
# STEP 3 — Apply SMOTE to Training Data Only
# ============================================================
# SMOTE = Synthetic Minority Over-sampling TEchnique
#
# 🧒 SMOTE explained like you are 10 years old:
#   Imagine you are teaching a kid to recognise cats vs dogs,
#   but you only have 10 cat photos and 30 dog photos.
#   The kid will just learn "everything is a dog!"
#
#   SMOTE is like a SMART PHOTOCOPIER for the rare class.
#   It does not just duplicate (copy-paste) diabetic samples —
#   it BLENDS two nearby diabetic samples together to make a
#   brand-new, realistic synthetic diabetic sample.
#   Now the AI sees 30 real + 20 synthetic diabetics = 50 each.
#   Equal examples → fair, unbiased learning. 🎉
#
# k_neighbors=5 (default): each new point is generated by
#   picking a real minority sample, finding its 5 nearest
#   diabetic neighbours, and interpolating between them.
# ------------------------------------------------------------

print(f"\n{RULER}")
print("📊 STEP 3 — SMOTE  (Training data only)")
print(RULER)
print("   🧒 SMOTE is like a smart photocopier for the rare class.")
print("      It blends nearby diabetic samples to create realistic")
print("      synthetic diabetics — so the AI sees equal examples.")
print(f"\n   Applying SMOTE to training set...")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

after_counts = Counter(y_train_resampled)
new_total    = len(y_train_resampled)

print(f"\n   AFTER SMOTE (training data only):")
for cls in sorted(after_counts):
    count = after_counts[cls]
    pct   = count / new_total * 100
    label = "Non-Diabetic" if cls == 0 else "Diabetic"
    print(f"   Class {cls} ({label:14s}): {count:5d} samples  ({pct:.1f}%)")

print(f"\n   Total training samples after SMOTE: {new_total}")

# Confirm balance
if after_counts[0] == after_counts[1]:
    print("   ✅ Classes are NOW perfectly balanced (50/50) in training set!")
else:
    diff = abs(after_counts[0] - after_counts[1])
    print(f"   ⚠️  Classes still differ by {diff} samples. Check SMOTE settings.")

# Reminder that test data is untouched
print(f"\n   ✔ Test set is UNCHANGED — still {test_counts[0]} vs {test_counts[1]} (real data only).")


# ============================================================
# STEP 4 — Feature Scaling with StandardScaler
# ============================================================
# StandardScaler transforms each feature so that:
#   Mean = 0  and  Standard Deviation = 1
#
# Why scale? Many ML algorithms (especially SVM, neural nets,
# and anything that uses distances) are sensitive to the scale
# of features. If MFCC values range from -100 to +100 but
# Jitter values range from 0 to 0.01, the model will pay far
# more attention to MFCC simply because the numbers are bigger.
# Scaling puts every feature on equal footing.
#
# ⚠️  CRITICAL RULE: FIT on training data ONLY. NEVER on test.
#
#   WHY? If we fit the scaler on test data too, we are leaking
#   information about the test set into our preprocessing.
#   The scaler would "know" the mean and range of test samples,
#   which it should NEVER have seen. This inflates performance.
#
#   Correct workflow:
#     scaler.fit(X_train)         ← learns mean/std from train only
#     scaler.transform(X_train)   ← scales training data
#     scaler.transform(X_test)    ← applies the SAME scale to test
#                                    (using train's mean/std, not test's)
# ------------------------------------------------------------

print(f"\n{RULER}")
print("📊 STEP 4 — FEATURE SCALING  (StandardScaler)")
print(RULER)
print("   ⚠️  Scaler is fitted on X_train ONLY — never on X_test.")
print("       This prevents test-data leakage into preprocessing.")

scaler = StandardScaler()

# FIT the scaler: compute mean and std from training data only
scaler.fit(X_train_resampled)

# TRANSFORM: apply the learned scale to both sets
X_train_scaled = scaler.transform(X_train_resampled)
X_test_scaled  = scaler.transform(X_test)           # uses train's mean/std

print(f"\n   ✅ Scaling complete.")
print(f"   X_train_scaled shape : {X_train_scaled.shape}")
print(f"   X_test_scaled  shape : {X_test_scaled.shape}")
print(f"   Training feature mean (should ≈ 0.00): "
      f"{X_train_scaled.mean():.4f}")
print(f"   Training feature std  (should ≈ 1.00): "
      f"{X_train_scaled.std():.4f}")

# Save the scaler so we can reuse it during inference later
joblib.dump(scaler, "scaler.pkl")
print(f"\n   💾 Scaler saved as: scaler.pkl")
print(f"      (You need this to scale new voice samples before prediction!)")


# ============================================================
# STEP 5 — Save Everything to Disk
# ============================================================
# We save as .npy (NumPy binary format):
#   • Much faster to load than CSV for large arrays
#   • Preserves exact floating-point values without rounding
# ------------------------------------------------------------

print(f"\n{RULER}")
print("📊 STEP 5 — SAVING FILES")
print(RULER)

np.save("X_train.npy", X_train_scaled)
np.save("X_test.npy",  X_test_scaled)
np.save("y_train.npy", y_train_resampled)
np.save("y_test.npy",  y_test)

print("   ✅ Saved X_train.npy   ← scaled, SMOTE-balanced training features")
print("   ✅ Saved X_test.npy    ← scaled, REAL test features (no SMOTE)")
print("   ✅ Saved y_train.npy   ← training labels (balanced after SMOTE)")
print("   ✅ Saved y_test.npy    ← test labels (original distribution)")
print("   ✅ Saved scaler.pkl    ← fitted scaler for future predictions")


# ============================================================
# STEP 6 — Final Summary Table
# ============================================================

print(f"\n{RULER}")
print("📊 STEP 6 — SUMMARY")
print(RULER)

train_class_pct = (
    f"{after_counts[0]/(after_counts[0]+after_counts[1])*100:.0f}/"
    f"{after_counts[1]/(after_counts[0]+after_counts[1])*100:.0f}"
)
test_class_pct = (
    f"{test_counts[0]/(test_counts[0]+test_counts[1])*100:.0f}/"
    f"{test_counts[1]/(test_counts[0]+test_counts[1])*100:.0f}"
)

summary_rows = [
    ("Total samples",          str(n_samples)),
    ("Training (raw)",         str(len(y_train))),
    ("Training (after SMOTE)", str(len(y_train_resampled))),
    ("Test samples",           str(len(y_test))),
    ("Features per sample",    str(n_features)),
    ("Class balance (train)",  train_class_pct),
    ("Class balance (test)",   test_class_pct),
]

W = 38   # box width
print("╔" + "═" * W + "╗")
print("║" + " DATA PREPARATION SUMMARY".center(W) + "║")
print("╠" + "═" * W + "╣")
for label, value in summary_rows:
    row = f" {label} : {value}"
    print("║" + row.ljust(W) + "║")
print("╚" + "═" * W + "╝")


# ============================================================
# STEP 7 — Evaluation Metric Warning
# ============================================================
# This is the most important reminder in the entire pipeline.
# A naive model that always predicts "Non-Diabetic" (class 0)
# will get 75% accuracy on this dataset — and that sounds good!
# But it is completely useless. It will NEVER detect a diabetic
# patient. In medicine, missing a positive case (False Negative)
# is far more dangerous than a False Positive.
#
# Always use these metrics for medical classification:
#   AUC-ROC     → overall discrimination ability of the model
#   Sensitivity → "Of all real diabetics, how many did we catch?"
#                 (also called Recall or True Positive Rate)
#   Specificity → "Of all real non-diabetics, how many did we
#                 correctly clear?" (True Negative Rate)
#   F1-Score    → balance between precision and recall
# ------------------------------------------------------------

print(f"\n{'⚠️ ' * 20}")
print("⚠️  REMINDER:")
print()
print("   Never use ACCURACY ALONE as your evaluation metric!")
print()
print("   Always report:")
print("     • AUC-ROC      — overall model discrimination")
print("     • Sensitivity  — % of diabetics correctly detected")
print("     • Specificity  — % of non-diabetics correctly cleared")
print("     • F1-Score     — balance of precision & recall")
print()
print("   A model that predicts ALL patients as Non-Diabetic")
print("   gets 75% ACCURACY — but catches ZERO diabetics.")
print("   That model is completely USELESS and DANGEROUS.")
print(f"{'⚠️ ' * 20}")
print()
print(RULER)
print("🚀 Data preparation complete! Next: Model Training.")
print(RULER)
