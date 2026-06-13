"""
colab_21_final_evaluation.py
============================
Official evaluation of all 5 diabetes prediction models
on the held-out test set (320 samples, never seen during training).

Upload to Google Colab and run all cells.
Required files in /content/:
  model1_voice_only.pkl, model5M_male.pkl, model5F_female.pkl
  model6M_male_bmi.pkl,  model6F_female_bmi.pkl
  scaler_A.pkl, scaler_male.pkl, scaler_female.pkl
  scaler_maleB.pkl, scaler_femaleB.pkl, scaler_bmi.pkl
  X_test_raw.npy, y_test.npy, X_test_meta.csv

NOTE: Use the scalers from voice_diabetes_app/models/ — they are the
      correct 267-feature versions. If you upload an older scaler fitted
      on 269 features, the script will automatically slice it to match.
"""

import numpy as np
import pandas as pd
import joblib
import copy
import matplotlib
matplotlib.use('Agg')        # prevents display errors in Colab
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, f1_score,
    confusion_matrix, matthews_corrcoef
)

# =====================================================================
# HELPER: safe_transform
# =====================================================================
def safe_transform(scaler, X, scaler_name="scaler"):
    """
    Applies a StandardScaler to X regardless of whether their feature
    counts match.

    Root cause: sklearn's scaler.transform() internally does
        X -= self.mean_
    which raises ValueError if len(mean_) != X.shape[1].
    scaler.n_features_in_ is sometimes unreliable in older serialized
    objects, so we check len(scaler.mean_) — the ACTUAL operating size.

    If scaler has more features than X (e.g., fitted on 269 but X has 267):
      - The extra features are MFCC1 (position 0) and/or AGE (last position)
      - We skip them by taking mean_ and scale_ from position [n_extra:n_scaler-n_trailing]
      - n_extra  = how many features to skip from the front
      - We always take exactly n_X features
    """
    n_scaler = len(scaler.mean_)
    n_X      = X.shape[1]

    # Manually compute StandardScaler: (X - mean) / std
    # This avoids calling scaler.transform() which has the shape check baked in
    if n_scaler == n_X:
        mean  = scaler.mean_
        scale = scaler.scale_

    elif n_scaler > n_X:
        n_extra = n_scaler - n_X   # number of extra features (always skip from front)
        print(f"  [safe_transform] {scaler_name}: fitted on {n_scaler} features, "
              f"X has {n_X}. Skipping first {n_extra} scaler feature(s) "
              f"(e.g. MFCC1, AGE).")
        mean  = scaler.mean_[n_extra : n_extra + n_X]
        scale = scaler.scale_[n_extra : n_extra + n_X]

    else:
        raise ValueError(
            f"{scaler_name}: scaler expects {n_scaler} features but X has {n_X}. "
            "Cannot transform — upload the correct scaler."
        )

    # Replace any zero std values with 1 to avoid division by zero
    scale = np.where(scale == 0, 1.0, scale)

    return (X - mean) / scale



# =====================================================================
# LOAD MODELS AND SCALERS
# =====================================================================
print("=" * 60)
print("Loading models and scalers...")

model1  = joblib.load('/content/model1_voice_only.pkl')
model5M = joblib.load('/content/model5M_male.pkl')
model5F = joblib.load('/content/model5F_female.pkl')
model6M = joblib.load('/content/model6M_male_bmi.pkl')
model6F = joblib.load('/content/model6F_female_bmi.pkl')

scaler_A       = joblib.load('/content/scaler_A.pkl')
scaler_male    = joblib.load('/content/scaler_male.pkl')
scaler_female  = joblib.load('/content/scaler_female.pkl')
scaler_maleB   = joblib.load('/content/scaler_maleB.pkl')
scaler_femaleB = joblib.load('/content/scaler_femaleB.pkl')
scaler_bmi     = joblib.load('/content/scaler_bmi.pkl')

# Print feature counts so we can spot any mismatches immediately
for name, sc in [
    ('scaler_A', scaler_A), ('scaler_male', scaler_male),
    ('scaler_female', scaler_female), ('scaler_maleB', scaler_maleB),
    ('scaler_femaleB', scaler_femaleB), ('scaler_bmi', scaler_bmi)
]:
    print(f"  {name:<16}: fitted on {sc.n_features_in_} features")

print("All models and scalers loaded.")

# =====================================================================
# LOAD TEST DATA
# =====================================================================
# X_test_raw  : raw (unscaled) voice features, shape (320, 267)
# y_test      : true labels — 1=Diabetic, 0=Non-Diabetic
# X_test_meta : BMI and Gender for each test sample
print("\nLoading test data...")

X_test_raw  = np.load('/content/X_test_raw.npy').astype(np.float64)
y_test      = np.load('/content/y_test.npy').astype(int)
X_test_meta = pd.read_csv('/content/X_test_meta.csv')

# Auto-detect gender and BMI column names (handles capitalization differences)
gender_col = next((c for c in X_test_meta.columns if c.lower() in ['gender', 'sex']), None)
bmi_col    = next((c for c in X_test_meta.columns if c.lower() == 'bmi'), None)

if gender_col is None:
    raise ValueError(f"No gender column in X_test_meta. Columns: {X_test_meta.columns.tolist()}")
if bmi_col is None:
    raise ValueError(f"No BMI column in X_test_meta. Columns: {X_test_meta.columns.tolist()}")

# Replace NaNs in voice features with 0 (safety net)
X_test_raw = np.nan_to_num(X_test_raw)

print(f"  X_test_raw shape : {X_test_raw.shape}  (expected (320, 267))")
print(f"  y_test shape     : {y_test.shape}")
print(f"  Diabetic         : {int(np.sum(y_test == 1))}")
print(f"  Non-diabetic     : {int(np.sum(y_test == 0))}")
print(f"  Gender column    : '{gender_col}'")
print(f"  BMI column       : '{bmi_col}'")
print("=" * 60)

# =====================================================================
# STEP 1 — PREPARE SCALED INPUTS FOR EACH MODEL
# =====================================================================
print("\nScaling features for each model...")

# Extract BMI values as a (320, 1) array for the BMI scaler
bmi_values     = X_test_meta[bmi_col].values.reshape(-1, 1)
bmi_scaled_col = scaler_bmi.transform(bmi_values).flatten()  # shape (320,)

# Model 1: scale voice features using scaler_A
X1 = safe_transform(scaler_A, X_test_raw)               # (320, 267)

# Model 2 (Male only): scale voice features using scaler_male
X2 = safe_transform(scaler_male, X_test_raw)             # (320, 267)

# Model 3 (Female only): scale voice features using scaler_female
X3 = safe_transform(scaler_female, X_test_raw)           # (320, 267)

# Model 4 (Male + BMI): voice scaled with scaler_maleB, then BMI appended
voice_m = safe_transform(scaler_maleB, X_test_raw)       # (320, 267)
X4 = np.column_stack([voice_m, bmi_scaled_col])          # (320, 268)

# Model 5 (Female + BMI): voice scaled with scaler_femaleB, then BMI appended
voice_f = safe_transform(scaler_femaleB, X_test_raw)     # (320, 267)
X5 = np.column_stack([voice_f, bmi_scaled_col])          # (320, 268)

print(f"  Model 1: {X1.shape} | Model 2: {X2.shape} | Model 3: {X3.shape}")
print(f"  Model 4: {X4.shape} | Model 5: {X5.shape}")

# =====================================================================
# STEP 2 — EVALUATE EACH MODEL
# =====================================================================

def evaluate_model(model, X, y_true, threshold, name):
    """
    Runs a model on X, applies the threshold to get binary predictions,
    then computes all classification metrics.

    threshold: we use model-specific thresholds (not always 0.5) because
    in medical screening we lower the threshold to catch more diabetics
    (higher sensitivity), even at the cost of more false alarms (FP).
    """
    # Get probability of class 1 (Diabetic) for every test sample
    proba  = model.predict_proba(X)[:, 1]              # shape (320,)

    # Classify as Diabetic if probability >= threshold
    y_pred = (proba >= threshold).astype(int)

    # AUC-ROC: area under the ROC curve. 1.0 = perfect, 0.5 = random
    auc = roc_auc_score(y_true, proba)

    # Accuracy: overall fraction of correct predictions
    acc = accuracy_score(y_true, y_pred)

    # F1-Score: harmonic mean of precision and recall (good for imbalanced data)
    f1  = f1_score(y_true, y_pred, zero_division=0)

    # MCC: Matthews Correlation Coefficient. +1 = perfect, 0 = random
    mcc = matthews_corrcoef(y_true, y_pred)

    # Confusion matrix values
    cm          = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Sensitivity: of all actual diabetics, how many did we catch?
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity: of all healthy people, how many did we correctly clear?
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'name'       : name,
        'threshold'  : threshold,
        'auc'        : round(float(auc),         4),
        'accuracy'   : round(float(acc),         4),
        'sensitivity': round(float(sensitivity), 4),
        'specificity': round(float(specificity), 4),
        'f1'         : round(float(f1),          4),
        'mcc'        : round(float(mcc),         4),
        'tp'         : int(tp),
        'tn'         : int(tn),
        'fp'         : int(fp),
        'fn'         : int(fn),
        'proba'      : proba,
        'y_pred'     : y_pred,
        'cm'         : cm,
    }

print("\nEvaluating models...")
results = [
    evaluate_model(model1,  X1, y_test, 0.7, 'M1 Voice only'),
    evaluate_model(model5M, X2, y_test, 0.6, 'M2 Male only'),
    evaluate_model(model5F, X3, y_test, 0.6, 'M3 Female only'),
    evaluate_model(model6M, X4, y_test, 0.6, 'M4 Male+BMI'),
    evaluate_model(model6F, X5, y_test, 0.6, 'M5 Female+BMI'),
]
print("All models evaluated.")

# =====================================================================
# STEP 3 — PRINT CONFUSION MATRICES
# =====================================================================
print("\n" + "=" * 60)
print("STEP 3 — CONFUSION MATRICES")
print("=" * 60)

for r in results:
    tn, fp, fn, tp = r['tn'], r['fp'], r['fn'], r['tp']
    print(f"\n  {r['name']}  (threshold = {r['threshold']})")
    print(f"                     Predicted")
    print(f"                     NEG      POS")
    print(f"  Actual  NEG  [ {tn:5d} ]  [ {fp:5d} ]")
    print(f"          POS  [ {fn:5d} ]  [ {tp:5d} ]")
    print(f"  → Caught {tp}/80 diabetics | Missed {fn} | False alarms {fp}")

# =====================================================================
# STEP 4 — FINAL COMPARISON TABLE
# =====================================================================
print("\n" + "=" * 60)
print("STEP 4 — FINAL COMPARISON TABLE")
print("=" * 60)

hdr = (f"{'Model':<16} {'AUC':>7} {'Acc':>7} {'Sens':>7} "
       f"{'Spec':>7} {'F1':>7} {'MCC':>7} {'FN':>5} {'FP':>5}")
print(hdr)
print("─" * len(hdr))
for r in results:
    print(
        f"{r['name']:<16} "
        f"{r['auc']:>7.4f} "
        f"{r['accuracy']:>7.4f} "
        f"{r['sensitivity']:>7.4f} "
        f"{r['specificity']:>7.4f} "
        f"{r['f1']:>7.4f} "
        f"{r['mcc']:>7.4f} "
        f"{r['fn']:>5} "
        f"{r['fp']:>5}"
    )

# =====================================================================
# STEP 5 — GENDER BREAKDOWN FOR MODEL 1
# =====================================================================
print("\n" + "=" * 60)
print("STEP 5 — GENDER BREAKDOWN (Model 1 only)")
print("=" * 60)

gender_values = sorted(X_test_meta[gender_col].unique())
m_val = gender_values[-1]   # Male (usually 1)
f_val = gender_values[0]    # Female (usually 0)

for g_label, g_val in [("Males", m_val), ("Females", f_val)]:
    mask  = X_test_meta[gender_col].values == g_val
    X_sub = X1[mask]
    y_sub = y_test[mask]

    if len(y_sub) == 0:
        print(f"  Model 1 on {g_label}: no samples found")
        continue
    if len(np.unique(y_sub)) < 2:
        print(f"  Model 1 on {g_label}: only one class present — AUC undefined")
        continue

    proba_sub  = model1.predict_proba(X_sub)[:, 1]
    auc_sub    = roc_auc_score(y_sub, proba_sub)
    y_pred_sub = (proba_sub >= 0.7).astype(int)

    tp_s = int(np.sum((y_pred_sub == 1) & (y_sub == 1)))
    fn_s = int(np.sum((y_pred_sub == 0) & (y_sub == 1)))
    n_diab = int(np.sum(y_sub == 1))
    sens_s = tp_s / n_diab if n_diab > 0 else 0.0

    print(f"  Model 1 on {g_label:7s}: n={int(mask.sum()):3d} | "
          f"AUC={auc_sub:.4f} | Sensitivity={sens_s:.4f} | "
          f"Caught {tp_s}/{n_diab} diabetics | Missed {fn_s}")

# =====================================================================
# STEP 6 — PLOTS
# =====================================================================
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

# ── Figure 1: ROC Curves ─────────────────────────────────────────────
print("\nGenerating Figure 1: ROC curves...")
fig1, ax1 = plt.subplots(figsize=(8, 6))
for i, r in enumerate(results):
    fpr, tpr, _ = roc_curve(y_test, r['proba'])
    ax1.plot(fpr, tpr, color=COLORS[i], lw=2,
             label=f"{r['name']}  (AUC = {r['auc']:.4f})")
ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.50)')
ax1.set_xlabel('False Positive Rate  (1 – Specificity)', fontsize=12)
ax1.set_ylabel('True Positive Rate  (Sensitivity)', fontsize=12)
ax1.set_title('ROC Curves — All Models on Test Set', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(alpha=0.3)
fig1.tight_layout()
fig1.savefig('/content/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close(fig1)
print("  Saved: /content/roc_curves.png")

# ── Figure 2: Confusion Matrices ─────────────────────────────────────
print("Generating Figure 2: Confusion matrices...")
fig2, axes = plt.subplots(1, 5, figsize=(22, 4))
fig2.suptitle('Confusion Matrices — All Models on Test Set',
              fontsize=14, fontweight='bold', y=1.02)
for i, r in enumerate(results):
    sns.heatmap(
        r['cm'],
        annot=True, fmt='d', cmap='Blues',
        xticklabels=['Pred NEG', 'Pred POS'],
        yticklabels=['Actual NEG', 'Actual POS'],
        ax=axes[i], cbar=False, linewidths=0.5, annot_kws={'size': 14}
    )
    axes[i].set_title(
        f"{r['name']}\nSens={r['sensitivity']:.2f} | Spec={r['specificity']:.2f}",
        fontsize=10
    )
fig2.tight_layout()
fig2.savefig('/content/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print("  Saved: /content/confusion_matrices.png")

# ── Figure 3: Grouped Bar Chart ───────────────────────────────────────
print("Generating Figure 3: Metric comparison...")
metrics_to_plot = ['auc', 'sensitivity', 'specificity', 'f1', 'mcc']
metric_labels   = ['AUC-ROC', 'Sensitivity', 'Specificity', 'F1-Score', 'MCC']
model_names     = [r['name'] for r in results]
values          = np.array([[r[m] for m in metrics_to_plot] for r in results])

x       = np.arange(len(metric_labels))
width   = 0.14
offsets = np.linspace(-(2 * width), 2 * width, len(results))

fig3, ax3 = plt.subplots(figsize=(13, 6))
for i, (r, offset) in enumerate(zip(results, offsets)):
    bars = ax3.bar(x + offset, values[i], width,
                   label=r['name'], color=COLORS[i], alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        if h > 0.02:
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                     f'{h:.2f}', ha='center', va='bottom', fontsize=7)

ax3.set_xlabel('Metric', fontsize=12)
ax3.set_ylabel('Score', fontsize=12)
ax3.set_title('Model Performance Comparison — Test Set', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(metric_labels, fontsize=11)
ax3.set_ylim(-0.05, 1.15)
ax3.legend(loc='upper right', fontsize=9)
ax3.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax3.axhline(0.0, color='black', linewidth=0.5)
ax3.grid(axis='y', alpha=0.3)
fig3.tight_layout()
fig3.savefig('/content/metric_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig3)
print("  Saved: /content/metric_comparison.png")

# =====================================================================
# STEP 7 — FINAL VERDICT
# =====================================================================
best_auc  = max(results, key=lambda r: r['auc'])
best_sens = max(results, key=lambda r: r['sensitivity'])
fewest_fn = min(results, key=lambda r: r['fn'])
best_f1   = max(results, key=lambda r: r['f1'])
best_mcc  = max(results, key=lambda r: r['mcc'])

# Recommended = highest sensitivity (medical priority: catch all diabetics)
recommended = best_sens
reason = (
    f"Highest sensitivity ({recommended['sensitivity']:.4f}). "
    "In medical screening, missing a diabetic (FN) is "
    "more dangerous than a false alarm (FP)."
)

W = 54
print("\n")
print("╔" + "═" * W + "╗")
print("║" + "  FINAL EVALUATION VERDICT  ".center(W) + "║")
print("╠" + "═" * W + "╣")
print(f"║  Best AUC-ROC         : {best_auc['name']:<16} ({best_auc['auc']:.4f})  ║")
print(f"║  Best Sensitivity     : {best_sens['name']:<16} ({best_sens['sensitivity']:.4f})  ║")
print(f"║  Fewest Missed Cases  : {fewest_fn['name']:<16} ({fewest_fn['fn']} FN)      ║")
print(f"║  Best F1-Score        : {best_f1['name']:<16} ({best_f1['f1']:.4f})  ║")
print(f"║  Best MCC             : {best_mcc['name']:<16} ({best_mcc['mcc']:.4f})  ║")
print("╠" + "═" * W + "╣")
print(f"║  RECOMMENDED MODEL    : {recommended['name']:<30}  ║")
# Wrap reason across lines
words, line = reason.split(), "║  REASON : "
for w in words:
    if len(line) + len(w) + 1 <= W + 1:
        line += w + " "
    else:
        print(line.ljust(W + 1) + "║")
        line = "║           " + w + " "
if line.strip() not in ("║", ""):
    print(line.ljust(W + 1) + "║")
print("╠" + "═" * W + "╣")
print(f"║  Total test samples   : {len(y_test):<29}  ║")
print(f"║  Diabetic samples     : {int(np.sum(y_test==1)):<29}  ║")
print(f"║  Non-diabetic samples : {int(np.sum(y_test==0)):<29}  ║")
print(f"║  Test set balanced    : {'No (real distribution)':<29}  ║")
print("╚" + "═" * W + "╝")

print("\nDone. Download from /content/:")
print("  roc_curves.png")
print("  confusion_matrices.png")
print("  metric_comparison.png")
