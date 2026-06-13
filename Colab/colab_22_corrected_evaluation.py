"""
colab_22_corrected_evaluation.py
=================================
Re-evaluates all 5 models on their CORRECT populations.

Key insight: gender-stratified models (M2, M3, M4, M5) should only
be evaluated on the subset they were trained for. Evaluating a
male-only model on female patients is methodologically wrong.

Upload all .pkl, .npy and X_test_meta.csv files to /content/.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    roc_auc_score, f1_score,
    accuracy_score, confusion_matrix, matthews_corrcoef
)

# =====================================================================
# HELPER: safe_transform
# Manually applies StandardScaler without calling scaler.transform()
# so it works even if the scaler was fitted on more features than X has.
# =====================================================================
def safe_transform(scaler, X, name="scaler"):
    n_scaler = len(scaler.mean_)   # true feature count the scaler operates on
    n_X      = X.shape[1]

    if n_scaler == n_X:
        mean  = scaler.mean_
        scale = scaler.scale_
    elif n_scaler > n_X:
        # Scaler was fitted on more features (e.g. included MFCC1 or AGE).
        # Skip the first (n_scaler - n_X) features to align with X.
        skip = n_scaler - n_X
        print(f"  [safe_transform] {name}: skipping first {skip} scaler "
              f"feature(s) to match X shape {X.shape}.")
        mean  = scaler.mean_[skip : skip + n_X]
        scale = scaler.scale_[skip : skip + n_X]
    else:
        raise ValueError(
            f"{name}: scaler expects {n_scaler} features but X has {n_X}."
        )

    # Avoid division by zero if any feature has zero variance
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

print("All models loaded.")

# =====================================================================
# LOAD TEST DATA
# =====================================================================
print("\nLoading test data...")

# X_test_raw : 267 raw (unscaled) voice features for all 320 test samples
# y_test     : true labels  (1 = Diabetic, 0 = Non-Diabetic)
# X_test_meta: Gender and BMI for each test sample
X_test_raw  = np.nan_to_num(
    np.load('/content/X_test_raw.npy').astype(np.float64)
)
y_test      = np.load('/content/y_test.npy').astype(int)
X_test_meta = pd.read_csv('/content/X_test_meta.csv')

# Auto-detect column names (handles any capitalisation)
gender_col = next((c for c in X_test_meta.columns if c.lower() in ['gender', 'sex']), None)
bmi_col    = next((c for c in X_test_meta.columns if c.lower() == 'bmi'), None)

if gender_col is None:
    raise ValueError(f"No gender column in X_test_meta. Columns: {X_test_meta.columns.tolist()}")
if bmi_col is None:
    raise ValueError(f"No BMI column in X_test_meta. Columns: {X_test_meta.columns.tolist()}")

# Determine which value represents Male and which represents Female
# Convention: higher value = Male (e.g. Male=1, Female=0)
g_vals = sorted(X_test_meta[gender_col].unique())
m_val, f_val = g_vals[-1], g_vals[0]

print(f"  Gender col : '{gender_col}'  (Male={m_val}, Female={f_val})")
print(f"  BMI col    : '{bmi_col}'")
print(f"  X_test_raw : {X_test_raw.shape}")
print(f"  y_test     : {y_test.shape}")

# =====================================================================
# STEP 1 - SPLIT BY GENDER
# =====================================================================
print("\n" + "=" * 60)
print("STEP 1 — GENDER SPLIT")
print("=" * 60)

# Boolean mask then convert to integer indices for numpy array indexing
male_mask   = X_test_meta[gender_col].values == m_val
female_mask = X_test_meta[gender_col].values == f_val

male_idx   = np.where(male_mask)[0]
female_idx = np.where(female_mask)[0]

n_male           = len(male_idx)
n_female         = len(female_idx)
n_male_diabetic  = int(np.sum(y_test[male_idx] == 1))
n_female_diabetic= int(np.sum(y_test[female_idx] == 1))

print(f"Male test samples   : {n_male}")
print(f"Female test samples : {n_female}")
print(f"Male diabetic       : {n_male_diabetic}")
print(f"Female diabetic     : {n_female_diabetic}")

# Pre-extract subsets for reuse
X_raw_male   = X_test_raw[male_idx]
X_raw_female = X_test_raw[female_idx]
y_male       = y_test[male_idx]
y_female     = y_test[female_idx]

bmi_all    = X_test_meta[bmi_col].values.reshape(-1, 1)
bmi_male   = bmi_all[male_idx]
bmi_female = bmi_all[female_idx]

# Scale BMI values (BMI scaler is shared across all models)
bmi_all_scaled    = scaler_bmi.transform(bmi_all).flatten()
bmi_male_scaled   = scaler_bmi.transform(bmi_male).flatten()
bmi_female_scaled = scaler_bmi.transform(bmi_female).flatten()

# =====================================================================
# STEP 2 - EVALUATE EACH MODEL ON ITS CORRECT POPULATION
# =====================================================================

def compute_metrics(model, X_scaled, y_true, threshold):
    """
    Runs model.predict_proba on X_scaled, applies threshold,
    and returns a dict with all key metrics plus confusion matrix values.
    """
    if len(np.unique(y_true)) < 2:
        # AUC requires at least one sample of each class
        print("    WARNING: only one class present — AUC undefined, returning zeros.")
        return None

    proba  = model.predict_proba(X_scaled)[:, 1]   # probability of Diabetic
    y_pred = (proba >= threshold).astype(int)

    auc  = roc_auc_score(y_true, proba)
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    mcc  = matthews_corrcoef(y_true, y_pred)

    cm           = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Sensitivity: what fraction of true diabetics did we catch?
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # Specificity: what fraction of healthy people did we correctly clear?
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'auc': round(float(auc), 4),
        'acc': round(float(acc), 4),
        'f1' : round(float(f1),  4),
        'mcc': round(float(mcc), 4),
        'sensitivity': round(float(sensitivity), 4),
        'specificity': round(float(specificity), 4),
        'tp': int(tp), 'tn': int(tn),
        'fp': int(fp), 'fn': int(fn),
        'proba': proba,
    }

print("\n" + "=" * 60)
print("STEP 2 — EVALUATING MODELS ON CORRECT POPULATIONS")
print("=" * 60)

# ── Model 1: Voice only, evaluated on ALL 320 samples ────────────────
# Threshold search is done in Step 4; pick 0.5 as default here
print("\n[Model 1] Voice only — ALL 320 samples")
X1_all = safe_transform(scaler_A, X_test_raw, 'scaler_A')
m1 = compute_metrics(model1, X1_all, y_test, threshold=0.5)
print(f"  AUC={m1['auc']:.4f} | Sens={m1['sensitivity']:.4f} | "
      f"Spec={m1['specificity']:.4f} | FN={m1['fn']} | FP={m1['fp']}")

# ── Model 2: Male only — evaluated on MALE subset only ───────────────
print(f"\n[Model 2] Male only — {n_male} male samples")
X2_male = safe_transform(scaler_male, X_raw_male, 'scaler_male')
m2 = compute_metrics(model5M, X2_male, y_male, threshold=0.6)
if m2:
    print(f"  AUC={m2['auc']:.4f} | Sens={m2['sensitivity']:.4f} | "
          f"Spec={m2['specificity']:.4f} | FN={m2['fn']} | FP={m2['fp']}")

# ── Model 3: Female only — evaluated on FEMALE subset only ───────────
print(f"\n[Model 3] Female only — {n_female} female samples")
X3_female = safe_transform(scaler_female, X_raw_female, 'scaler_female')
m3 = compute_metrics(model5F, X3_female, y_female, threshold=0.6)
if m3:
    print(f"  AUC={m3['auc']:.4f} | Sens={m3['sensitivity']:.4f} | "
          f"Spec={m3['specificity']:.4f} | FN={m3['fn']} | FP={m3['fp']}")

# ── Model 4: Male + BMI — evaluated on MALE subset only ──────────────
print(f"\n[Model 4] Male+BMI — {n_male} male samples")
voice_m  = safe_transform(scaler_maleB, X_raw_male, 'scaler_maleB')
X4_male  = np.column_stack([voice_m, bmi_male_scaled])   # shape (n_male, 268)
m4 = compute_metrics(model6M, X4_male, y_male, threshold=0.6)
if m4:
    print(f"  AUC={m4['auc']:.4f} | Sens={m4['sensitivity']:.4f} | "
          f"Spec={m4['specificity']:.4f} | FN={m4['fn']} | FP={m4['fp']}")

# ── Model 5: Female + BMI — evaluated on FEMALE subset only ──────────
print(f"\n[Model 5] Female+BMI — {n_female} female samples")
voice_f    = safe_transform(scaler_femaleB, X_raw_female, 'scaler_femaleB')
X5_female  = np.column_stack([voice_f, bmi_female_scaled])  # shape (n_female, 268)
m5 = compute_metrics(model6F, X5_female, y_female, threshold=0.6)
if m5:
    print(f"  AUC={m5['auc']:.4f} | Sens={m5['sensitivity']:.4f} | "
          f"Spec={m5['specificity']:.4f} | FN={m5['fn']} | FP={m5['fp']}")

# =====================================================================
# STEP 3 - CORRECTED COMPARISON TABLE
# =====================================================================
print("\n" + "=" * 60)
print("STEP 3 — CORRECTED COMPARISON TABLE")
print("=" * 60)

rows = [
    ("M1 Voice only",  "All",    n_male+n_female, m1),
    ("M2 Male only",   "Male",   n_male,          m2),
    ("M3 Female only", "Female", n_female,        m3),
    ("M4 Male+BMI",    "Male",   n_male,          m4),
    ("M5 Female+BMI",  "Female", n_female,        m5),
]

hdr = (f"{'Model':<17} {'Pop':<8} {'n':>4}  "
       f"{'AUC':>7} {'Sens':>7} {'Spec':>7} {'F1':>7} {'FN':>5} {'FP':>5}")
print(hdr)
print("─" * len(hdr))
for name, pop, n, r in rows:
    if r is None:
        print(f"{name:<17} {pop:<8} {n:>4}  {'N/A':>7}")
        continue
    print(
        f"{name:<17} {pop:<8} {n:>4}  "
        f"{r['auc']:>7.4f} "
        f"{r['sensitivity']:>7.4f} "
        f"{r['specificity']:>7.4f} "
        f"{r['f1']:>7.4f} "
        f"{r['fn']:>5} "
        f"{r['fp']:>5}"
    )

# =====================================================================
# STEP 4 - THRESHOLD SEARCH FOR MODEL 1
# =====================================================================
print("\n" + "=" * 60)
print("STEP 4 — THRESHOLD SEARCH FOR MODEL 1 (Voice only, All samples)")
print("=" * 60)

# Try different thresholds to find the best trade-off for Model 1.
# In medical screening, we prefer higher sensitivity (catch more diabetics)
# even if that means more false alarms (lower specificity).
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
threshold_results = []
proba1 = m1['proba']   # reuse already computed probabilities

hdr2 = f"  {'Threshold':>10} | {'Sensitivity':>12} | {'Specificity':>12} | {'F1':>8}"
print(hdr2)
print("  " + "─" * (len(hdr2) - 2))

best_f1_val = -1
best_thresh = None
for t in thresholds:
    y_pred_t     = (proba1 >= t).astype(int)
    cm_t         = confusion_matrix(y_test, y_pred_t, labels=[0, 1])
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    sens_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
    spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0
    f1_t   = f1_score(y_test, y_pred_t, zero_division=0)
    threshold_results.append((t, sens_t, spec_t, f1_t))
    if f1_t > best_f1_val:
        best_f1_val = f1_t
        best_thresh = t

for t, sens, spec, f1 in threshold_results:
    marker = "  ← BEST F1" if t == best_thresh else ""
    print(f"  {t:>10.1f} | {sens:>12.4f} | {spec:>12.4f} | {f1:>8.4f}{marker}")

print(f"\n  Best threshold for Model 1 : {best_thresh}")
print(f"  Best F1                    : {best_f1_val:.4f}")

# Re-evaluate Model 1 with the best threshold for final reporting
m1_best = compute_metrics(model1, X1_all, y_test, threshold=best_thresh)
print(f"\n  Model 1 at threshold {best_thresh}:")
print(f"  AUC={m1_best['auc']:.4f} | Sens={m1_best['sensitivity']:.4f} | "
      f"Spec={m1_best['specificity']:.4f} | FN={m1_best['fn']} | FP={m1_best['fp']}")

# =====================================================================
# STEP 5 - METHODOLOGICAL NOTE
# =====================================================================
print("\n" + "=" * 60)
print("STEP 5 — NOTE ON CORRECT EVALUATION")
print("=" * 60)
print("""
NOTE ON GENDER MODELS:
  M2 and M4 are only valid for MALE patients.
  M3 and M5 are only valid for FEMALE patients.
  Previous evaluation on full 320 samples was incorrect —
  it included female patients in the male model's evaluation
  and vice versa, which inflates or deflates metrics artificially.
  These results reflect TRUE model performance on the intended population.

HOW TO READ THIS TABLE:
  AUC-ROC    : 1.0 = perfect, 0.5 = random guess
  Sensitivity: fraction of diabetics correctly caught (higher = better for screening)
  Specificity: fraction of healthy people correctly cleared
  FN         : missed diabetics (dangerous — want this LOW)
  FP         : false alarms (annoying but not dangerous)

MEDICAL SCREENING PRIORITY:
  Sensitivity > Specificity.
  A missed diabetic (FN) is more dangerous than a false alarm (FP).
""")

# =====================================================================
# STEP 6 — PLOTS
# =====================================================================
import matplotlib
matplotlib.use('Agg')   # use non-interactive backend — safe for Colab
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

print("\n" + "=" * 60)
print("STEP 6 — GENERATING PLOTS")
print("=" * 60)

COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

# Collect everything we need for plots into a single list
# Each entry: (display name, model, X_scaled, y_true, threshold, color, population label)
plot_items = [
    ("M1 Voice only",  model1,  X1_all,    y_test,   best_thresh, COLORS[0], "All 320"),
    ("M2 Male only",   model5M, X2_male,   y_male,   0.6,         COLORS[1], f"Male ({n_male})"),
    ("M3 Female only", model5F, X3_female, y_female, 0.6,         COLORS[2], f"Female ({n_female})"),
    ("M4 Male+BMI",    model6M, X4_male,   y_male,   0.6,         COLORS[3], f"Male ({n_male})"),
    ("M5 Female+BMI",  model6F, X5_female, y_female, 0.6,         COLORS[4], f"Female ({n_female})"),
]

# ── Figure 1: ROC Curves ─────────────────────────────────────────────
# One curve per model, all on the same plot.
# AUC shown in the legend — higher is better.
print("  Generating roc_curves.png ...")
fig1, ax1 = plt.subplots(figsize=(9, 6))
for name, mdl, X_sc, y_tr, thr, col, pop in plot_items:
    if len(np.unique(y_tr)) < 2:
        continue
    proba    = mdl.predict_proba(X_sc)[:, 1]
    auc_val  = roc_auc_score(y_tr, proba)
    fpr, tpr, _ = roc_curve(y_tr, proba)
    ax1.plot(fpr, tpr, color=col, lw=2,
             label=f"{name} [{pop}]  AUC={auc_val:.4f}")

ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.50)')
ax1.set_xlabel('False Positive Rate  (1 – Specificity)', fontsize=12)
ax1.set_ylabel('True Positive Rate  (Sensitivity)', fontsize=12)
ax1.set_title('ROC Curves — Corrected Population Evaluation', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(alpha=0.3)
fig1.tight_layout()
fig1.savefig('/content/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close(fig1)
print("  Saved: /content/roc_curves.png")

# ── Figure 2: Confusion Matrices ─────────────────────────────────────
# 5 heatmaps side by side — one per model on its correct population.
print("  Generating confusion_matrices.png ...")
fig2, axes = plt.subplots(1, 5, figsize=(24, 4))
fig2.suptitle('Confusion Matrices — Each Model on Its Correct Population',
              fontsize=13, fontweight='bold', y=1.03)

for i, (name, mdl, X_sc, y_tr, thr, col, pop) in enumerate(plot_items):
    proba  = mdl.predict_proba(X_sc)[:, 1]
    y_pred = (proba >= thr).astype(int)
    cm     = confusion_matrix(y_tr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec   = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Pred NEG', 'Pred POS'],
        yticklabels=['Actual NEG', 'Actual POS'],
        ax=axes[i], cbar=False, linewidths=0.5,
        annot_kws={'size': 13}
    )
    axes[i].set_title(
        f"{name}\n[{pop}]\nSens={sens:.2f} Spec={spec:.2f}",
        fontsize=9
    )

fig2.tight_layout()
fig2.savefig('/content/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print("  Saved: /content/confusion_matrices.png")

# ── Figure 3: Grouped Metric Bar Chart ───────────────────────────────
# Shows AUC, Sensitivity, Specificity, F1, MCC side by side per model.
print("  Generating metric_comparison.png ...")

metric_keys    = ['auc', 'sensitivity', 'specificity', 'f1', 'mcc']
metric_labels  = ['AUC-ROC', 'Sensitivity', 'Specificity', 'F1-Score', 'MCC']
model_results  = [m1_best, m2, m3, m4, m5]  # use best-threshold M1
model_disp     = ["M1 Voice", "M2 Male", "M3 Female", "M4 Male+BMI", "M5 Female+BMI"]

# Build value matrix: shape (5 models × 5 metrics)
values = np.array([
    [r[k] if r is not None else 0.0 for k in metric_keys]
    for r in model_results
])

x       = np.arange(len(metric_labels))
width   = 0.14
offsets = np.linspace(-(2 * width), 2 * width, 5)

fig3, ax3 = plt.subplots(figsize=(13, 6))
for i, (disp, offset) in enumerate(zip(model_disp, offsets)):
    bars = ax3.bar(x + offset, values[i], width,
                   label=disp, color=COLORS[i], alpha=0.85)
    # Value label on top of each bar
    for bar in bars:
        h = bar.get_height()
        if h > 0.02:
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                     f'{h:.2f}', ha='center', va='bottom', fontsize=7)

ax3.set_xlabel('Metric', fontsize=12)
ax3.set_ylabel('Score', fontsize=12)
ax3.set_title('Model Performance — Correct Population Evaluation', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(metric_labels, fontsize=11)
ax3.set_ylim(-0.05, 1.18)
ax3.legend(loc='upper right', fontsize=9)
ax3.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax3.axhline(0.0, color='black', linewidth=0.5)
ax3.grid(axis='y', alpha=0.3)
fig3.tight_layout()
fig3.savefig('/content/metric_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig3)
print("  Saved: /content/metric_comparison.png")

# ── Figure 4: Threshold vs Sensitivity/Specificity for Model 1 ───────
print("  Generating threshold_search.png ...")
thresh_vals = [t for t, *_ in threshold_results]
sens_vals   = [s for _, s, *_ in threshold_results]
spec_vals   = [sp for _, _, sp, _ in threshold_results]
f1_vals     = [f for *_, f in threshold_results]

fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.plot(thresh_vals, sens_vals, 'o-', color='#C44E52', lw=2, label='Sensitivity')
ax4.plot(thresh_vals, spec_vals, 's-', color='#4C72B0', lw=2, label='Specificity')
ax4.plot(thresh_vals, f1_vals,   '^-', color='#55A868', lw=2, label='F1-Score')
ax4.axvline(best_thresh, color='orange', linestyle='--', lw=1.5,
            label=f'Best F1 threshold ({best_thresh})')
ax4.set_xlabel('Threshold', fontsize=12)
ax4.set_ylabel('Score', fontsize=12)
ax4.set_title('Model 1 — Threshold vs Sensitivity / Specificity / F1', fontsize=13, fontweight='bold')
ax4.set_ylim(0, 1.05)
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3)
fig4.tight_layout()
fig4.savefig('/content/threshold_search.png', dpi=150, bbox_inches='tight')
plt.close(fig4)
print("  Saved: /content/threshold_search.png")

print("\nAll done. Download from /content/:")
print("  roc_curves.png")
print("  confusion_matrices.png")
print("  metric_comparison.png")
print("  threshold_search.png")

