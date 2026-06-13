import gradio as gr
import numpy as np
import pandas as pd
import joblib
import os
import traceback

# =====================================================================
# SECTION 1 - PATHS AND MODEL LOADING
# =====================================================================
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

def _load(filename):
    """Load a file from models/ folder, raise clear error if missing."""
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    return joblib.load(path)

print("=" * 55)
print("Loading models...")
try:
    model1   = _load('model1_voice_only.pkl')
    model5M  = _load('model5M_male.pkl')
    model5F  = _load('model5F_female.pkl')
    model6M  = _load('model6M_male_bmi.pkl')
    model6F  = _load('model6F_female_bmi.pkl')

    scaler_A      = _load('scaler_A.pkl')
    scaler_male   = _load('scaler_male.pkl')
    scaler_female = _load('scaler_female.pkl')
    scaler_maleB  = _load('scaler_maleB.pkl')
    scaler_femaleB= _load('scaler_femaleB.pkl')
    scaler_bmi    = _load('scaler_bmi.pkl')

    print("All 5 models loaded successfully.")
    print("  Model 1  : Voice only      (267 features, scaler_A)")
    print("  Model 5M : Male only       (267 features, scaler_male)")
    print("  Model 5F : Female only     (267 features, scaler_female)")
    print("  Model 6M : Male + BMI      (268 features, scaler_maleB)")
    print("  Model 6F : Female + BMI    (268 features, scaler_femaleB)")
except Exception as e:
    print(f"FATAL — could not load models: {e}")
    raise

# =====================================================================
# SECTION 2 - TEST DATA LOADING
# =====================================================================
print("\nLoading test data...")
try:
    X_test_raw = np.load(os.path.join(MODELS_DIR, 'X_test_raw.npy'))
    y_test     = np.load(os.path.join(MODELS_DIR, 'y_test.npy'))

    # Load meta file — check for .csv or .npy extension
    meta_csv = os.path.join(MODELS_DIR, 'X_test_meta.csv')
    meta_npy = os.path.join(MODELS_DIR, 'X_test_meta.npy')
    if os.path.exists(meta_csv):
        X_test_meta = pd.read_csv(meta_csv)
    elif os.path.exists(meta_npy):
        X_test_meta = pd.DataFrame(np.load(meta_npy, allow_pickle=True))
    else:
        raise FileNotFoundError(
            "X_test_meta.csv not found in models/. "
            "Run colab_20_save_test_set.py and copy the file here."
        )

    # Print meta column names so we can confirm gender/BMI column names
    print(f"X_test_meta columns: {X_test_meta.columns.tolist()}")

    # Auto-detect gender and BMI column names (case-insensitive)
    gender_col = next((c for c in X_test_meta.columns
                       if c.lower() in ['gender', 'sex']), None)
    bmi_col    = next((c for c in X_test_meta.columns
                       if c.lower() == 'bmi'), None)

    if gender_col is None:
        raise ValueError(f"No gender column found in X_test_meta. Columns: {X_test_meta.columns.tolist()}")
    if bmi_col is None:
        raise ValueError(f"No BMI column found in X_test_meta. Columns: {X_test_meta.columns.tolist()}")

    # Determine male/female encoding values
    gender_values = sorted(X_test_meta[gender_col].unique().tolist())
    m_val = gender_values[-1]   # higher value = Male (convention: Male=1, Female=0)
    f_val = gender_values[0]

    diabetic_count     = int(np.sum(y_test == 1))
    non_diabetic_count = int(np.sum(y_test == 0))
    total              = len(y_test)

    print(f"Test set loaded.")
    print(f"  Total samples : {total}")
    print(f"  Diabetic      : {diabetic_count}")
    print(f"  Non-diabetic  : {non_diabetic_count}")
    print(f"  Gender col    : '{gender_col}'  (Male={m_val}, Female={f_val})")
    print(f"  BMI col       : '{bmi_col}'")
    print(f"  X_test_raw    : {X_test_raw.shape}")
    print("=" * 55)

except Exception as e:
    print(f"FATAL — could not load test data: {e}")
    raise

# =====================================================================
# SECTION 3 - RANDOM SAMPLE LOADER
# =====================================================================
def load_random_sample(gender_filter):
    """
    Picks a random index from the test set matching the filter.
    Filters:
      Any          → any sample
      Male         → samples where gender == m_val
      Female       → samples where gender == f_val
      Diabetic     → samples where y_test == 1
      Non-Diabetic → samples where y_test == 0
    Returns: voice_features (267,), bmi, gender_str, true_label, idx
    """
    if gender_filter == "Male":
        valid_idx = np.where(X_test_meta[gender_col].values == m_val)[0]
    elif gender_filter == "Female":
        valid_idx = np.where(X_test_meta[gender_col].values == f_val)[0]
    elif gender_filter == "Diabetic":
        valid_idx = np.where(y_test == 1)[0]
    elif gender_filter == "Non-Diabetic":
        valid_idx = np.where(y_test == 0)[0]
    else:  # "Any"
        valid_idx = np.arange(len(y_test))

    if len(valid_idx) == 0:
        raise ValueError(f"No samples found for filter: '{gender_filter}'")

    idx = int(np.random.choice(valid_idx))

    voice_features = X_test_raw[idx]                        # shape (267,)
    bmi            = float(X_test_meta[bmi_col].iloc[idx])
    gender_val     = X_test_meta[gender_col].iloc[idx]
    gender_str     = "Male" if gender_val == m_val else "Female"
    true_label     = "DIABETIC" if int(y_test[idx]) == 1 else "NON-DIABETIC"

    return voice_features, bmi, gender_str, true_label, idx


# =====================================================================
# SECTION 4 - PREDICTION FUNCTION
# =====================================================================
def predict_all(voice_features, bmi):
    """
    Runs all 5 models on the given 267-feature voice vector.
    Returns (p1, p2, p3, p4, p5) — raw probabilities.
    """
    vf = np.nan_to_num(voice_features).reshape(1, -1)

    # Model 1 — Voice only (scaler_A, 267 features, threshold 0.7)
    p1 = model1.predict_proba(scaler_A.transform(vf))[0][1]

    # Model 2 — Male only (scaler_male, 267 features, threshold 0.6)
    p2 = model5M.predict_proba(scaler_male.transform(vf))[0][1]

    # Model 3 — Female only (scaler_female, 267 features, threshold 0.6)
    p3 = model5F.predict_proba(scaler_female.transform(vf))[0][1]

    # Model 4 — Male + BMI (scaler_maleB + scaler_bmi, 268 features)
    bmi_scaled = scaler_bmi.transform([[float(bmi)]])[0][0]
    voice_m    = scaler_maleB.transform(vf)[0]
    p4 = model6M.predict_proba(
        np.append(voice_m, bmi_scaled).reshape(1, -1))[0][1]

    # Model 5 — Female + BMI (scaler_femaleB + scaler_bmi, 268 features)
    voice_f = scaler_femaleB.transform(vf)[0]
    p5 = model6F.predict_proba(
        np.append(voice_f, bmi_scaled).reshape(1, -1))[0][1]

    return p1, p2, p3, p4, p5


# =====================================================================
# SECTION 5 - RESULT FORMATTER
# =====================================================================
def format_single(prob, threshold, true_label=None):
    """
    Formats a probability into a readable result box.
    Shows ✅ CORRECT or ❌ WRONG if true_label is provided.
    """
    percentage = f"{prob * 100:.1f}"

    if prob >= threshold:
        risk, emoji, bar = "HIGH RISK",     "⚠️",  "█████████░"
        predicted = "DIABETIC"
    elif prob >= 0.4:
        risk, emoji, bar = "MODERATE RISK", "🔶", "██████░░░░"
        predicted = "DIABETIC"
    else:
        risk, emoji, bar = "LOW RISK",      "✅",  "███░░░░░░░"
        predicted = "NON-DIABETIC"

    if true_label is not None:
        verdict = "✅ CORRECT" if predicted == true_label else "❌ WRONG"
    else:
        verdict = ""

    return (
        f"{emoji} {risk}\n"
        f"{'─'*24}\n"
        f"{bar}\n"
        f"{percentage}%\n"
        f"{'─'*24}\n"
        f"{verdict}\n"
        f"⚕️ Research only."
    )


# =====================================================================
# SECTION 6 - MAIN HANDLER
# =====================================================================
def run_test(gender_filter):
    """Loads a random sample, runs all 5 models, returns formatted results."""
    try:
        voice_features, bmi, gender_str, true_label, idx = \
            load_random_sample(gender_filter)

        p1, p2, p3, p4, p5 = predict_all(voice_features, bmi)

        label_icon = "🔴 DIABETIC" if true_label == "DIABETIC" else "🟢 NON-DIABETIC"

        ground_truth = (
            f"SAMPLE INFO\n"
            f"{'─'*28}\n"
            f"Index      : {idx}\n"
            f"Gender     : {gender_str}\n"
            f"BMI        : {bmi:.1f}\n"
            f"{'─'*28}\n"
            f"TRUE LABEL :\n"
            f"{label_icon}\n"
            f"{'─'*28}\n"
            f"Check ✅/❌ in each model box.\n"
            f"A ❌ on DIABETIC sample = missed case (dangerous).\n"
            f"A ❌ on NON-DIABETIC    = false alarm."
        )

        r1 = format_single(p1, 0.7, true_label)
        r2 = format_single(p2, 0.6, true_label)
        r3 = format_single(p3, 0.6, true_label)
        r4 = format_single(p4, 0.6, true_label)
        r5 = format_single(p5, 0.6, true_label)

        return ground_truth, r1, r2, r3, r4, r5

    except Exception as e:
        err = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        return err, err, err, err, err, err


# =====================================================================
# SECTION 7 - GRADIO INTERFACE
# =====================================================================
with gr.Blocks(title="Model Testing Interface") as demo:

    gr.Markdown("""
    # 🧪 Model Testing Interface
    **Internal Testing Tool — app2.py**
    Tests all 5 models on real unseen data samples.
    Ground truth shown. ✅/❌ verdict per model.
    """)

    # ── Input Row ─────────────────────────────────────────────────────
    with gr.Row():
        gender_filter = gr.Radio(
            choices=["Any", "Male", "Female", "Diabetic", "Non-Diabetic"],
            value="Any",
            label="Filter samples by:"
        )
        test_btn = gr.Button(
            "🎲 Load Random Sample + Test All Models",
            variant="primary",
            size="lg"
        )

    # ── Ground Truth Box ──────────────────────────────────────────────
    ground_box = gr.Textbox(
        label="📋 Sample Info + True Label",
        lines=10
    )

    # ── Results Row ───────────────────────────────────────────────────
    gr.Markdown("## 📊 Model Predictions vs Ground Truth")

    with gr.Row():
        out1 = gr.Textbox(label="Model 1\nVoice Only\nAUC 0.80",    lines=9)
        out2 = gr.Textbox(label="Model 2\nMale Only\nAUC 0.88",     lines=9)
        out3 = gr.Textbox(label="Model 3\nFemale Only\nAUC 0.73",   lines=9)
        out4 = gr.Textbox(label="Model 4\nMale+BMI\nAUC 0.86",      lines=9)
        out5 = gr.Textbox(label="Model 5\nFemale+BMI\nAUC 0.79",    lines=9)

    test_btn.click(
        fn=run_test,
        inputs=[gender_filter],
        outputs=[ground_box, out1, out2, out3, out4, out5]
    )

    gr.Markdown("""
    ---
    **What to look for:**
    Filter = Diabetic → models should show HIGH RISK ⚠️
    Filter = Non-Diabetic → models should show LOW RISK ✅

    Run 10–20 times and count ✅ vs ❌.
    That gives you real sensitivity and specificity on unseen data.
    """)


# =====================================================================
# SECTION 8 - LAUNCH
# =====================================================================
if __name__ == "__main__":
    print("Test app running at: http://localhost:7861")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )
