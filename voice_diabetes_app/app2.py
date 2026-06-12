import gradio as gr
import numpy as np
import pandas as pd
import joblib
import os
import traceback

# =====================================================================
# SECTION 1 - STARTUP: LOAD ALL MODELS AND DATASET
# =====================================================================
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Look for dataset.csv in the new data/processed/ folder first, then fall back
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'dataset.csv')
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset.csv')
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = os.path.join(os.path.dirname(__file__), 'dataset.csv')

def _load(filename):
    """Load a joblib file from models/ folder with a clear error if missing."""
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    return joblib.load(path)

try:
    model1         = _load('model1_voice_only.pkl')
    scaler_A       = _load('scaler_A.pkl')

    model5M        = _load('model5M_male.pkl')
    scaler_male    = _load('scaler_male.pkl')

    model5F        = _load('model5F_female.pkl')
    scaler_female  = _load('scaler_female.pkl')

    model6M        = _load('model6M_male_bmi.pkl')
    scaler_maleB   = _load('scaler_maleB.pkl')

    model6F        = _load('model6F_female_bmi.pkl')
    scaler_femaleB = _load('scaler_femaleB.pkl')

    scaler_bmi     = _load('scaler_bmi.pkl')

    feats_A = pd.read_csv(os.path.join(MODELS_DIR, 'feature_names_A.csv')).iloc[:, 0].str.strip().tolist()

    print("All 5 models loaded successfully.")
    print("  Model 1  : Voice only      (267 features, scaler_A)")
    print("  Model 5M : Male only       (267 features, scaler_male)")
    print("  Model 5F : Female only     (267 features, scaler_female)")
    print("  Model 6M : Male + BMI      (268 features, scaler_maleB)")
    print("  Model 6F : Female + BMI    (268 features, scaler_femaleB)")

except FileNotFoundError as e:
    print(f"Failed to load models: {e}")
    raise SystemExit(1)

# ── Load dataset ─────────────────────────────────────────────────────
try:
    df_full = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded. Total samples: {len(df_full)}")
except Exception as e:
    print(f"Could not load dataset: {e}")
    raise SystemExit(1)

# Dynamically find exact column names regardless of capitalisation
gender_col = next((c for c in df_full.columns if c.lower() in ['gender', 'sex']), None)
label_col  = next((c for c in df_full.columns if c.lower() in ['label', 'class']), None)
bmi_col    = next((c for c in df_full.columns if c.lower() == 'bmi'), None)

# Only keep voice feature columns that exist in the dataset
voice_cols = [f for f in feats_A if f in df_full.columns]
print(f"Voice feature columns matched in dataset: {len(voice_cols)} (expected 267)")

# Auto-detect the male / female indicator values in the gender column
m_val, f_val = None, None
for g in df_full[gender_col].unique():
    if isinstance(g, str):
        if 'm' in g.lower() and 'f' not in g.lower():
            m_val = g
        elif 'f' in g.lower():
            f_val = g
    elif g == 1:
        m_val = 1
    elif g == 0:
        f_val = 0

if m_val is None: m_val = 1
if f_val is None: f_val = 0
print(f"Gender mapping — Male: '{m_val}', Female: '{f_val}'")


# =====================================================================
# SECTION 2 - RANDOM SAMPLE LOADER
# =====================================================================
def load_random_sample(gender_filter, label_filter):
    """
    Picks a random row from the dataset matching both the gender filter
    and the diabetic/non-diabetic label filter.
    Returns raw unscaled voice features, BMI, gender string, true label, and row index.
    """
    pool = df_full.copy()

    # Apply gender filter
    if gender_filter == "Male":
        pool = pool[pool[gender_col] == m_val]
    elif gender_filter == "Female":
        pool = pool[pool[gender_col] == f_val]

    # Apply label filter
    if label_filter == "Diabetic":
        pool = pool[pool[label_col] == 1]
    elif label_filter == "Non-Diabetic":
        pool = pool[pool[label_col] == 0]

    if len(pool) == 0:
        raise ValueError(
            f"No samples found for filter: Gender='{gender_filter}', Label='{label_filter}'. "
            "Try a less restrictive combination."
        )

    row = pool.sample(1, random_state=None).iloc[0]
    sample_index = int(row.name)

    # Extract raw voice features as a numpy array (267,)
    voice_features = row[voice_cols].values.astype(np.float32)

    bmi = float(row[bmi_col]) if bmi_col else 25.0

    # Map the raw gender value back to a human-readable string
    gender_str = "Male" if row[gender_col] == m_val else "Female"

    # Map label to human-readable string
    raw_label = row[label_col]
    true_label = "DIABETIC" if int(raw_label) == 1 else "NON-DIABETIC"

    return voice_features, bmi, gender_str, true_label, sample_index


# =====================================================================
# SECTION 3 - PREDICTION ON RAW FEATURES
# =====================================================================
def predict_all_from_features(voice_features, bmi):
    """
    Accepts raw unscaled voice features (267,) and a float BMI.
    Applies the appropriate scaler per model, then runs prediction.
    Returns five probabilities: p1 through p5.
    """
    # Safety — replace NaN or Inf values that might be in the dataset
    voice_features = np.nan_to_num(voice_features.astype(np.float32))

    # Model 1 — Voice Only
    p1 = model1.predict_proba(
        scaler_A.transform(voice_features.reshape(1, -1))
    )[0][1]

    # Model 5M — Male Voice Only
    p2 = model5M.predict_proba(
        scaler_male.transform(voice_features.reshape(1, -1))
    )[0][1]

    # Model 5F — Female Voice Only
    p3 = model5F.predict_proba(
        scaler_female.transform(voice_features.reshape(1, -1))
    )[0][1]

    # Model 6M — Male + BMI
    # Voice and BMI must be scaled SEPARATELY then concatenated
    voice_scaled_m = scaler_maleB.transform(voice_features.reshape(1, -1))[0]
    bmi_scaled     = scaler_bmi.transform([[float(bmi)]])[0][0]
    features_m     = np.append(voice_scaled_m, bmi_scaled)   # shape (268,)
    p4 = model6M.predict_proba(features_m.reshape(1, -1))[0][1]

    # Model 6F — Female + BMI
    voice_scaled_f = scaler_femaleB.transform(voice_features.reshape(1, -1))[0]
    features_f     = np.append(voice_scaled_f, bmi_scaled)   # shape (268,)
    p5 = model6F.predict_proba(features_f.reshape(1, -1))[0][1]

    return float(p1), float(p2), float(p3), float(p4), float(p5)


# =====================================================================
# SECTION 4 - RESULT FORMATTER
# =====================================================================
def format_single(prob, threshold):
    """Converts a raw probability into a compact risk card string."""
    percentage = f"{prob * 100:.1f}"

    if prob >= threshold:
        risk, emoji = "HIGH RISK", "⚠️"
        bar = "█████████░"
    elif prob >= 0.4:
        risk, emoji = "MODERATE RISK", "🔶"
        bar = "██████░░░░"
    else:
        risk, emoji = "LOW RISK", "✅"
        bar = "███░░░░░░░"

    return f"""{emoji} {risk}
{'─'*24}
{bar}
{percentage}%
{'─'*24}
⚕️ Research only."""


# =====================================================================
# SECTION 5 - MAIN HANDLER
# =====================================================================
def run_test(gender_filter, label_filter):
    """
    Loads a random sample matching both filters, runs all 5 models on it, and returns
    the ground truth info plus 5 formatted result strings.
    """
    try:
        voice_features, bmi, gender, true_label, sample_index = \
            load_random_sample(gender_filter, label_filter)

        p1, p2, p3, p4, p5 = predict_all_from_features(voice_features, bmi)

        label_display = "🔴  DIABETIC" if true_label == "DIABETIC" else "🟢  NON-DIABETIC"

        ground_truth = (
            f"SAMPLE INFO\n"
            f"{'─'*24}\n"
            f"Index  : {sample_index}\n"
            f"Gender : {gender}\n"
            f"BMI    : {bmi:.1f}\n"
            f"\n"
            f"TRUE LABEL:\n"
            f"{label_display}\n"
            f"{'─'*24}\n"
            f"Use this to check if models are correct."
        )

        r1 = format_single(p1, threshold=0.7)
        r2 = format_single(p2, threshold=0.6)
        r3 = format_single(p3, threshold=0.6)
        r4 = format_single(p4, threshold=0.6)
        r5 = format_single(p5, threshold=0.6)

        return ground_truth, r1, r2, r3, r4, r5

    except Exception as e:
        err = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        return err, err, err, err, err, err


# =====================================================================
# SECTION 6 - GRADIO INTERFACE
# =====================================================================
with gr.Blocks(title="Model Testing Interface") as demo:

    gr.Markdown("""
    # 🧪 Model Testing Interface
    **Internal Testing Tool — app2.py**
    Load random samples from dataset and test all 5 models.
    Ground truth label shown for accuracy verification.
    """)

    # ── Input Row ─────────────────────────────────────────────────────
    with gr.Row():
        gender_filter = gr.Radio(
            choices=["Any", "Male", "Female"],
            value="Any",
            label="Filter by Gender"
        )
        label_filter = gr.Radio(
            choices=["Any", "Diabetic", "Non-Diabetic"],
            value="Any",
            label="Filter by True Label"
        )
        test_btn = gr.Button(
            "🎲 Load Random Sample + Run All Models",
            variant="primary",
            size="lg"
        )

    # ── Ground Truth Box ──────────────────────────────────────────────
    ground_truth_box = gr.Textbox(
        label="📋 Sample Info + True Label",
        lines=10
    )

    # ── Results Row ───────────────────────────────────────────────────
    gr.Markdown("## Model Predictions")

    with gr.Row():
        out1 = gr.Textbox(label="Model 1\nVoice Only\nAUC 0.80",    lines=8)
        out2 = gr.Textbox(label="Model 2\nMale Only\nAUC 0.88",     lines=8)
        out3 = gr.Textbox(label="Model 3\nFemale Only\nAUC 0.73",   lines=8)
        out4 = gr.Textbox(label="Model 4\nMale+BMI\nAUC 0.86",      lines=8)
        out5 = gr.Textbox(label="Model 5\nFemale+BMI\nAUC 0.79",    lines=8)

    test_btn.click(
        fn=run_test,
        inputs=[gender_filter, label_filter],
        outputs=[ground_truth_box, out1, out2, out3, out4, out5]
    )

    gr.Markdown("""
    ---
    **How to use:**
    1. Select gender filter or leave as Any
    2. Click the button to load a random sample
    3. Check if model predictions match True Label
    4. Click again for a new random sample

    **What to look for:**
    If True Label = DIABETIC and models show LOW RISK → missed case (bad for medical screening)
    If True Label = NON-DIABETIC and models show HIGH RISK → false alarm
    If all models agree with True Label → correct prediction
    """)


# =====================================================================
# SECTION 7 - LAUNCH
# =====================================================================
if __name__ == "__main__":
    print("Test app running at: http://localhost:7861")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )
