import gradio as gr
import librosa
import numpy as np
import joblib
import parselmouth
import pandas as pd
import noisereduce as nr
import soundfile as sf
import os
import traceback

# =====================================================================
# SECTION 1 - IMPORTS AND STARTUP
# =====================================================================
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

def _load(filename):
    """Load a file from models/ folder, raise clear error if missing."""
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    return joblib.load(path)

try:
    # Voice-only model (267 features)
    model1     = _load('model1_voice_only.pkl')
    scaler_A   = _load('scaler_A.pkl')

    # Gender-stratified voice-only models (267 features each)
    model5M       = _load('model5M_male.pkl')
    scaler_male   = _load('scaler_male.pkl')
    model5F       = _load('model5F_female.pkl')
    scaler_female = _load('scaler_female.pkl')

    # Gender-stratified voice + BMI models (268 features each)
    model6M        = _load('model6M_male_bmi.pkl')
    scaler_maleB   = _load('scaler_maleB.pkl')
    model6F        = _load('model6F_female_bmi.pkl')
    scaler_femaleB = _load('scaler_femaleB.pkl')

    # Shared BMI scaler (1 feature only)
    scaler_bmi = _load('scaler_bmi.pkl')

    print("All 5 models loaded successfully.")
    print("  Model 1  : Voice only                  (267 features)")
    print("  Model 5M : Male voice only              (267 features)")
    print("  Model 5F : Female voice only            (267 features)")
    print("  Model 6M : Male voice + BMI             (268 features)")
    print("  Model 6F : Female voice + BMI           (268 features)")

except FileNotFoundError as e:
    print(f"Failed to load models: {e}")
    raise SystemExit(1)


# =====================================================================
# SECTION 2 - AUDIO CLEANING
# =====================================================================
def clean_audio(audio_path):
    """
    Cleans a raw audio file before feature extraction.
    Steps:
      1. Load audio at 22050 Hz
      2. Remove background noise (uses first 0.5s as noise profile)
      3. Trim silence from start and end
      4. Peak-normalize volume
      5. Save cleaned version to temp_audio/ as a .wav file
      6. Return (clean_path, cleaned_y_array, sample_rate)
    """
    # Step 1 — Load
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    # Step 2 — Noise reduction
    # Use the first 0.5 seconds as a noise reference profile.
    # Assumes recording begins with a brief moment of silence/ambient noise.
    noise_sample_duration = int(sr * 0.5)
    noise_sample = y[:noise_sample_duration] if len(y) > noise_sample_duration else y

    y_denoised = nr.reduce_noise(
        y=y,
        sr=sr,
        y_noise=noise_sample,
        prop_decrease=0.75,
        stationary=False
    )

    # Step 3 — Trim silence
    # top_db=20 cuts anything quieter than 20dB below the peak
    y_trimmed, _ = librosa.effects.trim(y_denoised, top_db=20)

    # Safety fallback: if trimming removed almost everything, keep denoised
    if len(y_trimmed) < sr * 0.5:
        y_trimmed = y_denoised
        print("Warning: trim removed too much audio, using denoised version.")

    # Step 4 — Peak normalization
    # Makes the loudest sample = 1.0 so all recordings are same loudness
    max_val = np.max(np.abs(y_trimmed))
    y_normalized = y_trimmed / max_val if max_val > 0 else y_trimmed

    # Step 5 — Save cleaned audio to temp file
    # parselmouth needs a real file path, not a numpy array
    os.makedirs("temp_audio", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    clean_path = os.path.join("temp_audio", f"cleaned_{base_name}.wav")
    sf.write(clean_path, y_normalized, sr)

    return clean_path, y_normalized, sr


# =====================================================================
# SECTION 3 - FEATURE EXTRACTION
# =====================================================================
def extract_voice_features(audio_path):
    """
    Cleans audio then extracts exactly 267 acoustic features.

    Feature layout:
      MFCC x79        (indices 0-78)    — vocal tract shape
      DELTA x80       (indices 79-158)  — rate of MFCC change
      DELTA2 x80      (indices 159-238) — acceleration of MFCC change
      ZCR x2          (indices 239-240) — signal sign-change rate
      SPEC_CENTROID x2(indices 241-242) — spectral center of mass
      SPEC_BW x2      (indices 243-244) — spectral width
      ROLLOFF x2      (indices 245-246) — 85% energy frequency
      RMS x2          (indices 247-248) — loudness
      LPC x16         (indices 249-264) — vocal tract resonance
      JITTER x1       (index 265)       — pitch stability
      SHIMMER x1      (index 266)       — amplitude stability
    """
    # Step 0 — Clean audio before extracting features
    print(f"Cleaning audio: {audio_path}")
    clean_path, y, sr = clean_audio(audio_path)
    print(f"Audio cleaned. Duration: {len(y)/sr:.2f}s")

    features = []

    # 1. MFCC x79 — extract 80, drop MFCC1 to match training data
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=80)
    features.extend(np.mean(mfcc, axis=1)[1:])          # 79 features

    # 2. DELTA_MFCC x80 — all 80 kept
    delta = librosa.feature.delta(mfcc)
    features.extend(np.mean(delta, axis=1))              # 80 features

    # 3. DELTA2_MFCC x80 — all 80 kept
    delta2 = librosa.feature.delta(mfcc, order=2)
    features.extend(np.mean(delta2, axis=1))             # 80 features

    # 4. ZCR x2
    zcr = librosa.feature.zero_crossing_rate(y=y)
    features.extend([np.mean(zcr), np.std(zcr)])         # 2 features

    # 5. SPEC_CENTROID x2
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.extend([np.mean(spec_cent), np.std(spec_cent)])  # 2 features

    # 6. SPEC_BW x2
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.extend([np.mean(spec_bw), np.std(spec_bw)])  # 2 features

    # 7. ROLLOFF x2
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.extend([np.mean(rolloff), np.std(rolloff)])  # 2 features

    # 8. RMS x2
    rms = librosa.feature.rms(y=y)
    features.extend([np.mean(rms), np.std(rms)])          # 2 features

    # 9. LPC x16 — linear predictive coding
    lpc = librosa.lpc(y=y, order=17)
    features.extend(lpc[1:17])                             # 16 features

    # 10. JITTER x1 — use clean_path so parselmouth reads the saved .wav
    try:
        snd = parselmouth.Sound(clean_path)
        pp  = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter = parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        features.append(float(jitter) if jitter is not None else 0.0)
    except Exception:
        features.append(0.0)

    # 11. SHIMMER x1 — reuse same sound and point process
    try:
        shimmer = parselmouth.praat.call(
            [snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        features.append(float(shimmer) if shimmer is not None else 0.0)
    except Exception:
        features.append(0.0)

    features_arr = np.nan_to_num(np.array(features, dtype=np.float32))

    if features_arr.shape[0] != 267:
        raise ValueError(
            f"Feature extraction produced {features_arr.shape[0]} features — expected 267. "
            "Check your audio file and librosa version."
        )

    return features_arr


# =====================================================================
# SECTION 3 - RESULT FORMATTER
# =====================================================================
def format_single(prob, model_name, threshold):
    """Formats a probability into a human-readable risk card."""
    percentage = round(prob * 100, 1)

    if prob >= threshold:
        risk, emoji = "HIGH RISK", "⚠️"
        advice = "Consult a doctor."
        bar = "█████████░"
    elif prob >= 0.4:
        risk, emoji = "MODERATE RISK", "🔶"
        advice = "Consider a checkup."
        bar = "██████░░░░"
    else:
        risk, emoji = "LOW RISK", "✅"
        advice = "Patterns appear normal."
        bar = "███░░░░░░░"

    return f"""{emoji} {risk}
{'─'*28}
Score : {bar}
        {percentage}%
{'─'*28}
{advice}
⚕️ Research only."""


# =====================================================================
# SECTION 4 - SINGLE UNIFIED PREDICTION FUNCTION
# =====================================================================
def run_all_models(audio, bmi):
    """
    Runs all 5 models on the same audio file.
    Returns 5 formatted result strings simultaneously.
    """
    # Sentinel response — reused for all outputs if audio is missing
    NO_AUDIO = "⚠️ Please upload an audio file."

    if audio is None:
        return NO_AUDIO, NO_AUDIO, NO_AUDIO, NO_AUDIO, NO_AUDIO

    try:
        # Extract voice features ONCE and reuse for all models
        features = extract_voice_features(audio)

        # Scale BMI ONCE — shared by models 6M and 6F
        bmi_scaled = scaler_bmi.transform([[float(bmi)]])[0][0]

        # ----------------------------------------------------------
        # Model 1 — Voice Only (scaler_A, 267 features, threshold 0.7)
        # ----------------------------------------------------------
        scaled_1 = scaler_A.transform(features.reshape(1, -1))
        p1 = model1.predict_proba(scaled_1)[0][1]
        r1 = format_single(p1, "Model 1", threshold=0.7)

        # ----------------------------------------------------------
        # Model 5M — Male Voice Only (scaler_male, 267 features, threshold 0.6)
        # ----------------------------------------------------------
        scaled_2 = scaler_male.transform(features.reshape(1, -1))
        p2 = model5M.predict_proba(scaled_2)[0][1]
        r2 = format_single(p2, "Model 5M", threshold=0.6)

        # ----------------------------------------------------------
        # Model 5F — Female Voice Only (scaler_female, 267 features, threshold 0.6)
        # ----------------------------------------------------------
        scaled_3 = scaler_female.transform(features.reshape(1, -1))
        p3 = model5F.predict_proba(scaled_3)[0][1]
        r3 = format_single(p3, "Model 5F", threshold=0.6)

        # ----------------------------------------------------------
        # Model 6M — Male + BMI (scaler_maleB for voice, scaler_bmi for BMI)
        # Voice and BMI are scaled separately, then concatenated → (268,)
        # ----------------------------------------------------------
        voice_scaled_m = scaler_maleB.transform(features.reshape(1, -1))[0]
        features_m = np.append(voice_scaled_m, bmi_scaled)
        assert features_m.shape == (268,), f"Model 6M: expected 268 features, got {features_m.shape[0]}"
        p4 = model6M.predict_proba(features_m.reshape(1, -1))[0][1]
        r4 = format_single(p4, "Model 6M", threshold=0.6)

        # ----------------------------------------------------------
        # Model 6F — Female + BMI (scaler_femaleB for voice, scaler_bmi for BMI)
        # ----------------------------------------------------------
        voice_scaled_f = scaler_femaleB.transform(features.reshape(1, -1))[0]
        features_f = np.append(voice_scaled_f, bmi_scaled)
        assert features_f.shape == (268,), f"Model 6F: expected 268 features, got {features_f.shape[0]}"
        p5 = model6F.predict_proba(features_f.reshape(1, -1))[0][1]
        r5 = format_single(p5, "Model 6F", threshold=0.6)

        return r1, r2, r3, r4, r5

    except Exception as e:
        err = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        return err, err, err, err, err


# =====================================================================
# SECTION 5 - GRADIO INTERFACE
# =====================================================================
with gr.Blocks(title="Voice Diabetes Screening") as demo:

    gr.Markdown("""
    # 🩺 Voice Diabetes Screening System
    **Research Prototype | College Project**
    Upload your voice once — compare all 5 models instantly.
    > ⚕️ Not a medical device. For research only.
    """)

    # ── Input Row ────────────────────────────────────────────────────
    with gr.Row():
        audio_input = gr.Audio(
            type="filepath",
            label="🎙️ Record or Upload Voice (.wav)",
            sources=["microphone", "upload"],
            scale=2
        )
        gr.Markdown("""
        > ℹ️ **After recording:** The player may show `00:00` — this is normal.  
        > The audio IS captured. Just click **Analyse** to run the models.
        """, scale=2)
        with gr.Column(scale=1):
            bmi_input = gr.Slider(
                minimum=10, maximum=60,
                step=0.1, value=25.0,
                label="BMI (used by Male+BMI and Female+BMI models)"
            )
            gr.Markdown("""
            **BMI = weight(kg) / height(m)²**
            70kg, 1.75m → BMI = 22.9

            **Tip:** Say 'aaah' for 5 seconds.
            """)
            analyse_btn = gr.Button(
                "🔍 Analyse All Models",
                variant="primary",
                size="lg"
            )

    # ── Results Row ──────────────────────────────────────────────────
    gr.Markdown("## 📊 Results — All Models")

    with gr.Row():
        out1 = gr.Textbox(label="Model 1\nVoice Only\nAUC 0.80",    lines=10)
        out2 = gr.Textbox(label="Model 2\nMale Only\nAUC 0.88",     lines=10)
        out3 = gr.Textbox(label="Model 3\nFemale Only\nAUC 0.73",   lines=10)
        out4 = gr.Textbox(label="Model 4\nMale + BMI\nAUC 0.86",    lines=10)
        out5 = gr.Textbox(label="Model 5\nFemale + BMI\nAUC 0.79",  lines=10)

    analyse_btn.click(
        fn=run_all_models,
        inputs=[audio_input, bmi_input],
        outputs=[out1, out2, out3, out4, out5]
    )

    gr.Markdown("""
    ---
    **How to interpret:**
    Use Model 1 for anonymous screening.
    Use Model 2 or 4 if you are male.
    Use Model 3 or 5 if you are female.
    Models 4 and 5 also require your BMI.
    """)


# =====================================================================
# SECTION 6 - LAUNCH
# =====================================================================
if __name__ == "__main__":
    print("App running at: http://localhost:7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
