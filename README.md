# 🩺 Voice Diabetes Screening System

> **Status: Work in Progress — College Research Project**

A machine learning system that predicts diabetes risk from voice recordings using acoustic biomarkers. The system extracts 267 features from voice audio and runs predictions across multiple specialized models.

---

## 📋 Project Overview

Diabetes affects vocal tract muscles and breathing patterns in measurable ways. This project investigates whether those changes can be detected automatically from a short voice recording using ML.

**Key Features:**
- No blood test or medical device required
- 5 prediction models covering voice-only, gender-stratified, and BMI-inclusive variants
- Audio preprocessing pipeline (denoising, trimming, normalization)
- Web app built with Gradio (runs locally)

---

## 📁 Folder Structure

```
Diabetes/
├── data/
│   ├── raw/              ← Original unmodified datasets (CSV from Librosa/OpenSMILE/PRAAT)
│   └── processed/        ← Cleaned dataset used for training, plus gender-stratified pickles
│
├── Colab/                ← Full ML pipeline scripts (colab_01 → colab_19)
│   ├── colab_01_data_inspection.py
│   ├── colab_02_preprocessing.py
│   ├── colab_03_eda.py
│   ├── colab_04_data_preparation.py
│   ├── colab_05_model_training.py
│   ├── colab_06_data_validation.py
│   ├── colab_07_data_fixes.py
│   ├── colab_08_rebuild_versions.py
│   ├── colab_09_final_diagnostic.py
│   ├── colab_10_post_training.py
│   ├── colab_11_gender_confound.py
│   ├── colab_12_final_models_export.py
│   ├── colab_13_pre_deployment_tests.py
│   ├── colab_14_refit_scalers.py
│   ├── colab_15_check_scaling.py
│   ├── colab_16_slice_scalers.py
│   ├── colab_17_feature_order_check.py
│   ├── colab_18_train_new_models.py
│   └── colab_19_train_models_6MF.py
│
└── voice_diabetes_app/   ← Production web application
    ├── app.py            ← Main Gradio app (5 models, single-page comparison)
    ├── app2.py           ← Internal testing tool (dataset random sampling)
    ├── setup_check.py    ← Environment checker before running
    ├── requirements.txt
    ├── models/           ← All trained models and scalers
    └── temp_audio/       ← Cleaned audio temp files (auto-generated, git-ignored)
```

---

## 🧠 Models

| Model | Features | AUC-ROC | Sensitivity |
|-------|----------|---------|-------------|
| Model 1 | Voice only (267) | 0.80 | 0.64 |
| Model 5M | Male patients only (267) | 0.88 | 0.64 |
| Model 5F | Female patients only (267) | 0.73 | 0.41 |
| Model 6M | Male + BMI (268) | ~0.86 | — |
| Model 6F | Female + BMI (268) | ~0.79 | — |

---

## 🚀 Running the App

```bash
cd voice_diabetes_app
pip install -r requirements.txt
python setup_check.py      # Verify all models are present
python app.py              # Main app → http://localhost:7860
python app2.py             # Test app → http://localhost:7861
```

**Tip:** Say "aaah" steadily for 5 seconds when recording.  
After recording, the player may show `00:00` — this is a browser WebM limitation. The audio is captured correctly; just click Analyse.

---

## 🔬 Feature Extraction Pipeline

Each audio file is:
1. **Denoised** — background noise removed using first 0.5s as noise profile
2. **Trimmed** — silence stripped from start/end
3. **Normalized** — peak normalization to uniform loudness
4. **Feature extracted** — 267 features: MFCC×79, Delta×80, Delta2×80, ZCR×2, Spectral features×8, LPC×16, Jitter×1, Shimmer×1

---

## ⚕️ Disclaimer

This is a **research prototype** built as a college project. It is **not a medical device** and must not be used for clinical diagnosis. Always consult a qualified doctor.

---

## 👤 Author

College Project — Machine Learning / Healthcare AI  
*Work in Progress*
