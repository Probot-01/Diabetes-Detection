# ============================================================
# CELL 1 — Install all required libraries
# Run this cell first. It may take 1–3 minutes.
# ============================================================

# librosa      → Audio analysis: pitch, tempo, MFCCs, spectrograms
# parselmouth  → Praat-based voice analysis (jitter, shimmer, formants)
# noisereduce  → Background noise removal from audio signals
# soundfile    → Read/write audio files (WAV, FLAC, OGG, etc.)
# scikit-learn → Classic ML: SVM, Random Forest, preprocessing, metrics
# xgboost      → Gradient boosting trees — great for tabular/feature data
# shap         → Explain model predictions (feature importance)
# gradio       → Build interactive web demos for your ML models
# pandas       → DataFrames — load, clean, and explore tabular data
# numpy        → Numerical computing: arrays, math operations
# matplotlib   → Core plotting library (line plots, histograms, etc.)
# seaborn      → Statistical data visualization built on matplotlib

!pip install -q \
    librosa \
    praat-parselmouth \
    noisereduce \
    soundfile \
    scikit-learn \
    xgboost \
    shap \
    gradio \
    pandas \
    numpy \
    matplotlib \
    seaborn

print("✅ All packages installed successfully!")


# ============================================================
# CELL 2 — Import all libraries & confirm setup
# Run this cell after Cell 1 finishes.
# ============================================================

# --- Audio Processing ---
import librosa                  # Audio feature extraction (MFCCs, pitch, etc.)
import librosa.display          # Visualizing audio (spectrograms, waveforms)
import parselmouth              # Praat voice analysis (jitter, shimmer)
import noisereduce as nr        # Noise reduction on raw audio
import soundfile as sf          # Load and save audio files

# --- Machine Learning ---
from sklearn.model_selection import train_test_split   # Split data into train/test sets
from sklearn.preprocessing import StandardScaler       # Normalize/scale features
from sklearn.metrics import classification_report      # Evaluate model performance
import xgboost as xgb           # XGBoost gradient boosting classifier/regressor
import shap                     # Explain which features drive predictions

# --- Web Demo ---
import gradio as gr             # Build interactive ML demos with a simple UI

# --- Data Handling ---
import pandas as pd             # DataFrames: load CSVs, filter, group, merge
import numpy as np              # Arrays, matrix math, random numbers

# --- Visualization ---
import matplotlib.pyplot as plt # Core plotting: line, bar, scatter, histogram
import seaborn as sns           # Statistical plots: heatmaps, violin, pair plots

# --- Confirmation ---
print("=" * 40)
print("  Setup complete!")
print("=" * 40)
print(f"  librosa       : {librosa.__version__}")
print(f"  parselmouth   : {parselmouth.__version__}")
print(f"  noisereduce   : {nr.__version__}")
print(f"  soundfile     : {sf.__version__}")
print(f"  scikit-learn  : {__import__('sklearn').__version__}")
print(f"  xgboost       : {xgb.__version__}")
print(f"  shap          : {shap.__version__}")
print(f"  gradio        : {gr.__version__}")
print(f"  pandas        : {pd.__version__}")
print(f"  numpy         : {np.__version__}")
print(f"  matplotlib    : {__import__('matplotlib').__version__}")
print(f"  seaborn       : {sns.__version__}")
print("=" * 40)
