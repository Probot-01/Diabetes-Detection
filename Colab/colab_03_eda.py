# ============================================================
# colab_03_eda.py
# Diabetes Voice Dataset — Exploratory Data Analysis (EDA)
#
# IMPORTANT: Run colab_02_preprocessing.py first!
#   This script expects these files to exist:
#     features.npy       — cleaned feature matrix
#     labels.npy         — label array (0 / 1)
#     feature_names.csv  — column names for every feature
#
# EDA = "Exploratory Data Analysis"
# Before training any model, we LOOK at the data to understand:
#   • Is the dataset balanced?
#   • Do diabetic and non-diabetic patients differ visibly?
#   • Which features are most related to the label?
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RULER = "=" * 60

# ── Global style ──────────────────────────────────────────────
# seaborn "whitegrid" gives clean plots with gridlines.
# font_scale=1.2 makes all text slightly larger — easier to read.
sns.set_theme(style="whitegrid", font_scale=1.2)

# Colour palette used throughout:
#   Green (#4CAF50) = Non-Diabetic  (class 0)
#   Red   (#E53935) = Diabetic      (class 1)
COLORS = {0: "#4CAF50", 1: "#E53935"}
COLOR_LIST = [COLORS[0], COLORS[1]]
LABEL_MAP  = {0: "Non-Diabetic", 1: "Diabetic"}

print(RULER)
print("📊 Starting Exploratory Data Analysis (EDA)")
print(RULER)


# ============================================================
# LOAD DATA
# ============================================================
# np.load() reads the .npy binary files we saved in Script 02.
# pd.read_csv() reads the feature names list.
# We rebuild a full DataFrame so plotting is easy.
# ------------------------------------------------------------

X      = np.load("features.npy")                          # shape: (n_samples, n_features)
y      = np.load("labels.npy")                            # shape: (n_samples,)
names  = pd.read_csv("feature_names.csv")["feature_name"].tolist()

# Combine X and y into one DataFrame for easy slicing
df = pd.DataFrame(X, columns=names)
df["label"] = y

print(f"\n✅ Data loaded successfully.")
print(f"   Samples   : {X.shape[0]}")
print(f"   Features  : {X.shape[1]}")
print(f"   Label col : 0 = Non-Diabetic  |  1 = Diabetic")


# ── Helper: map numeric label to text ─────────────────────────
df["group"] = df["label"].map(LABEL_MAP)    # "Non-Diabetic" / "Diabetic"


# ============================================================
# PLOT 1 — Class Balance Bar Chart
# ============================================================
# Purpose: Understand how many patients are diabetic vs not.
# Why it matters: If one class dominates, the model will be
# biased toward predicting that class and we need special
# handling (e.g. class weights or oversampling with SMOTE).
# ------------------------------------------------------------

print(f"\n{RULER}")
print("📈 PLOT 1 — Class Balance Bar Chart")
print(RULER)

counts    = df["label"].value_counts().sort_index()       # {0: n, 1: m}
bar_labels = [LABEL_MAP[i] for i in counts.index]

fig, ax = plt.subplots(figsize=(6, 5))

bars = ax.bar(
    bar_labels,
    counts.values,
    color=COLOR_LIST,
    edgecolor="black",
    linewidth=0.8,
    width=0.5
)

# Add count + percentage labels on top of each bar
for bar, count in zip(bars, counts.values):
    pct = count / counts.sum() * 100
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + counts.max() * 0.02,
        f"{count}\n({pct:.1f}%)",
        ha="center", va="bottom", fontsize=11, fontweight="bold"
    )

ax.set_title("Class Distribution: Diabetic vs Non-Diabetic",
             fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Number of Patients", fontsize=12)
ax.set_xlabel("Group", fontsize=12)
ax.set_ylim(0, counts.max() * 1.2)
ax.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("eda_class_balance.png", dpi=150, bbox_inches="tight")
plt.show()
print("   ✅ Saved: eda_class_balance.png")


# ============================================================
# PLOT 2 — Age and BMI Distribution by Class
# ============================================================
# Purpose: Check if diabetic patients tend to be older or
# heavier (higher BMI) than non-diabetic patients.
# A boxplot shows: median (middle line), inter-quartile range
# (box), and outlier points beyond the whiskers.
# Side-by-side lets us compare both demographics at once.
# ------------------------------------------------------------

print(f"\n{RULER}")
print("📈 PLOT 2 — Age & BMI Distribution by Class")
print(RULER)

# Check that Age and BMI exist (they come from demographics)
demo_found = [c for c in ["Age", "BMI"] if c in df.columns]
if len(demo_found) < 2:
    print(f"   ⚠️  Expected 'Age' and 'BMI' columns. Found: {demo_found}")
else:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, col in zip(axes, ["Age", "BMI"]):
        sns.boxplot(
            data=df,
            x="group",
            y=col,
            palette={"Non-Diabetic": COLORS[0], "Diabetic": COLORS[1]},
            order=["Non-Diabetic", "Diabetic"],
            width=0.5,
            linewidth=1.5,
            flierprops=dict(marker="o", markersize=4, alpha=0.5),
            ax=ax
        )
        # Overlay individual data points for transparency
        sns.stripplot(
            data=df,
            x="group",
            y=col,
            order=["Non-Diabetic", "Diabetic"],
            color="black",
            alpha=0.25,
            size=3,
            jitter=True,
            ax=ax
        )
        ax.set_title(f"{col} by Diabetes Group", fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(col, fontsize=12)

    fig.suptitle("Demographic Feature Distribution: Diabetic vs Non-Diabetic",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("eda_demographics.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("   ✅ Saved: eda_demographics.png")


# ============================================================
# PLOT 3 — Jitter and Shimmer Comparison
# ============================================================
# Purpose: Jitter and Shimmer are the two most clinically
# important voice features for detecting voice pathology.
#
# Jitter   = cycle-to-cycle variation in pitch (frequency)
#            → High jitter means the voice sounds shaky/unsteady
# Shimmer  = cycle-to-cycle variation in amplitude (loudness)
#            → High shimmer means the voice varies in volume
#
# Research suggests diabetic patients often have higher
# Jitter and Shimmer due to neuropathy affecting vocal muscles.
# ------------------------------------------------------------

print(f"\n{RULER}")
print("📈 PLOT 3 — Jitter & Shimmer Comparison")
print(RULER)

# Find Jitter and Shimmer columns by name
jitter_cols  = [c for c in df.columns if "jitter"  in c.lower()]
shimmer_cols = [c for c in df.columns if "shimmer" in c.lower()]

print(f"   Jitter columns found  : {jitter_cols}")
print(f"   Shimmer columns found : {shimmer_cols}")

if not jitter_cols or not shimmer_cols:
    print("   ⚠️  Jitter or Shimmer columns not found by name.")
    print(f"   Sample column names: {list(df.columns[:10])}")
else:
    # Use the first match if multiple exist
    jitter_col  = jitter_cols[0]
    shimmer_col = shimmer_cols[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, label in zip(axes,
                               [jitter_col, shimmer_col],
                               ["Jitter", "Shimmer"]):
        sns.boxplot(
            data=df,
            x="group",
            y=col,
            palette={"Non-Diabetic": COLORS[0], "Diabetic": COLORS[1]},
            order=["Non-Diabetic", "Diabetic"],
            width=0.5,
            linewidth=1.5,
            flierprops=dict(marker="o", markersize=4, alpha=0.4),
            ax=ax
        )
        sns.stripplot(
            data=df,
            x="group",
            y=col,
            order=["Non-Diabetic", "Diabetic"],
            color="black",
            alpha=0.2,
            size=3,
            jitter=True,
            ax=ax
        )
        ax.set_title(f"{label} by Diabetes Group", fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(f"{label} Value", fontsize=12)

    fig.suptitle("Key Voice Biomarkers: Jitter & Shimmer",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("eda_jitter_shimmer.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("   ✅ Saved: eda_jitter_shimmer.png")


# ============================================================
# PLOT 4 — Correlation Heatmap of Top 20 Features
# ============================================================
# Purpose: Correlation measures how strongly two variables
# move together. Values range from -1 to +1:
#   +1 = as one goes up, the other always goes up
#    0 = no relationship
#   -1 = as one goes up, the other always goes down
#
# We first find the 20 features with the HIGHEST absolute
# correlation with the label column. These are the most
# "predictive" features. Then we plot their correlation
# with EACH OTHER to spot redundant (duplicate) features.
# A heatmap colour-codes each correlation value so patterns
# are immediately visible.
# ------------------------------------------------------------

print(f"\n{RULER}")
print("📈 PLOT 4 — Correlation Heatmap (Top 20 Features)")
print(RULER)

# Step 4a: Compute correlation of every feature with the label
label_corr = df.drop(columns=["group"]).corr()["label"].drop("label")

# Step 4b: Sort by absolute value — we care about strength, not direction
top20_names = label_corr.abs().sort_values(ascending=False).head(20).index.tolist()

print(f"   Top 20 features selected by correlation with label.")

# Step 4c: Build a sub-DataFrame with only those 20 columns + label
top20_df  = df[top20_names + ["label"]]

# Step 4d: Compute the correlation MATRIX (20×20 + label)
corr_matrix = top20_df.corr()

# Step 4e: Plot the heatmap
fig, ax = plt.subplots(figsize=(14, 11))

sns.heatmap(
    corr_matrix,
    annot=True,                   # print correlation number in each cell
    fmt=".2f",                    # 2 decimal places
    cmap="coolwarm",              # blue=negative, red=positive correlation
    center=0,                     # centre the colour scale at 0
    linewidths=0.5,
    linecolor="white",
    square=True,
    cbar_kws={"shrink": 0.8},
    ax=ax
)

ax.set_title("Correlation Heatmap — Top 20 Features vs Label",
             fontsize=14, fontweight="bold", pad=15)

# Rotate x-axis labels for readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
print("   ✅ Saved: eda_correlation.png")


# ============================================================
# PRINTED SUMMARY — Key Clinical Findings
# ============================================================
# After visualising, we summarise the key numbers so you
# have a written record of what the plots showed.
# ------------------------------------------------------------

print(f"\n{RULER}")
print("🔬 CLINICAL FINDINGS SUMMARY")
print(RULER)

# ── Finding 1: Which demographic differs most between groups? ──
print("\n📌 FINDING 1 — Demographic Differences Between Groups:")

if "Age" in df.columns and "BMI" in df.columns:
    for col in ["Age", "BMI"]:
        mean_0 = df[df["label"] == 0][col].mean()
        mean_1 = df[df["label"] == 1][col].mean()
        diff   = abs(mean_1 - mean_0)
        print(f"   {col:4s} | Non-Diabetic avg: {mean_0:.2f}  "
              f"| Diabetic avg: {mean_1:.2f}  | Difference: {diff:.2f}")

    age_diff = abs(df[df["label"]==1]["Age"].mean() - df[df["label"]==0]["Age"].mean())
    bmi_diff = abs(df[df["label"]==1]["BMI"].mean() - df[df["label"]==0]["BMI"].mean())

    bigger = "Age" if age_diff > bmi_diff else "BMI"
    print(f"\n   → '{bigger}' differs MORE between groups.")

# ── Finding 2: Is Jitter higher in diabetic group? ────────────
print("\n📌 FINDING 2 — Jitter: Higher in Diabetic Group?")

if jitter_cols:
    jitter_col = jitter_cols[0]
    jitter_nd  = df[df["label"] == 0][jitter_col].mean()
    jitter_d   = df[df["label"] == 1][jitter_col].mean()
    higher     = "YES ✅" if jitter_d > jitter_nd else "NO ❌"
    print(f"   {jitter_col}")
    print(f"   Non-Diabetic mean : {jitter_nd:.6f}")
    print(f"   Diabetic mean     : {jitter_d:.6f}")
    print(f"   Jitter higher in diabetic group? → {higher}")
else:
    print("   ⚠️  Jitter column not found.")

# ── Finding 3: Is Shimmer higher in diabetic group? ───────────
print("\n📌 FINDING 3 — Shimmer: Higher in Diabetic Group?")

if shimmer_cols:
    shimmer_col = shimmer_cols[0]
    shimmer_nd  = df[df["label"] == 0][shimmer_col].mean()
    shimmer_d   = df[df["label"] == 1][shimmer_col].mean()
    higher      = "YES ✅" if shimmer_d > shimmer_nd else "NO ❌"
    print(f"   {shimmer_col}")
    print(f"   Non-Diabetic mean : {shimmer_nd:.6f}")
    print(f"   Diabetic mean     : {shimmer_d:.6f}")
    print(f"   Shimmer higher in diabetic group? → {higher}")
else:
    print("   ⚠️  Shimmer column not found.")

# ── Finding 4: Top 5 voice features correlated with label ─────
print("\n📌 FINDING 4 — Top 5 Voice Features Correlated with Diabetes Label:")

# Exclude demographic columns from this ranking
demo_cols_excl = ["Age", "BMI", "Gender", "label", "group"]
voice_only_corr = label_corr.drop(
    labels=[c for c in demo_cols_excl if c in label_corr.index],
    errors="ignore"
)
top5 = voice_only_corr.abs().sort_values(ascending=False).head(5)

for rank, (feat, corr_val) in enumerate(top5.items(), start=1):
    direction = "↑ positive" if corr_val > 0 else "↓ negative"
    print(f"   {rank}. {feat:35s}  corr = {corr_val:+.4f}  ({direction})")

# ── Done ──────────────────────────────────────────────────────
print(f"\n{RULER}")
print("✅ EDA COMPLETE — 4 plots saved:")
print("   📊 eda_class_balance.png")
print("   📊 eda_demographics.png")
print("   📊 eda_jitter_shimmer.png")
print("   📊 eda_correlation.png")
print(RULER)
print("🚀 Next step: Feature Scaling & Model Training!")
print(RULER)
