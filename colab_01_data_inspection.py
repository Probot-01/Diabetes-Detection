# ============================================================
# CELL 1 — Upload & Load the Dataset
# ============================================================
# Before running this cell, upload your CSV file to Colab using:
#   Files (left sidebar) → Upload → select your CSV
# Then update FILE_PATH below to match the filename.
# ------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# 👇 Change this to your actual CSV filename
FILE_PATH = "diabetes_voice_dataset.csv"

# Load the CSV into a DataFrame (like an Excel sheet in Python)
df = pd.read_csv(FILE_PATH)

print("=" * 60)
print("✅ Dataset loaded successfully!")
print("=" * 60)


# ============================================================
# CELL 2 — Basic Shape Check
# ============================================================
# df.shape tells us (number of rows, number of columns)
# We expect: (N samples, 309 columns)
# ------------------------------------------------------------

print("\n📐 DATASET SHAPE (rows, columns):")
print(f"   {df.shape[0]} samples  ×  {df.shape[1]} columns")

# Quick sanity check — warn if column count is unexpected
if df.shape[1] != 309:
    print(f"⚠️  WARNING: Expected 309 columns but found {df.shape[1]}. Check your CSV!")
else:
    print("   ✔ Column count matches expected (309)")


# ============================================================
# CELL 3 — Preview the First 5 Rows
# ============================================================
# df.head() shows the first 5 rows so you can visually confirm
# the data loaded correctly and columns look right.
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("👀 FIRST 5 ROWS OF THE DATASET:")
print("=" * 60)
print(df.head().to_string())   # .to_string() prevents column truncation


# ============================================================
# CELL 4 — Class Distribution (Diabetic vs Non-Diabetic)
# ============================================================
# The label column tells us: 0 = Non-Diabetic, 1 = Diabetic
# We check how many samples exist in each class.
# Imbalanced classes can bias the model — we flag it here.
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("🏷️  CLASS DISTRIBUTION (Label Column):")
print("=" * 60)

# Count samples per class
class_counts = df['label'].value_counts().sort_index()
label_names  = {0: "Non-Diabetic (0)", 1: "Diabetic (1)"}

for cls, count in class_counts.items():
    pct = count / len(df) * 100
    print(f"   {label_names.get(cls, cls)}: {count} samples  ({pct:.1f}%)")

total = class_counts.sum()
print(f"\n   Total samples: {total}")


# ============================================================
# CELL 5 — Check for Missing / Null Values
# ============================================================
# Missing values can crash model training or silently skew results.
# We print the count of nulls per column.
# A count of 0 for all columns means we're clean! ✔
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("🔍 MISSING / NULL VALUES PER COLUMN:")
print("=" * 60)

null_counts = df.isnull().sum()
total_nulls = null_counts.sum()

if total_nulls == 0:
    print("   ✔ No missing values found! Dataset is clean.")
else:
    # Only print columns that actually have nulls
    cols_with_nulls = null_counts[null_counts > 0]
    print(f"   ⚠️  {total_nulls} missing value(s) found in {len(cols_with_nulls)} column(s):\n")
    for col, count in cols_with_nulls.items():
        print(f"   {col:30s}  →  {count} missing")


# ============================================================
# CELL 6 — Basic Statistics for Demographic Columns
# ============================================================
# df.describe() gives: count, mean, std, min, 25%, 50%, 75%, max
# We focus on Age, BMI, and BSL (Blood Sugar Level) because
# understanding their ranges helps spot data entry errors.
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("📊 BASIC STATISTICS — Age, BMI, BSL:")
print("=" * 60)

# Select only the three demographic columns we care about
demo_cols = ['Age', 'BMI', 'BSL']

# Check all three columns exist before proceeding
missing_cols = [c for c in demo_cols if c not in df.columns]
if missing_cols:
    print(f"   ⚠️  Column(s) not found in dataset: {missing_cols}")
    print(f"   Available columns: {list(df.columns[-10:])}")  # show last 10 as hint
else:
    print(df[demo_cols].describe().round(2).to_string())


# ============================================================
# CELL 7 — Gender Distribution
# ============================================================
# value_counts() tells us how many Male / Female samples exist.
# Useful to spot if the dataset is heavily skewed by gender,
# which could affect model fairness.
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("⚥  GENDER DISTRIBUTION:")
print("=" * 60)

if 'Gender' not in df.columns:
    print("   ⚠️  'Gender' column not found. Check your column name.")
else:
    gender_counts = df['Gender'].value_counts()
    for gender, count in gender_counts.items():
        pct = count / len(df) * 100
        print(f"   {str(gender):15s}: {count} samples  ({pct:.1f}%)")


# ============================================================
# CELL 8 — Bar Chart: Class Balance
# ============================================================
# A bar chart makes it instantly obvious if one class dominates.
# Ideally both bars should be roughly equal height.
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("📈 PLOTTING CLASS BALANCE BAR CHART...")
print("=" * 60)

fig, ax = plt.subplots(figsize=(6, 4))

bars = ax.bar(
    ["Non-Diabetic (0)", "Diabetic (1)"],
    [class_counts.get(0, 0), class_counts.get(1, 0)],
    color=["#4CAF50", "#F44336"],   # green = healthy, red = diabetic
    edgecolor="black",
    width=0.5
)

# Add count labels on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        str(int(height)),
        ha='center', va='bottom', fontsize=12, fontweight='bold'
    )

ax.set_title("Class Distribution: Diabetic vs Non-Diabetic", fontsize=14, fontweight='bold')
ax.set_ylabel("Number of Samples", fontsize=12)
ax.set_xlabel("Class", fontsize=12)
ax.set_ylim(0, max(class_counts.values) * 1.15)   # headroom for labels
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("class_balance.png", dpi=150)   # saves a copy for your records
plt.show()
print("   Chart saved as 'class_balance.png'")


# ============================================================
# CELL 9 — Imbalance Warning
# ============================================================
# If one class has > 20% more samples than the other,
# the model could learn to just predict the majority class.
# We warn you here so you can apply techniques like SMOTE
# or class_weight='balanced' later.
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("⚖️  CLASS IMBALANCE CHECK:")
print("=" * 60)

count_0 = class_counts.get(0, 0)
count_1 = class_counts.get(1, 0)

if total > 0:
    pct_0 = count_0 / total * 100
    pct_1 = count_1 / total * 100
    difference = abs(pct_0 - pct_1)

    print(f"   Non-Diabetic : {pct_0:.1f}%")
    print(f"   Diabetic     : {pct_1:.1f}%")
    print(f"   Difference   : {difference:.1f}%")

    if difference > 20:
        print(f"\n   ⚠️  WARNING: Classes are imbalanced by {difference:.1f}%!")
        print("   👉 Consider using SMOTE, oversampling, or class_weight='balanced'")
        print("      when training your model to avoid biased predictions.")
    else:
        print(f"\n   ✔ Classes are reasonably balanced (difference = {difference:.1f}%)")

print("\n" + "=" * 60)
print("✅ Data inspection complete! Ready for preprocessing.")
print("=" * 60)
