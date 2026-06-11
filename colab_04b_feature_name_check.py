# ============================================================
# DIAGNOSTIC CELL — Inspect Feature Names
# ============================================================
# Run this BEFORE building feature versions (A / B / C).
# Goal: confirm exactly which demographic columns exist in
# our feature list, and what they are called.
# (Column names matter — a typo like "age" vs "Age" will
#  silently cause the wrong columns to be selected.)
# ============================================================

import pandas as pd

# Load the feature name list saved in colab_02_preprocessing.py
feature_names = pd.read_csv("feature_names.csv")["feature_name"].tolist()

RULER = "=" * 60
print(RULER)
print("🔍 FEATURE NAME DIAGNOSTIC")
print(RULER)

# ── 1. Total feature count ────────────────────────────────────
print(f"\n📌 TOTAL FEATURES: {len(feature_names)}")

# ── 2. Search for each keyword (case-insensitive) ─────────────
keywords = ["age", "bmi", "gender", "bsl"]

print(f"\n{'─'*60}")
print("📌 DEMOGRAPHIC KEYWORD SEARCH:")
print(f"{'─'*60}")

for kw in keywords:
    matches = [col for col in feature_names if kw in col.lower()]
    if matches:
        print(f"\n   '{kw.upper()}' → {len(matches)} match(es) found:")
        for m in matches:
            print(f"      ✅  '{m}'")
    else:
        print(f"\n   '{kw.upper()}' → ❌  No matches found.")
        print(f"      (This column may have been dropped already,")
        print(f"       or it uses a different name in your CSV.)")

# ── 3. Last 10 column names ───────────────────────────────────
print(f"\n{'─'*60}")
print("📌 LAST 10 COLUMN NAMES IN FEATURE LIST:")
print(f"{'─'*60}")
for i, name in enumerate(feature_names[-10:], start=len(feature_names)-9):
    print(f"   [{i:3d}]  {name}")

# ── 4. First 5 column names (sanity check) ────────────────────
print(f"\n{'─'*60}")
print("📌 FIRST 5 COLUMN NAMES (sanity check):")
print(f"{'─'*60}")
for i, name in enumerate(feature_names[:5], start=1):
    print(f"   [{i:3d}]  {name}")

print(f"\n{RULER}")
print("✅ Diagnostic complete.")
print("   Use the exact names above when building versions A / B / C.")
print(RULER)
