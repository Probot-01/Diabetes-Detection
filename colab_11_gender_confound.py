import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("🔍 GENDER CONFOUNDER ANALYSIS")
    print("=" * 60)

    # ---------------------------------------------------------
    # STEP 1 - Load Original CSV
    # ---------------------------------------------------------
    try:
        df = pd.read_csv('diabetes_voice_dataset.csv')
    except Exception as e:
        print(f"Failed to load diabetes_voice_dataset.csv: {e}")
        return

    # Dynamically find the correct columns in case of capitalization differences
    col_gender = [c for c in df.columns if 'gender' in str(c).lower() or 'sex' in str(c).lower()]
    col_label = [c for c in df.columns if 'label' in str(c).lower() or 'class' in str(c).lower() or 'diabet' in str(c).lower()]
    
    if not col_gender or not col_label:
        print("Error: Could not automatically locate Gender or Label columns in the dataset.")
        print(f"Available columns: {list(df.columns)}")
        return
        
    col_gender = col_gender[0]
    col_label = col_label[-1]  # The label is usually the last matching column
    
    # ---------------------------------------------------------
    # STEP 2 - Cross-tabulate Gender vs Label
    # ---------------------------------------------------------
    print("\n[STEP 2] Cross-tabulation (Gender vs Label):\n")
    
    # Clean any NA rows to ensure accurate percentages
    df_clean = df.dropna(subset=[col_gender, col_label])
    
    # Generate the crosstab
    ct = pd.crosstab(df_clean[col_gender], df_clean[col_label])
    
    # Check if Label contains exactly two distinct classes
    if len(ct.columns) != 2:
        print(f"Error: Label column does not have exactly 2 classes. Found: {list(ct.columns)}")
        return

    # Rename columns to strictly Non-Diabetic and Diabetic
    # Assuming 0/lower is Non-Diabetic, 1/higher is Diabetic
    ct.columns = ['Non-Diabetic', 'Diabetic']
    
    # Standardize Gender index capitalization for aesthetics
    ct.index = [str(idx).capitalize() for idx in ct.index]
    
    print(ct.to_string())
    print("\n" + "-" * 60)

    # Calculate absolute columns
    diabetic_col = ct['Diabetic']
    non_diabetic_col = ct['Non-Diabetic']
    
    diabetic_total = diabetic_col.sum()
    non_diabetic_total = non_diabetic_col.sum()
    
    if diabetic_total == 0 or non_diabetic_total == 0:
        print("Error: Not enough data in one of the label classes to calculate percentages.")
        return
        
    diabetic_pct = (diabetic_col / diabetic_total) * 100
    non_diabetic_pct = (non_diabetic_col / non_diabetic_total) * 100
    
    # ---------------------------------------------------------
    # STEP 3 - Diabetic breakdown
    # ---------------------------------------------------------
    print("\n[STEP 3] Percentage of Diabetic patients by Gender:")
    for gender, pct in diabetic_pct.items():
        print(f"  - {gender:<8}: {pct:.1f}%")
        
    # ---------------------------------------------------------
    # STEP 4 - Non-Diabetic breakdown
    # ---------------------------------------------------------
    print("\n[STEP 4] Percentage of Non-Diabetic patients by Gender:")
    for gender, pct in non_diabetic_pct.items():
        print(f"  - {gender:<8}: {pct:.1f}%")
        
    print("\n" + "-" * 60)
        
    # ---------------------------------------------------------
    # STEP 5 - Confound Logic Check
    # ---------------------------------------------------------
    print("\n[STEP 5] Confound Analysis:")
    
    # Calculate the maximum difference between the two groups for any gender
    max_diff = 0
    for gender in ct.index:
        diff = abs(diabetic_pct[gender] - non_diabetic_pct[gender])
        if diff > max_diff:
            max_diff = diff
            
    print(f"Maximum class difference: {max_diff:.1f}%\n")
    
    if max_diff > 20.0:
        print("⚠️" * 25)
        print("WARNING: Gender is a confound in this dataset.")
        print("Remove it from training features.")
        print("⚠️" * 25)
    else:
        print("✅ Gender distribution looks reasonably balanced. No severe >20% confound detected.")

if __name__ == "__main__":
    main()
