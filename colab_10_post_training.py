import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier

def main():
    print("=" * 60)
    print("🧠 POST-TRAINING SHAP EXPLAINABILITY")
    print("=" * 60)
    
    # ---------------------------------------------------------
    # STEP 1 - Save best model
    # ---------------------------------------------------------
    print("\nSTEP 1 - Retraining Best Model (XGBoost on Version B)...")
    
    try:
        X_train_B = np.load('X_train_B.npy', allow_pickle=True).astype(np.float32)
        X_test_B  = np.load('X_test_B.npy', allow_pickle=True).astype(np.float32)
        y_train_B = np.load('y_train_B.npy', allow_pickle=True).astype(int)
        y_test    = np.load('y_test.npy', allow_pickle=True).astype(int)
    except Exception as e:
        print(f"Error loading arrays: {e}")
        return
        
    model = XGBClassifier(
        n_estimators=100, 
        random_state=42,
        scale_pos_weight=3, 
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    model.fit(X_train_B, y_train_B)
    joblib.dump(model, "best_model.pkl")
    print("Best model saved.")

    # ---------------------------------------------------------
    # STEP 2 - SHAP explainability
    # ---------------------------------------------------------
    print("\nSTEP 2 - Generating SHAP explainability plots...")
    
    # 2.1 Load and align feature names
    df_feats = pd.read_csv('feature_names.csv')
    if 'feature_name' in df_feats.columns:
        features = df_feats['feature_name'].tolist()
    else:
        df_feats = pd.read_csv('feature_names.csv', header=None)
        features = df_feats.iloc[:, 0].tolist()
        
    # Strip any potential garbage headers left over
    features = [f for f in features if str(f).strip().lower() not in [
        'feature_name', '0', 'unnamed: 0', 'is_synthetic', '', 'nan'
    ]]
    
    # Align front of the list with target size just in case (as done in diagnostic)
    target_C_size = np.load('X_train.npy', allow_pickle=True).shape[1]
    while len(features) > target_C_size:
        features = features[1:]
        
    # Drop Age since Version B drops Age
    def is_age(f):
        f_lower = str(f).lower()
        return ('age(in years)' in f_lower or f_lower == 'age' or 
                'age(' in f_lower or f_lower.startswith('age_') or f_lower.endswith('_age'))
        
    features_B = [f for f in features if not is_age(f)]
    
    # Safety slice if length mismatch occurs
    if len(features_B) != X_train_B.shape[1]:
        features_B = features_B[:X_train_B.shape[1]]
        
    # 2.2 Create TreeExplainer
    print("Calculating SHAP values (this may take a moment)...")
    X_test_B_df = pd.DataFrame(X_test_B, columns=features_B)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_B_df)
    
    # 2.3 Generate plots
    print("Saving Plot 1 -> shap_bar.png")
    plt.figure()
    if hasattr(shap.plots, 'bar'):
        shap.plots.bar(shap_values, max_display=20, show=False)
    else:
        shap.summary_plot(shap_values.values, X_test_B_df, plot_type="bar", max_display=20, show=False)
    fig = plt.gcf()
    fig.suptitle("Top 20 Features — Voice Diabetes Model", fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saving Plot 2 -> shap_beeswarm.png")
    plt.figure()
    if hasattr(shap.plots, 'beeswarm'):
        shap.plots.beeswarm(shap_values, max_display=20, show=False)
    else:
        shap.summary_plot(shap_values.values, X_test_B_df, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig('shap_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saving Plot 3 -> shap_single.png")
    diabetic_idx = np.where(y_test == 1)[0]
    if len(diabetic_idx) > 0:
        first_diabetic = diabetic_idx[0]
        plt.figure()
        if hasattr(shap.plots, 'waterfall'):
            shap.plots.waterfall(shap_values[first_diabetic], show=False)
            fig = plt.gcf()
            fig.suptitle("Why did the model predict DIABETIC\nfor this patient?", fontsize=14)
            plt.tight_layout()
            plt.savefig('shap_single.png', dpi=300, bbox_inches='tight')
        else:
            shap.plots.force(shap_values[first_diabetic], show=False, matplotlib=True)
            fig = plt.gcf()
            fig.suptitle("Why did the model predict DIABETIC for this patient?", fontsize=14)
            plt.savefig('shap_single.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("No diabetic patients found in test set to plot.")

    # ---------------------------------------------------------
    # STEP 3 - Print top 10 features by SHAP importance
    # ---------------------------------------------------------
    print("\nSTEP 3 - Top 10 Features by SHAP Importance:")
    
    # Calculate mean absolute SHAP values across all test samples
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_shap)[::-1]
    
    top_10_feats = []
    for rank in range(10):
        idx = top_indices[rank]
        feat_name = features_B[idx]
        imp = mean_shap[idx]
        top_10_feats.append(feat_name)
        print(f"  Rank {rank+1:<2}: {feat_name:<16} importance = {imp:.4f}")

    # ---------------------------------------------------------
    # STEP 4 - Print research insight
    # ---------------------------------------------------------
    print("\nSTEP 4 - Research Insight:")
    demographics = ['bmi', 'gender', 'sex', 'weight', 'height']
    
    voice_count = 0
    demo_count = 0
    
    for f in top_10_feats:
        f_lower = str(f).lower()
        if any(d in f_lower for d in demographics):
            demo_count += 1
        else:
            voice_count += 1
            
    print(f"Voice features in top 10: {voice_count}/10")
    print(f"Demographic features in top 10: {demo_count}/10")
    print("=" * 60)

if __name__ == "__main__":
    main()
