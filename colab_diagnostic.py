import os

def get_file_info(filepath):
    if not os.path.exists(filepath):
        return "not found"
    
    try:
        if filepath.endswith('.csv'):
            import pandas as pd
            # Use nrows=0 to just read columns/shape without loading the whole file if it's huge, 
            # but to get full shape we need to read it. For diagnostic, reading it is usually fine.
            df = pd.read_csv(filepath)
            return f"shape: {df.shape}"
        elif filepath.endswith('.npy'):
            import numpy as np
            arr = np.load(filepath, allow_pickle=True)
            return f"shape: {arr.shape}"
        else:
            size = os.path.getsize(filepath)
            return f"size: {size} bytes"
    except Exception as e:
        return f"error reading: {str(e)}"

def check_status():
    files_to_check = [
        'dataset.csv',
        'features.npy',
        'labels.npy',
        'feature_names.csv',
        'X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy',
        'X_train_A.npy', 'X_test_A.npy',
        'X_train_B.npy', 'X_test_B.npy',
        'scaler.pkl',
        'best_model.pkl'
    ]

    print("Checking files...")
    print("-" * 60)
    
    file_exists = {}
    for f in files_to_check:
        info = get_file_info(f)
        print(f"{f:<20} | {info}")
        file_exists[f] = os.path.exists(f)
        
    print("-" * 60)
    print("\n")

    # Determine Statuses
    dataset_loaded = file_exists['dataset.csv']
    features_extracted = file_exists['features.npy'] and file_exists['labels.npy']
    split_done = file_exists['X_train.npy'] and file_exists['X_test.npy']
    
    smote_done = False
    # Heuristic: If y_train exists and has exactly balanced classes, or if X_train_smote exists
    if file_exists.get('y_train.npy', False):
        try:
            import numpy as np
            y = np.load('y_train.npy', allow_pickle=True)
            _, counts = np.unique(y, return_counts=True)
            if len(counts) > 1 and max(counts) == min(counts):
                smote_done = True
        except:
            pass
    if os.path.exists('X_train_smote.npy') or os.path.exists('y_train_smote.npy'):
        smote_done = True

    version_abc = file_exists['X_train_A.npy'] and file_exists['X_train_B.npy']
    model_trained = file_exists['best_model.pkl']

    statuses = [
        ("Dataset loaded", dataset_loaded, "Load the dataset and save it as dataset.csv"),
        ("Features extracted", features_extracted, "Extract features and save as features.npy & labels.npy"),
        ("Train/test split done", split_done, "Perform train/test split and save X_train, X_test, etc."),
        ("SMOTE balancing done", smote_done, "Apply SMOTE and balance classes in training set"),
        ("Version A/B/C created", version_abc, "Create version A/B data splits and save them"),
        ("Model trained", model_trained, "Train a model and save as best_model.pkl")
    ]

    # Print Status Board
    # Adjust width depending on double-width characters in terminal if needed
    # Standard string length calculation used here.
    board_width = 42
    print("╔" + "═" * board_width + "╗")
    print("║" + "PROJECT STATUS DIAGNOSTIC".center(board_width) + "║")
    print("╠" + "═" * board_width + "╣")
    
    next_step = None
    
    for label, is_done, action in statuses:
        icon = "✅" if is_done else "❌"
        inner_text = f" [{icon}] {label}"
        
        # Colab/terminals often render emojis as 2 chars wide, but python len() is 1. 
        # Using ljust usually works fine, but we pad manually to be safe.
        padding_needed = board_width - len(inner_text)
        
        # We might need an extra space adjustment depending on the console's emoji rendering width,
        # but standard python spacing is 1 emoji = 1 char.
        # To make it perfectly align in consoles where ✅ is width 2, you might subtract 1 space.
        # We will use python's string length here which is the standard programmatic way.
        # If it's off by 1 in Colab due to font-width, it's a minor visual glitch.
        # Python 3: len("✅") == 1
        
        # Add a special width check if we want to be hyper-accurate for Colab
        import unicodedata
        display_width = sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in inner_text)
        actual_padding = board_width - display_width
        
        print(f"║{inner_text}" + " " * actual_padding + "║")
        
        if not is_done and next_step is None:
            next_step = action

    print("╚" + "═" * board_width + "╝")
    print()
    if next_step:
        print(f"NEXT STEP: {next_step}")
    else:
        print("NEXT STEP: Project complete! 🎉")

if __name__ == "__main__":
    check_status()
