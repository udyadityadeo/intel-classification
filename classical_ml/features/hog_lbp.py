import os
import numpy as np
import argparse
from hog import extract_hog
from lbp import extract_lbp

valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def build_features(base_dir, expected_classes=None):

    class_names = sorted([
            c for c in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, c)) and not c.startswith(".")
        ])
    
    if not class_names:
        raise ValueError(f"No class folders found in {base_dir}")
    
    if expected_classes is not None:
        if class_names != expected_classes:
            raise ValueError(
                f"Class mismatch!\n"
                f"  Expected: {expected_classes}\n"
                f"  Found:    {class_names}"
            )
        
        
    X, y = [], []

    for label, class_name in enumerate(class_names):
        folder = os.path.join(base_dir, class_name)

        files = sorted([
            f for f in os.listdir(folder)
            if not f.startswith(".") and os.path.splitext(f)[1].lower() in valid_exts
        ])
        count = 0

        for fname in files:
            path = os.path.join(folder, fname)
            
            try:
                hog_feat = extract_hog(path)
                lbp_feat = extract_lbp(path)
                combined_feat = np.concatenate([hog_feat, lbp_feat])
                X.append(combined_feat)
                y.append(label)
                count += 1

            except Exception as e:
                print(f"Error processing {path}: {e}")

            print(f"  Class '{class_name}': {count}/{len(files)} images processed")

            if count == 0:
                print(f"No features extracted for class: {class_name}. Skipping this class.")

        if len(X) == 0:
            raise ValueError(f"No features could be extracted from {base_dir}. Check image files.")

    return np.vstack(X), np.array(y), class_names

def save_features(X,y, class_names, save_dir, prefix, overwrite=False):
    os.makedirs(save_dir, exist_ok=True)

    X_path = os.path.join(save_dir, f"{prefix}_X.npy")
    y_path = os.path.join(save_dir, f"{prefix}_y.npy")
    class_path = os.path.join(save_dir, "class_names.txt")
 
    np.save(X_path, X)
    np.save(y_path, y)

    if not os.path.exists(class_path) or overwrite:
        with open(class_path, "w") as f:
            for c in class_names:
                f.write(f"{c}\n")
            
        print(f"  Saved class names: {class_path}")
    else:
        print(f"  [INFO] class_names.txt already exists, skipping overwrite.")

    
    print(f"  Saved features : {X_path}  shape={X.shape}")
    print(f"  Saved labels   : {y_path}  shape={y.shape}")

def parse_args():
    parser = argparse.ArgumentParser(description="Extract HOG+LBP features from image dataset.")
    parser.add_argument("--train_dir", type=str, default=None, help="Path to train directory")
    parser.add_argument("--test_dir",  type=str, default=None, help="Path to test directory")
    parser.add_argument("--feature_dir", type=str, default=None, help="Path to save features")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
 
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir    = os.path.join(base_dir, "data")
    train_dir   = args.train_dir   or os.path.join(data_dir, "train")
    test_dir    = args.test_dir    or os.path.join(data_dir, "test")
    feature_dir = args.feature_dir or os.path.join(data_dir, "features")

    print("\nExtracting TRAIN features...")
    X_train, y_train, class_names = build_features(train_dir)
    save_features(X_train, y_train, class_names, feature_dir, "hog_lbp_train", overwrite=True)
 
    print("\nExtracting TEST features...")
    X_test, y_test, _ = build_features(test_dir, expected_classes=class_names)
    save_features(X_test, y_test, class_names, feature_dir, "hog_lbp_test", overwrite=True)

    print("\nDone.")