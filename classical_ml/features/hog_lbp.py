import os
import numpy as np
from hog import extract_hog
from lbp import extract_lbp

def build_features(base_dir):

    class_names = sorted([c for c in os.listdir(base_dir) if not c.startswith(".")])

    X, y = [], []

    for label, class_name in enumerate(class_names):

        folder = os.path.join(base_dir, class_name)

        for fname in os.listdir(folder):

            if fname.startswith("."):
                continue

            path = os.path.join(folder, fname)

            try:
                hog_feat = extract_hog(path)
                lbp_feat = extract_lbp(path)

                combined_feat = np.concatenate([hog_feat, lbp_feat])

                X.append(combined_feat)
                y.append(label)

            except Exception as e:
                print(f"Error processing {path}: {e}")

    return np.vstack(X), np.array(y)