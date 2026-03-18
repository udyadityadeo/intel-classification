''' Extracting Histogram of Oriented Gradients (HOG) features from the intel
dataset, saving them on disk for reuse '''

import os
from pyexpat import features
import numpy as np
from skimage.feature import hog
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from tqdm import tqdm

base_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, 'data')
feature_dir = os.path.join(data_dir, 'features')

train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')


def extract_hog(path):
    img = imread(path)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img = img.astype("float32") / 255.0

    img = resize(img, (150, 150), anti_aliasing=True)
    img_gray = rgb2gray(img)

    features = hog(
        img_gray,
        orientations = 9,
        pixels_per_cell = (8, 8),
        cells_per_block = (2, 2),
        block_norm = 'L2-Hys',
    )

    return features

def extract_hog_features(folder_path, class_names):
    X = []
    y = []

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(folder_path, class_name)

        print(f"Processing Class: {class_name}")

        for file in tqdm(os.listdir(class_dir)):
            img_path = os.path.join(class_dir, file)

            try:
                features = extract_hog(img_path)
                X.append(features)
                y.append(label)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    return np.array(X), np.array(y)

def save_features(X, y, filename):
    os.makedirs(feature_dir, exist_ok=True)

    X_path = os.path.join(feature_dir, f"{filename}_X.npy")
    y_path = os.path.join(feature_dir, f"{filename}_y.npy")

    np.save(X_path, X)
    print(f"Saved features to {X_path}")
    np.save(y_path, y)
    print(f"Saved labels to {y_path}")  

def main():
    os.makedirs(feature_dir, exist_ok=True)
    class_names = sorted([
    d for d in os.listdir(train_dir)
    if os.path.isdir(os.path.join(train_dir, d))
])
    print(f"Class Names: {class_names}")

    with open(os.path.join(feature_dir, "class_names.txt"), "w") as f:
        for c in class_names:
            f.write(f"{c}\n")


    print(f"Saved class names to {os.path.join(feature_dir, 'class_names.txt')}")
    print(f"Classes: {class_names}")

    print(f"extracting features for training set")
    X_train, y_train = extract_hog_features(train_dir, class_names)
    save_features(X_train, y_train, 'train')

    print(f"extracting features for test set")
    X_test, y_test = extract_hog_features(test_dir, class_names)
    save_features(X_test, y_test, 'test')

    print(f"Feature extraction and saving completed.")
    print(f"Features saved in {feature_dir}")

if __name__ == "__main__":
    main()

