import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from features.hog_lbp import build_features

base_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, "data", "train")

X, y = build_features(data_dir)
print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=100, svd_solver='randomized')
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

svm = SVC(kernel='rbf', C=1, gamma='scale')
svm.fit(X_train, y_train)

pred = svm.predict(X_test)
print(f"HOG + LBP SVM Accuracy: {accuracy_score(y_test, pred):.4f}")