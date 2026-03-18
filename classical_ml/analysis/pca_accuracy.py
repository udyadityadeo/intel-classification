import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
feature_dir = os.path.join(base_dir, "data", "features")

X_train = np.load(os.path.join(feature_dir, "train_X.npy"))
y_train = np.load(os.path.join(feature_dir, "train_y.npy"))
X_test = np.load(os.path.join(feature_dir, "test_X.npy"))
y_test = np.load(os.path.join(feature_dir, "test_y.npy"))

pca_dims = [50, 100, 300, 500, 1000, 2000]
accuracies = []

for dim in pca_dims:
    print(f"\nTraining with {dim} PCA components")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=dim, svd_solver="randomized")),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print(f"Accuracy: {acc:.4f}")

plt.figure(figsize=(8,5))
plt.plot(pca_dims, accuracies, marker='o')
plt.xlabel("Number of PCA Components")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs PCA Dimension")
plt.grid(True)
plt.show()