import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

base_dir = Path(__file__).resolve().parents[2]
feature_dir = base_dir / "data" / "features"

X_train = np.load(os.path.join(feature_dir, "train_X.npy"))
y_train = np.load(os.path.join(feature_dir, "train_y.npy"))
X_test  = np.load(os.path.join(feature_dir, "test_X.npy"))
y_test  = np.load(os.path.join(feature_dir, "test_y.npy"))

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# PCA Variance Curve (visualization only)
X_all = np.vstack([X_train, X_test])
scaler_vis = StandardScaler()
pca_vis = PCA(random_state=42)
pca_vis.fit(scaler_vis.fit_transform(X_all))
explained = np.cumsum(pca_vis.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(explained)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_explained_variance.png", dpi=150)
plt.show()

# Accuracy vs PCA Components
max_dim = X_train.shape[1]
pca_dims = [d for d in [50, 100, 300, 500, 1000] if d < max_dim]
accuracies = []

for dim in pca_dims:
    print(f"\nTraining with PCA components = {dim}")
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=dim, random_state=42)),
        ('svm', SVC(kernel='rbf', C=1, gamma='scale', random_state=42))
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    accuracies.append(acc)
    print(f"Accuracy: {acc:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(pca_dims, accuracies, marker='o')
plt.xlabel("Number of PCA Components")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs PCA Components")
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_accuracy_vs_components.png", dpi=150)
plt.show()
