import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

base_dir = os.path.dirname(os.path.dirname(__file__))
feature_dir = os.path.join(base_dir, "data", "features")

X_train = np.load(os.path.join(feature_dir, "train_X.npy"))
y_train = np.load(os.path.join(feature_dir, "train_y.npy"))
X_test = np.load(os.path.join(feature_dir, "test_X.npy"))
y_test = np.load(os.path.join(feature_dir, "test_y.npy"))

for C in [0.01, 0.1, 1, 10]:
    print(f"\nTraining Linear SVC with C={C}")
    pipeline = Pipeline(
    (('scalar', StandardScaler()),
     ('pca', PCA(n_components=100)),
     ('clf', LinearSVC(C=C, max_iter=5000))
    )
    )

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    print(f"Linear SVC (PCA 100) Accuracy : {accuracy_score(y_test, pred):.4f}")

