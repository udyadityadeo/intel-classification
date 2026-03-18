import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

base_dir = os.path.dirname(os.path.dirname(__file__))
feature_dir = os.path.join(base_dir, "data", "features")

X_train = np.load(os.path.join(feature_dir, "train_X.npy"))
y_train = np.load(os.path.join(feature_dir, "train_y.npy"))
X_test = np.load(os.path.join(feature_dir, "test_X.npy"))
y_test = np.load(os.path.join(feature_dir, "test_y.npy"))

for C in [0.01, 0.1, 1, 10]:
    for gamma in ['scale', 0.01, 0.001]:
        print(f"\nTraining Radial SVC with C={C} and gamma={gamma}")
        pipeline = Pipeline(
            (('scalar', StandardScaler()),
             ('pca', PCA(n_components=100)),
             ('clf', SVC(C=C, kernel='rbf', gamma=gamma))
            )
        )

        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)

        print(f"Radial SVC (PCA 100) Accuracy : {accuracy_score(y_test, pred):.4f}")

