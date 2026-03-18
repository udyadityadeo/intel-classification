import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from features.hog_lbp import build_features

base_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, "data", "train")

X,y = build_features(data_dir)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

pipeline = Pipeline(
    [('scalar', StandardScaler()),
     ('pca', PCA()),
     ('svm', SVC())]
    )

param_grid = {
    'pca__n_components': [80, 100, 150],
    'svm__C': [0.1, 1, 10],
    'svm__gamma': ['scale', 0.01, 0.001],
    'svm__kernel': ['rbf']
}

grid = GridSearchCV(
    pipeline, 
    param_grid, 
    n_jobs= 3, 
    scoring='accuracy', 
    cv=5, 
    verbose=2
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print(f"Best CV Accuracy: {grid.best_score_:.4f}")

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))