import os
from matplotlib.pyplot import grid
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

base_dir = os.path.dirname(os.path.dirname(__file__))
feature_dir = os.path.join(base_dir, 'data', 'features')    


def load_features():
    X_train = np.load(os.path.join(feature_dir, "train_X.npy"))
    y_train = np.load(os.path.join(feature_dir, "train_y.npy"))

    X_test = np.load(os.path.join(feature_dir, "test_X.npy"))
    y_test = np.load(os.path.join(feature_dir, "test_y.npy"))

    return X_train, y_train, X_test, y_test

def main():
    print(f"Loading features")
    X_train, y_train, X_test, y_test = load_features()

    print("train shape: ", X_train.shape)
    print("test shape: ", X_test.shape)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ("pca", PCA(n_components=300, svd_solver="randomized")),
        ('classifier', LogisticRegression(
            solver = "lbfgs",
            max_iter=1000))
        ])

    param_grid = {
        "classifier__C": [0.1, 1, 10],
        }

    print("Running Grid Search: ")

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv = 5,
        scoring = 'accuracy',
        n_jobs = -1
        )
    
    grid.fit(X_train, y_train)


    print("\n=== Cross-Validation Results ===")
    print("Best C:", grid.best_params_["classifier__C"])
    print("Best CV Accuracy:", grid.best_score_)

    best_model = grid.best_estimator_

    print("\nEvaluating on test set...")
    y_pred = best_model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\n=== Test Results ===")
    print("Test Accuracy:", test_accuracy)
    print("Macro F1:", macro_f1)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()