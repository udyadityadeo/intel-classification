import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from lbp import extract_lbp

project_root = os.path.dirname(os.path.dirname(__file__))
base_dir = os.path.join(project_root, "data", "train")
classes = [c for c in os.listdir(base_dir) if not c.startswith(".")]

X, y = [], []

for label, cls in enumerate(classes):
    folder = os.path.join(base_dir, cls)

    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)

        try:
            feat = extract_lbp(path)
            X.append(feat)
            y.append(label)
        except:
            continue

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

svm = SVC(kernel="rbf")
svm.fit(X_train, y_train)

pred = svm.predict(X_test)

print("LBP + SVM Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
