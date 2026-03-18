import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

base_dir = os.path.dirname(os.path.dirname(__file__))
feature_dir = os.path.join(base_dir,"data", "features")

X_train = np.load(os.path.join(feature_dir, "train_X.npy"))
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X_train)

pca = PCA(n_components=1500, svd_solver = "randomized")
pca.fit(X_scaled)
print("Number of PCA components computed:", len(pca.explained_variance_ratio_))
print("First 5 variance ratios:", pca.explained_variance_ratio_[:5])

explained = np.cumsum(pca.explained_variance_ratio_)
print("First 5 cumulative:", explained[:5])

for threshold in [0.90, 0.95, 0.99]:
    if explained[-1] < threshold:
        print(f"{int(threshold*100)}% variance NOT reached with 1500 components")
    else:
        components = np.argmax(explained >= threshold) + 1
        print(f"{int(threshold*100)}% variance explained with {components} components")

print("Variance explained by first 1500 components:", explained[1499])

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(explained)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance Curve")
plt.grid(True)
plt.show()