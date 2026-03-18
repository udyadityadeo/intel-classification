Intel Classification- Classical ML Pipeline
This project implements a classical computer vision pipeline for scene classification using the Intel Classification dataset.
The objective was to study how far classical ML pipelines can go for image classification and analyse their limitations compared to modern DL networks.

Dataset

    Dataset: Intel Image Classification Dataset, Kaggle.
    Classes: buildings, forest, glacier, mountain, sea, street

    | Split | Images  |
    | ----- | ------- |
    | Train | ~14,000 |
    | Test  | ~3,000  |

Pipeline Overview:

    Image
    ↓
    Feature Extraction
    ├── HOG (Histogram of Oriented Gradients)
    └── LBP (Local Binary Patterns)
    ↓
    Feature Fusion (concatenation)
    ↓
    StandardScaler
    ↓
    PCA (Dimensionality Reduction)
    ↓
    Classifier
    ├── Logistic Regression
    ├── Linear SVM
    └── RBF SVM

    Feature extraction converts images into high-dimensional feature vectors, after which dimensionality reduction and classification are applied.

Feature Engineering:

Histogram of Gradients (HOG):
    Captures edge structure and gradient orientation patterns in images- useful for building edges, road boundaries, skyline contours

Local binary Patterns (LBP):
    Captures local texture patterns- better for foilage in forests, snow/ice patterns, water surfaces

Feature Fusion:
    Concatenation of features extracted from HOG and LBP- complementary in nature, into a single vector

Dimensionality Reduction:
    PCA used to reduce dimensionality evaluated over 80, 100, 150 (n_components), best performance obtained with n-components = 100

Evaluation:
| Model               | Features  | Accuracy |
| ------------------- | --------- | -------- |
| Logistic Regression | HOG       | ~0.72    |
| Linear SVM          | HOG       | ~0.74    |
| RBF SVM             | HOG       | ~0.76    |
| RBF SVM             | HOG + LBP | ~0.76    |

Best Model:
    Pipeline- StandardScaler(), PCA(n_components = 100), SVM(kernel = 'rbf')
    Best Hyperparameters- C = 10, gamma = 'scale'

Performance- 
    Cross-validation accuracy : ~0.754
    Test accuracy             : ~0.757

Key findings:
    1. Classes like sea, glaciers and mountain show confusion due to similar 
        visual patterns.
    2. Combining HOG and LBP did not have any significant gains- PCA retains 
        100 components, information added by LBP maybe beyond these components.
    3. Plateauing around 75-80% accuracy highlights the limits of handcrafted 
        feature extraction.
