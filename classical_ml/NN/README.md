Absolutely. Here is a **clean, repository-ready `README.md`** that documents the project without overclaiming and keeps the experimental and deployment parts clearly separated.

````markdown
# Intel Image Scene Classification

End-to-end image scene classification project using classical computer vision, classical machine learning, and a PyTorch CNN, followed by REST API serving with FastAPI and containerized deployment using Docker.

---

## Project Overview

The project investigates image classification across six scene categories from the Intel Image Classification dataset:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

The project is developed in two stages:

1. **Classical ML pipeline**
   - HOG
   - LBP
   - PCA
   - Logistic Regression
   - SVM
   - Hyperparameter search

2. **Deep learning pipeline**
   - PyTorch CNN
   - Controlled augmentation experiment
   - Grad-CAM visualization
   - FastAPI inference API
   - Docker and Docker Compose deployment

The overall workflow is:

```text
                    Intel Scene Images
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
      Classical ML                    CNN
             │                           │
      HOG / LBP Features          Learned Features
             │                           │
            PCA                    Classification
             │                           │
       SVM / Logistic             Evaluation
             │                           │
             └─────────────┬─────────────┘
                           ▼
                    Model Evaluation
                           │
                           ▼
                    FastAPI Inference
                           │
                           ▼
                     Docker Image
                           │
                           ▼
                    Docker Compose
````

---

# Dataset

The project uses the Intel Image Classification dataset.

The six target classes are:

```text
buildings
forest
glacier
mountain
sea
street
```

The expected dataset structure is:

```text
intel-classification/
├── data/
│   ├── train/
│   │   ├── buildings/
│   │   ├── forest/
│   │   ├── glacier/
│   │   ├── mountain/
│   │   ├── sea/
│   │   └── street/
│   │
│   └── test/
│       ├── buildings/
│       ├── forest/
│       ├── glacier/
│       ├── mountain/
│       ├── sea/
│       └── street/
│
└── ...
```

Additional project data is stored under:

```text
data/
├── features/
├── pred/
├── raw/
├── test/
└── train/
```

---

# Classical Machine Learning

The classical pipeline investigates whether manually engineered image representations can provide useful classification performance before moving to learned CNN representations.

## Feature Extraction

### HOG

Histogram of Oriented Gradients (HOG) captures local edge and gradient information and provides a representation of image structure.

### LBP

Local Binary Patterns (LBP) capture local texture information by comparing neighboring pixel intensities.

Both feature representations are evaluated independently and can also be combined.

```text
Image
 │
 ├── HOG ──┐
 │         │
 └── LBP ──┤
           ▼
      Feature Vector
           │
           ▼
          PCA
           │
           ▼
       Classifier
```

## Models

The classical experiments include:

* Logistic Regression
* Support Vector Machine (SVM)
* PCA
* GridSearchCV for hyperparameter selection

Model performance is evaluated using classification metrics and confusion matrices.

---

# CNN Experiment

A convolutional neural network was implemented using PyTorch.

The CNN learns image representations directly from the input pixels rather than relying on manually engineered HOG or LBP features.

## Controlled Experiment

Two CNN configurations are compared:

1. **Baseline CNN**
2. **ColorJitter CNN**

The experiment keeps the training configuration fixed and changes the augmentation strategy.

Both models use the same stratified 85/15 training-validation split.

The split indices are stored in:

```text
checkpoints/split_indices.npz
```

This ensures that both models are evaluated on the same validation samples.

---

## Baseline Augmentation

```text
Resize
   ↓
RandomHorizontalFlip
   ↓
ToTensor
```

Train using:

```bash
python train.py --augmentation baseline
```

Checkpoint:

```text
checkpoints/baseline_cnn.pth
```

---

## ColorJitter Augmentation

```text
Resize
   ↓
RandomHorizontalFlip
   ↓
ColorJitter
   ↓
ToTensor
```

Train using:

```bash
python train.py --augmentation colorjitter
```

Checkpoint:

```text
checkpoints/colorjitter_cnn.pth
```

ColorJitter parameters:

```text
brightness = 0.15
contrast   = 0.15
saturation = 0.15
hue        = 0.05
```

---

## Training Configuration

Both CNN experiments use the same settings:

| Parameter               |      Value |
| ----------------------- | ---------: |
| Input size              |  128 × 128 |
| Batch size              |        128 |
| Optimizer               |       Adam |
| Learning rate           |       1e-3 |
| Maximum epochs          |         20 |
| Early stopping patience |          5 |
| Random seed             |         42 |
| Validation split        |        15% |
| Split type              | Stratified |

The augmentation strategy is therefore the primary experimental difference between the two CNN configurations.

---

# CNN Reproduction

## Installation

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Train Baseline

```bash
python train.py --augmentation baseline
```

## Train ColorJitter Model

```bash
python train.py --augmentation colorjitter
```

## Evaluate Baseline

```bash
python evaluate.py \
    --checkpoint checkpoints/baseline_cnn.pth
```

## Evaluate ColorJitter

```bash
python evaluate.py \
    --checkpoint checkpoints/colorjitter_cnn.pth
```

---

# Grad-CAM

Grad-CAM is used to visualize the spatial regions contributing to CNN predictions.

The final target layer used for Grad-CAM is:

```python
model.features[17]
```

This provides a qualitative interpretation of what regions of the input image are influencing the model's prediction.

---

# Controlled Checkpoints

The final controlled CNN experiment produces:

```text
checkpoints/
├── split_indices.npz
├── baseline_cnn.pth
└── colorjitter_cnn.pth
```

### Important

The older Drive checkpoint:

```text
best_cnn.pth
```

should **not** be used as the controlled baseline.

The controlled comparison uses:

```text
baseline_cnn.pth
colorjitter_cnn.pth
```

because these checkpoints correspond to the reproducible experimental setup described above.

---

# Model Serving

The trained CNN is exposed through a REST API using FastAPI.

The deployment application is located under:

```text
NN/deployment/
├── app/
│   ├── __init__.py
│   ├── inference.py
│   ├── main.py
│   └── model.py
│
├── models/
├── docker-compose.yml
├── dockerfile
└── requirements.txt
```

The API provides two primary endpoints.

---

## Health Check

```http
GET /health
```

Example:

```bash
curl http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "healthy"
}
```

---

## Prediction

```http
POST /predict
```

The endpoint accepts an image using multipart form data.

Example:

```bash
curl -X POST \
  "http://127.0.0.1:8000/predict" \
  -F "file=@../../../data/test/buildings/20057.jpg"
```

Example response:

```json
{
  "class": "buildings",
  "confidence": 0.981627881526947
}
```

---

# Interactive API Documentation

FastAPI automatically generates interactive API documentation.

Once the service is running, open:

```text
http://127.0.0.1:8000/docs
```

The OpenAPI specification is available at:

```text
http://127.0.0.1:8000/openapi.json
```

The Swagger UI can be used to upload an image and test the `/predict` endpoint directly.

---

# Docker Deployment

The inference service is packaged as a Docker image.

## Build the Image

From the deployment directory:

```bash
cd NN/deployment

docker build -t intel-scene-api .
```

## Run the Container

```bash
docker run --rm -p 8000:8000 intel-scene-api
```

The API will be available at:

```text
http://127.0.0.1:8000
```

---

# Docker Compose

Docker Compose is provided to simplify container lifecycle management.

## Start the Service

```bash
docker compose up -d --build
```

## Check Running Containers

```bash
docker compose ps
```

## Test the API

```bash
curl http://127.0.0.1:8000/health
```

## Stop the Service

```bash
docker compose down
```

The Compose configuration maps:

```text
Host port:       8000
Container port:  8000
```

---

# Deployment Architecture

```text
                Client
                  │
                  │ HTTP
                  ▼
        ┌─────────────────────┐
        │     FastAPI API     │
        │                     │
        │ GET  /health        │
        │ POST /predict       │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Image Preprocessing │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │    PyTorch CNN      │
        │                     │
        │ ColorJitter CNN     │
        └──────────┬──────────┘
                   │
                   ▼
          Class + Confidence
```

The application and model are packaged together inside the Docker image, allowing the inference environment to be reproduced independently of the local Python environment.

---

# Project Structure

```text
intel-classification/
│
├── data/
│   ├── features/
│   ├── pred/
│   ├── raw/
│   ├── test/
│   └── train/
│
├── classical_ml/
│   ├── analysis/
│   ├── exploration/
│   ├── features/
│   └── models/
│
├── NN/
│   ├── checkpoints/
│   ├── deployment/
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── inference.py
│   │   │   ├── main.py
│   │   │   └── model.py
│   │   │
│   │   ├── models/
│   │   ├── docker-compose.yml
│   │   ├── dockerfile
│   │   └── requirements.txt
│   │
│   ├── evaluate.py
│   ├── model.py
│   └── train.py
│
├── configs/
├── exploration/
├── tests/
├── requirements.txt
└── README.md
```

---

# Technologies

| Area             | Technologies         |
| ---------------- | -------------------- |
| Language         | Python               |
| Computer Vision  | HOG, LBP             |
| Classical ML     | Scikit-learn         |
| Deep Learning    | PyTorch, Torchvision |
| Model Serving    | FastAPI, Uvicorn     |
| Containerization | Docker               |
| Orchestration    | Docker Compose       |
| Development      | VS Code, Jupyter     |

---

# Key Outcomes

The project demonstrates the progression from traditional feature engineering to learned image representations and finally to model deployment:

```text
Handcrafted Features
       │
       ▼
 HOG / LBP + PCA
       │
       ▼
 Classical ML
       │
       ▼
     CNN
       │
       ▼
   Grad-CAM
       │
       ▼
   FastAPI
       │
       ▼
    Docker
       │
       ▼
 Docker Compose
```

This provides both an experimental comparison between classical and deep-learning approaches and a reproducible inference pipeline for serving the trained CNN.

```
