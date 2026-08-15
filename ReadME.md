Yes. Given the restructuring we just decided on, I would **clean this README before treating it as final**. The main issue is that it still refers to `NN/` as a subdirectory of `classical_ml/`, and some sections duplicate information that belongs in the CNN-specific README.

I would make the **root README** a project-level overview, with `classical_ml/README.md` and `neural_network/README.md` handling implementation details.

### Recommended root `README.md`

````markdown
# Intel Image Scene Classification

End-to-end image scene classification project comparing classical computer-vision pipelines with a PyTorch convolutional neural network, followed by model interpretability and containerized inference deployment.

---

## Project Overview

The project investigates classification across six scene categories from the Intel Image Classification dataset:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

The project follows a progression from handcrafted visual representations to learned image representations and finally to deployable inference.

```text
                         Intel Scene Images
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
          Classical ML                  Neural Network
                 │                             │
          HOG / LBP Features                 CNN
                 │                             │
                PCA                    Controlled Augmentation
                 │                             │
         SVM / Logistic                 Baseline vs ColorJitter
                 │                             │
                 └──────────────┬──────────────┘
                                │
                                ▼
                         Model Evaluation
                                │
                                ▼
                             Grad-CAM
                                │
                                ▼
                         FastAPI Inference
                                │
                                ▼
                             Docker
                                │
                                ▼
                        Docker Compose
````

---

# Dataset

The project uses the Intel Image Classification dataset containing six scene classes:

```text
buildings
forest
glacier
mountain
sea
street
```

Expected dataset structure:

```text
data/
├── train/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
│
└── test/
    ├── buildings/
    ├── forest/
    ├── glacier/
    ├── mountain/
    ├── sea/
    └── street/
```

---

# Classical Machine Learning

The classical pipeline evaluates whether handcrafted image representations can provide effective scene classification before moving to learned CNN representations.

## Feature Extraction

### HOG

Histogram of Oriented Gradients (HOG) captures local edge and gradient structure within an image.

### LBP

Local Binary Patterns (LBP) capture local texture information based on neighboring pixel intensities.

The resulting feature representations are evaluated individually and in combination.

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
* Principal Component Analysis (PCA)
* GridSearchCV for hyperparameter selection

The best classical configuration achieved approximately **76% test accuracy**, with confusion concentrated among visually similar scene classes such as glacier, mountain, and sea.

Further details are available in:

```text
classical_ml/README.md
```

---

# Neural Network

A convolutional neural network was implemented using PyTorch to learn image representations directly from raw pixels.

The CNN experiments investigate the effect of controlled image augmentation.

Two configurations were compared:

1. **Baseline CNN**
2. **ColorJitter CNN**

Both models use the same:

* Stratified 85/15 validation split
* Random seed
* Optimizer
* Learning rate
* Batch size
* Training schedule
* Early stopping configuration

The primary experimental difference is the augmentation strategy.

---

## CNN Results

| Model           | Test Accuracy |
| --------------- | ------------: |
| Baseline CNN    |    **84.17%** |
| ColorJitter CNN |    **86.47%** |

ColorJitter therefore produced a **2.30 percentage-point improvement** in test accuracy over the controlled baseline.

The improvement was particularly reflected in classes where appearance and illumination can vary substantially.

---

## Baseline Augmentation

```text
Resize
   ↓
RandomHorizontalFlip
   ↓
ToTensor
```

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

ColorJitter parameters:

```text
brightness = 0.15
contrast   = 0.15
saturation = 0.15
hue        = 0.05
```

Training configuration:

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

---

# Grad-CAM

Grad-CAM was used to provide qualitative interpretability for CNN predictions.

The final convolutional feature layer used as the Grad-CAM target was:

```python
model.features[17]
```

The visualizations allow inspection of the spatial regions contributing most strongly to the model's predicted class.

This provides an additional evaluation layer beyond aggregate classification metrics by examining whether the CNN is attending to semantically meaningful scene regions.

---

# Model Serving

The trained CNN is exposed through a REST API using FastAPI.

The deployment implementation is contained within:

```text
neural_network/
└── deployment/
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

The API provides:

### Health Check

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

### Prediction

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

FastAPI provides automatically generated API documentation.

Once the service is running:

```text
http://127.0.0.1:8000/docs
```

The OpenAPI specification is available at:

```text
http://127.0.0.1:8000/openapi.json
```

---

# Docker Deployment

The inference service is packaged into a Docker image containing the application, dependencies, and trained model.

From the deployment directory:

```bash
cd neural_network/deployment
```

Build the image:

```bash
docker build -t intel-scene-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 intel-scene-api
```

The API is then available at:

```text
http://127.0.0.1:8000
```

---

# Docker Compose

Docker Compose is provided for simplified service management.

Start the service:

```bash
docker compose up -d --build
```

Check the service:

```bash
docker compose ps
```

Test the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Stop the service:

```bash
docker compose down
```

The Compose configuration maps:

```text
Host:      8000
Container: 8000
```

---

# Deployment Architecture

```text
                    Client
                      │
                      │ HTTP
                      ▼
              ┌───────────────┐
              │    FastAPI    │
              │               │
              │ GET /health   │
              │ POST /predict │
              └───────┬───────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Image Processing│
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │    PyTorch CNN  │
             └────────┬────────┘
                      │
                      ▼
                Class + Score
```

The model and inference application are packaged together inside the Docker image, providing a reproducible inference environment.

---

# Project Structure

```text
intel-classification/
│
├── data/
│   ├── train/
│   └── test/
│
├── classical_ml/
│   ├── analysis/
│   ├── exploration/
│   ├── features/
│   ├── models/
│   └── README.md
│
├── neural_network/
│   ├── checkpoints/
│   ├── deployment/
│   │   ├── app/
│   │   ├── models/
│   │   ├── docker-compose.yml
│   │   ├── dockerfile
│   │   └── requirements.txt
│   │
│   ├── evaluate.py
│   ├── model.py
│   ├── train.py
│   └── README.md
│
├── configs/
├── logs/
├── notebooks/
├── tests/
└── README.md
```

---

# Technologies

| Area                   | Technologies         |
| ---------------------- | -------------------- |
| Language               | Python               |
| Computer Vision        | HOG, LBP             |
| Classical ML           | Scikit-learn         |
| Deep Learning          | PyTorch, Torchvision |
| Model Interpretability | Grad-CAM             |
| Model Serving          | FastAPI, Uvicorn     |
| Containerization       | Docker               |
| Container Management   | Docker Compose       |
| Development            | VS Code, Jupyter     |

---

# Key Outcomes

The project demonstrates a progression from handcrafted image features to learned representations and production-oriented inference:

```text
HOG / LBP
    │
    ▼
PCA + Classical ML
    │
    │ ~76% accuracy
    ▼
CNN
    │
    │ 84.17%
    ▼
Controlled ColorJitter Experiment
    │
    │ 86.47%
    ▼
Grad-CAM Interpretability
    │
    ▼
FastAPI Inference API
    │
    ▼
Docker / Docker Compose
```

The project therefore combines:

* **Classical computer vision**
* **Machine learning**
* **Deep learning**
* **Controlled experimentation**
* **Model interpretability**
* **REST API development**
* **Containerized deployment**

````

### One thing I would **not** put in the root README

Don't put the entire CNN reproduction procedure, checkpoint caveats, or every training detail here. Put those in:

```text
neural_network/README.md
````

Likewise, your HOG/LBP/PCA details belong in:

```text
classical_ml/README.md
```

That gives you a much more professional hierarchy:

```text
README.md
│
├── What is this project?
├── What was investigated?
├── Results
├── Architecture
└── Where to find each component
       │
       ├── classical_ml/README.md
       └── neural_network/README.md
```

And importantly, **the 84.17% → 86.47% result and Grad-CAM are now visible immediately**, rather than buried deep inside the implementation documentation.
