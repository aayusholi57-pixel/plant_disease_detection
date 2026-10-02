# 🌿 Plant Disease Detection API

A production-oriented **plant leaf disease classification system** built with **PyTorch, ResNet18, FastAPI, Docker, GitHub Actions, and AWS deployment tooling**.

The project takes a plant-leaf image, runs it through a trained ResNet18 classifier, and returns the predicted disease/health class together with a confidence score through a REST API.

[![CI](https://github.com/aayusholi57-pixel/plant_disease_detection/actions/workflows/ci.yml/badge.svg)](https://github.com/aayusholi57-pixel/plant_disease_detection/actions/workflows/ci.yml)

---

## 📌 Project Overview

### What this project does

1. Downloads a public plant-disease image dataset.
2. Prepares the images for ResNet18.
3. Uses **transfer learning** from pretrained ResNet18 weights.
4. Replaces the original ImageNet classifier with a plant-disease classifier.
5. Trains the classification head on plant-leaf classes.
6. Stores the trained model as a PyTorch checkpoint.
7. Converts legacy state-dict checkpoints into a self-contained model artifact when required.
8. Evaluates the model using classification metrics.
9. Loads the trained artifact into a FastAPI application.
10. Accepts leaf images through `POST /predict`.
11. Returns the predicted class, human-readable label, and confidence.
12. Packages the API with Docker.
13. Provides GitHub Actions for CI, model training, and AWS EC2 deployment.

> **Important:** S3 is used by the repository's AWS utility scripts for storing the dataset/model. Uploading a model to S3 is **storage**, not fine-tuning. Fine-tuning happens during the PyTorch training process.

---

## ✨ Key Features

- 🧠 ResNet18-based image classification
- 🔄 Transfer learning from pretrained ResNet18 weights
- 🖼️ Image resizing and ImageNet normalization
- 🔀 Training-time image augmentation
- 📦 Self-contained inference checkpoint with class metadata
- 📊 Accuracy, macro-F1, weighted-F1 and confusion-matrix evaluation
- ⚡ FastAPI REST API
- 📚 Automatic Swagger/OpenAPI documentation
- 🛡️ Image type and upload-size validation
- 🐳 Docker and Docker Compose support
- 🧪 Automated API and model tests
- 🔁 GitHub Actions CI
- ☁️ AWS S3 upload utilities for dataset/model storage
- 📦 AWS ECR + EC2 deployment workflow
- 💾 Git LFS support for the model artifact
- 📓 Original training notebook preserved for learning/reference

---

# 🏗️ System Architecture

## End-to-End Flow

```text
                    ┌─────────────────────────┐
                    │ Plant Disease Dataset   │
                    │ Public Kaggle Dataset   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Dataset Preparation     │
                    │ scripts/dataset.py      │
                    │                         │
                    │ • Resize 224 × 224      │
                    │ • Augmentation           │
                    │ • ImageNet normalization │
                    │ • 80/20 validation split │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Pretrained ResNet18     │
                    │                         │
                    │ Backbone frozen         │
                    │ Classification head     │
                    │ trained for plant data  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Trained Model           │
                    │ plant_disease_resnet18  │
                    │ .pth                    │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
       ┌────────────┐    ┌──────────────┐   ┌──────────────┐
       │ Evaluation │    │ AWS S3       │   │ Git LFS      │
       │ Metrics    │    │ Model/Data   │   │ Model file   │
       └─────┬──────┘    └──────────────┘   └──────────────┘
             │
             ▼
       ┌─────────────────────────┐
       │ FastAPI Inference API    │
       │                          │
       │ GET  /health             │
       │ POST /predict            │
       └────────────┬────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │ Docker Container         │
       │                          │
       │ Uvicorn + FastAPI        │
       │ ResNet18 + PyTorch       │
       └────────────┬────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │ AWS ECR → AWS EC2       │
       │ CI/CD deployment        │
       └─────────────────────────┘
```

---

# 📂 Repository Structure

```text
plant_disease_detection/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   └── schemas.py
│
├── data/
│   └── raw/
│       └── .gitkeep
│
├── models/
│   ├── .gitkeep
│   ├── README.md
│   ├── plant_disease_resnet18.pth
│   └── training_history.png
│
├── notebooks/
│   └── plant_disease_training.ipynb
│
├── scripts/
│   ├── __init__.py
│   ├── convert_checkpoint.py
│   ├── dataset.py
│   ├── download_dataset.py
│   ├── evaluate.py
│   └── train.py
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_model.py
│
├── docs/
│   └── PROJECT_DOCUMENTATION.md
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── deploy.yml
│       └── train-model.yml
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── test_s3.py
├── upload_dataset.py
├── upload_model.py
└── README.md
```

---

# 🧠 Machine Learning Pipeline

## 1. Dataset

The project uses the public **PlantDisease / PlantVillage-style dataset** downloaded through the repository's dataset script.

Download it with:

```bash
python scripts/download_dataset.py
```

The dataset is placed under:

```text
data/raw/
```

The dataset itself is not committed to the repository.

---

## 2. Image Preprocessing

Training images are processed using:

- Resize to **224 × 224**
- Random horizontal flip
- Random rotation up to 15°
- Conversion to tensors
- ImageNet normalization

Validation/inference uses resizing and normalization without random augmentation.

---

## 3. ResNet18 Transfer Learning

The project starts with pretrained **ResNet18** weights:

```python
model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)
```

The original ImageNet classifier is replaced with a new classifier:

```python
model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)
```

The current training implementation freezes the ResNet18 feature-extraction layers and trains the new classification head:

```python
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)
```

So this project uses **transfer learning with a trainable classification head**. It does **not** currently update every ResNet18 backbone layer.

---

# 🎯 What "Fine-Tuning" Means Here

In this project, the pretrained ResNet18 model is adapted to plant-disease classes.

The basic idea is:

```text
Pretrained ResNet18
        │
        ├── Learned visual features
        │
        ▼
Replace ImageNet classifier
        │
        ▼
Plant-disease classifier
        │
        ▼
Train on plant-leaf images
```

The current training code freezes the backbone and trains the new final classification layer.

That is commonly described as **transfer learning / head fine-tuning**.

Full-network fine-tuning would instead allow some or all ResNet18 backbone layers to update during training.

---

# 📦 Model Artifact

The inference application expects:

```text
models/plant_disease_resnet18.pth
```

The repository is configured to track `.pth` files with **Git LFS**:

```gitattributes
*.pth filter=lfs diff=lfs merge=lfs -text
```

The inference loader expects a self-contained checkpoint containing the model weights plus class/preprocessing metadata.

The conversion utility is available at:

```text
scripts/convert_checkpoint.py
```

It converts a compatible legacy ResNet18 state dictionary into the self-contained artifact format used by the API.

---

# 📊 Model Evaluation

The evaluation script calculates:

- Accuracy
- Macro F1
- Weighted F1
- Per-class precision
- Per-class recall
- Per-class F1
- Confusion matrix
- Number of validation samples
- Number of classes

Run:

```bash
python scripts/evaluate.py
```

The default evaluation output is:

```text
reports/evaluation.json
```

> Validation performance on a controlled dataset should not be treated as proof of real-world field performance. Lighting, camera quality, backgrounds, cultivars, disease stages, and other conditions can differ substantially in real environments.

---

# 🚀 Run the API Locally

## 1. Create a virtual environment

Python **3.10** is recommended and is the version used by the project configuration and CI.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

---

## 2. Install dependencies

```bash
pip install -r requirements-dev.txt
```

---

## 3. Ensure the model exists

Place the self-contained checkpoint at:

```text
models/plant_disease_resnet18.pth
```

If you are working with a legacy state-dict checkpoint, convert it first:

```bash
python scripts/convert_checkpoint.py
```

---

## 4. Start FastAPI

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

---

# 📚 Swagger / OpenAPI

FastAPI automatically provides interactive API documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

Alternative ReDoc documentation:

```text
http://127.0.0.1:8000/redoc
```

---

# 🔌 API Endpoints

## Health Check

### `GET /health`

Checks whether the model is loaded.

Example:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/plant_disease_resnet18.pth"
}
```

---

## Disease Prediction

### `POST /predict`

Upload a plant-leaf image as multipart form data.

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@leaf.jpg"
```

Example response:

```json
{
  "class_name": "Tomato___Late_blight",
  "clean_label": "Tomato - Late blight",
  "confidence": 96.42
}
```

The response is produced by the loaded model, so the actual class and confidence depend on the input image and trained artifact.

---

# 🛡️ API Validation

The prediction endpoint validates:

- File content type
- Empty uploads
- Maximum upload size
- Model availability
- Invalid image files

The default upload limit is **8 MB**.

It can be changed with:

```text
MAX_UPLOAD_BYTES
```

The model path can be changed with:

```text
MODEL_PATH
```

Allowed CORS origins can be configured with:

```text
ALLOWED_ORIGINS
```

---

# 🐳 Docker

## Build the image

```bash
docker build -t plant-disease-api .
```

## Run the container

```bash
docker run --rm -p 8000:8000 plant-disease-api
```

Then open:

```text
http://127.0.0.1:8000/docs
```

---

# 🧩 Docker Compose

The repository also contains:

```text
docker-compose.yml
```

The compose configuration expects the Docker image through:

```text
ECR_IMAGE
```

Example:

```bash
docker compose up -d
```

---

# ☁️ AWS Integration

The repository contains AWS utilities and an AWS-oriented CI/CD workflow.

## Amazon S3

Two utility scripts are provided:

```text
upload_dataset.py
upload_model.py
```

They can upload:

- Dataset files
- Trained model files

to an Amazon S3 bucket.

The repository also contains:

```text
test_s3.py
```

for testing AWS S3 connectivity.

For the upload utilities, install the AWS SDK if it is not already available:

```bash
pip install boto3
```

Configure AWS credentials using the AWS CLI or another secure credential mechanism.

> Never commit AWS access keys, secret keys, private keys, or `.env` files containing credentials.

---

# 📦 AWS ECR + EC2 Deployment

The repository contains:

```text
.github/workflows/deploy.yml
```

The deployment workflow is designed around:

```text
GitHub
   │
   ▼
GitHub Actions
   │
   ▼
Docker Build
   │
   ▼
Amazon ECR
   │
   ▼
Amazon EC2
   │
   ▼
Docker Compose
   │
   ▼
FastAPI + ResNet18
```

The deployment workflow includes validation, Docker image building, ECR publishing, EC2 deployment, health checks, and rollback handling.

AWS credentials and EC2 connection information are supplied through GitHub Actions secrets rather than hard-coded in the repository.

---

# 🧪 Testing

The project contains tests for both the model and API.

Run:

```bash
pytest -q
```

The tests cover areas including:

- Loading a self-contained model artifact
- Missing model handling
- Image prediction
- API response structure
- Invalid upload rejection
- Confidence range validation

The tests create small temporary ResNet18 artifacts, so they do not require the full training dataset.

---

# 🔄 GitHub Actions

## CI

Workflow:

```text
.github/workflows/ci.yml
```

It runs on pushes and pull requests to `main` and performs:

```text
Install dependencies
       │
       ▼
Compile Python files
       │
       ▼
Run pytest
```

---

## Model Training Workflow

Workflow:

```text
.github/workflows/train-model.yml
```

It is intended for separate model-training runs rather than normal CI.

The workflow performs:

```text
Download dataset
      ↓
Train ResNet18
      ↓
Evaluate model
      ↓
Upload model/reports as GitHub Actions artifacts
```

---

## Deployment Workflow

Workflow:

```text
.github/workflows/deploy.yml
```

It provides the AWS ECR → EC2 deployment pipeline described above.

---

# 📓 Notebook

The original training notebook is preserved at:

```text
notebooks/plant_disease_training.ipynb
```

The notebook is useful for understanding the learning process.

The `scripts/` directory contains the code intended for the reproducible project workflow.

---

# 🔐 Security Notes

Do not commit:

```text
.env
AWS access keys
AWS secret keys
EC2 private keys
SSH private keys
other production credentials
```

Use:

- Environment variables
- AWS IAM
- GitHub Actions Secrets
- EC2 IAM roles where appropriate
- AWS Secrets Manager when secret management is required

The repository's deployment workflow is designed to receive sensitive deployment values through GitHub Actions Secrets.

---

# ⚙️ Configuration

The FastAPI application supports environment-based configuration.

| Variable | Purpose | Default |
|---|---|---|
| `MODEL_PATH` | Path to model artifact | `models/plant_disease_resnet18.pth` |
| `MAX_UPLOAD_BYTES` | Maximum uploaded image size | `8388608` |
| `ALLOWED_ORIGINS` | CORS origins | `*` |

Example:

```bash
export MODEL_PATH=models/plant_disease_resnet18.pth
export MAX_UPLOAD_BYTES=8388608
export ALLOWED_ORIGINS=http://localhost:3000
```

Windows PowerShell:

```powershell
$env:MODEL_PATH="models/plant_disease_resnet18.pth"
$env:MAX_UPLOAD_BYTES="8388608"
$env:ALLOWED_ORIGINS="http://localhost:3000"
```

---

# 🧱 Technology Stack

| Layer | Technology |
|---|---|
| Programming language | Python 3.10 |
| Deep learning | PyTorch |
| Computer vision | Torchvision |
| Model | ResNet18 |
| API | FastAPI |
| Server | Uvicorn |
| Validation | Pydantic |
| Image processing | Pillow |
| Testing | Pytest |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Object storage | Amazon S3 |
| Container registry | Amazon ECR |
| Cloud compute | Amazon EC2 |
| Large model storage in Git | Git LFS |

---

# 📖 Project Documentation

Detailed project documentation is available at:

```text
docs/PROJECT_DOCUMENTATION.md
```

Model-specific information is available at:

```text
models/README.md
```

---

# ⚠️ Limitations

This project is an image-classification system and should not be treated as a replacement for professional agricultural diagnosis.

Important limitations include:

- Controlled datasets may differ from real field conditions.
- Similar visual symptoms can occur across different diseases.
- Image quality affects predictions.
- Backgrounds and lighting can affect model behavior.
- Confidence is the model's probability estimate, not a guarantee of correctness.
- Validation results do not automatically represent real-world field accuracy.

---

# 🚀 Project Summary

```text
Plant Image
    │
    ▼
Image Validation
    │
    ▼
Resize + Normalize
    │
    ▼
ResNet18
    │
    ▼
Plant Disease Classification
    │
    ▼
Class + Human-readable Label + Confidence
    │
    ▼
FastAPI REST API
    │
    ▼
Docker
    │
    ▼
AWS ECR / EC2
```

The project combines **computer vision, transfer learning, model evaluation, API development, testing, containerization, CI/CD, and AWS deployment tooling** into one end-to-end ML application.
