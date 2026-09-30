# Plant Disease Detection

Reproducible computer-vision system for plant leaf disease classification with PyTorch, ResNet18, FastAPI and Docker.

[![CI](https://github.com/aayusholi57-pixel/plant_disease_detection/actions/workflows/ci.yml/badge.svg)](https://github.com/aayusholi57-pixel/plant_disease_detection/actions/workflows/ci.yml)

## Highlights

- Transfer learning with ResNet18
- Deterministic train/validation split
- Image augmentation and ImageNet preprocessing
- Self-contained model artifact with class metadata
- Accuracy, macro-F1, weighted-F1 and confusion-matrix evaluation
- FastAPI + Swagger/OpenAPI inference
- Upload validation and size limits
- Docker packaging
- Automated tests and GitHub Actions CI
- Reproducible model-training workflow

## Structure

```text
app/                    API and inference
data/raw/               downloaded dataset, not committed
models/                 model artifact documentation
notebooks/              original learning notebook
scripts/                production training, evaluation and dataset pipeline
tests/                  model and API tests
docs/                   project documentation
.github/workflows/      CI and model-training automation
```

## Quick start

Python 3.10 is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

Download data:

```bash
python scripts/download_dataset.py
```

Train:

```bash
python scripts/train.py --epochs 5
```

Evaluate:

```bash
python scripts/evaluate.py
```

Test:

```bash
pytest -q
```

Run API:

```bash
uvicorn app.main:app --reload
```

Open Swagger at `http://127.0.0.1:8000/docs`.

## API

- `GET /health`
- `POST /predict` with an image multipart upload

Example response:

```json
{
  "class_name": "Tomato___Late_blight",
  "clean_label": "Tomato - Late blight",
  "confidence": 96.42
}
```

The example is illustrative; actual output depends on the trained artifact.

## Docker

```bash
docker build -t plant-disease-api .
docker run --rm -p 8000:8000 plant-disease-api
```

## Model artifact

Large `.pth` files are excluded from Git history. The training workflow creates a downloadable GitHub Actions artifact containing the checkpoint and evaluation reports.

The checkpoint is self-contained, so inference does not need `data/raw` to recover class names.

See `models/README.md` and `docs/PROJECT_DOCUMENTATION.md`.

## Evaluation

Every trained model is evaluated with accuracy, macro F1, weighted F1, per-class precision/recall/F1, confusion matrix and validation sample count.

Do not treat validation performance as proof of field performance.

## CI/CD

CI runs compile checks and tests on pushes and pull requests.

The training workflow downloads the dataset, trains, evaluates and publishes the model/report artifact.

## Notebook

The original notebook remains under `notebooks/` for learning history. `scripts/` is the reproducible production pipeline.
