# Plant Disease Detection — Project Documentation

## Objective

Classify plant leaf images into disease/health categories and expose the model through a FastAPI REST service.

## Architecture

Dataset -> deterministic split -> augmentation -> pretrained ResNet18 -> classifier head -> validation -> self-contained artifact -> FastAPI -> Docker.

## Dataset

The pipeline downloads the public Kaggle `emmarex/plantdisease` dataset and converts it to ImageFolder layout under `data/raw`.

Run:

```bash
python scripts/download_dataset.py
```

The dataset is intentionally not committed to Git.

## Training

Defaults:
- ResNet18 pretrained weights
- 224x224 RGB
- ImageNet normalization
- horizontal flip and 15-degree random rotation
- Adam, learning rate 1e-3
- batch size 32
- deterministic 80/20 split
- seed 42
- five epochs
- frozen backbone with trainable classifier head

Run:

```bash
python scripts/train.py
```

## Artifact contract

The checkpoint is a dictionary, not a bare state_dict. It contains `state_dict`, `class_names`, `image_size`, `mean`, `std`, seed and training metadata. This prevents class-index drift during inference.

## Evaluation

Run:

```bash
python scripts/evaluate.py
```

Reports include accuracy, macro F1, weighted F1, per-class precision/recall/F1 and a confusion matrix.

## API

After placing the model artifact at `models/plant_disease_resnet18.pth`:

```bash
uvicorn app.main:app --reload
```

Swagger: `/docs`

Health: `/health`

Prediction: `POST /predict`

## Docker

```bash
docker build -t plant-disease-api .
docker run --rm -p 8000:8000 plant-disease-api
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

Tests do not require the full dataset or a trained model.

## CI/CD

`.github/workflows/ci.yml` compiles application code and runs tests on pushes and pull requests.

`.github/workflows/train-model.yml` downloads the dataset, trains, evaluates and uploads the resulting model and reports as a GitHub Actions artifact. Training is separated from normal CI because full computer-vision training is much slower than unit tests.

## Limitations

PlantVillage-style datasets are controlled datasets. Field photographs can differ in lighting, backgrounds, camera quality, cultivars and disease stages. A validation score is therefore not proof of real-world field performance.
