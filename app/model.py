"""Model loading and inference utilities."""
from __future__ import annotations
import io
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)

class LeafClassifier:
    """Load a self-contained ResNet18 checkpoint and classify leaf images."""
    def __init__(self, model_path: str = "models/plant_disease_resnet18.pth"):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact not found at {path}. Run the training pipeline first.")
        checkpoint: dict[str, Any] = torch.load(path, map_location=DEVICE)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError("Unsupported checkpoint format: expected self-contained artifact.")
        self.class_names = list(checkpoint.get("class_names", []))
        if not self.class_names:
            raise ValueError("Checkpoint contains no class_names metadata.")
        self.image_size = int(checkpoint.get("image_size", 224))
        mean = tuple(checkpoint.get("mean", DEFAULT_MEAN))
        std = tuple(checkpoint.get("std", DEFAULT_STD))
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(DEVICE)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def predict(self, image_bytes: bytes) -> tuple[str, str, float]:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("The uploaded file is not a valid image.") from exc
        tensor_img = self.transform(image).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor_img), dim=1)
            confidence, class_index = torch.max(probabilities, dim=1)
        raw_label = self.class_names[int(class_index.item())]
        clean_label = raw_label.replace("___", " - ").replace("_", " ")
        return raw_label, clean_label, round(float(confidence.item()) * 100, 2)
