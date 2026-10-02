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
            raise FileNotFoundError(
                f"Model artifact not found at {path}. Run the training pipeline first."
            )

        # Explicitly set weights_only=False for PyTorch 2.6+ compatibility
        checkpoint: dict[str, Any] = torch.load(path, map_location=DEVICE, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("Unsupported checkpoint format: expected a checkpoint dictionary.")

        # Extract state dict safely from common wrapper keys
        if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Strip out any stray non-tensor metadata keys (like classes, epochs, etc.)
        state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}

        if not state_dict:
            raise ValueError("Unsupported checkpoint format: missing model weights tensors.")

        # Extract class names / metadata
        class_metadata = checkpoint.get("class_names") or checkpoint.get("classes")
        if not class_metadata and isinstance(checkpoint.get("state_dict"), dict):
            class_metadata = checkpoint["state_dict"].get("classes")
        if isinstance(class_metadata, dict):
            class_metadata = [class_metadata[key] for key in sorted(class_metadata)]
        
        self.class_names = list(class_metadata or [])
        if not self.class_names:
            raise ValueError("Checkpoint contains no class_names or classes metadata.")

        self.image_size = int(checkpoint.get("image_size", 224))
        mean = tuple(checkpoint.get("mean", DEFAULT_MEAN))
        std = tuple(checkpoint.get("std", DEFAULT_STD))

        # Initialize ResNet18 architecture
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))

        # Handle potential DataParallel 'module.' prefix
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

        self.model.load_state_dict(state_dict)
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