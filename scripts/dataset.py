"""Dataset preparation with a deterministic split."""
from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def build_datasets(data_dir="data/raw", image_size=224, val_split=0.2, seed=42):
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_base = datasets.ImageFolder(root=root, transform=train_transform)
    val_base = datasets.ImageFolder(root=root, transform=val_transform)
    if len(train_base) < 2:
        raise ValueError("Dataset must contain at least two images.")
    if len(train_base.classes) < 2:
        raise ValueError("Dataset must contain at least two classes.")
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1.")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_base), generator=generator).tolist()
    split = int(len(indices) * (1 - val_split))
    if split <= 0 or split >= len(indices):
        raise ValueError("Dataset is too small for the requested validation split.")
    return Subset(train_base, indices[:split]), Subset(val_base, indices[split:]), list(train_base.classes)

def build_loaders(data_dir="data/raw", batch_size=32, num_workers=2, **kwargs):
    train_ds, val_ds, class_names = build_datasets(data_dir=data_dir, **kwargs)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        class_names,
    )
