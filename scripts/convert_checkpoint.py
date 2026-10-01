from pathlib import Path

import torch

from scripts.dataset import MEAN, STD, build_datasets


MODEL_PATH = Path("models/plant_disease_resnet18.pth")
DATA_DIR = "data/raw"
IMAGE_SIZE = 224
SEED = 42


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print("Loading existing model...")
    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(state_dict, dict):
        raise ValueError("Expected a PyTorch state_dict.")

    print("Reading class names from dataset...")
    _, _, class_names = build_datasets(
        data_dir=DATA_DIR,
        image_size=IMAGE_SIZE,
        val_split=0.2,
        seed=SEED,
    )

    if "fc.weight" not in state_dict:
        raise ValueError("Checkpoint does not contain ResNet18 fc.weight.")

    num_classes = state_dict["fc.weight"].shape[0]

    if len(class_names) != num_classes:
        raise ValueError(
            f"Class mismatch: model has {num_classes} outputs, "
            f"but dataset has {len(class_names)} classes."
        )

    artifact = {
        "artifact_version": 1,
        "model_name": "resnet18",
        "state_dict": state_dict,
        "class_names": class_names,
        "num_classes": num_classes,
        "image_size": IMAGE_SIZE,
        "mean": MEAN,
        "std": STD,
        "seed": SEED,
        "dataset": "PlantVillage / emmarex/plantdisease",
    }

    backup_path = MODEL_PATH.with_suffix(".state_dict.pth")
    MODEL_PATH.rename(backup_path)

    torch.save(artifact, MODEL_PATH)

    print()
    print("SUCCESS")
    print(f"Classes: {num_classes}")
    print(f"Model: {MODEL_PATH}")
    print(f"Backup: {backup_path}")


if __name__ == "__main__":
    main()