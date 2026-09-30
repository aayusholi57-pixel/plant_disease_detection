from pathlib import Path
import io
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from app.model import LeafClassifier

def make_artifact(path: Path):
    model=models.resnet18(weights=None); model.fc=nn.Linear(model.fc.in_features,2)
    torch.save({"artifact_version":1,"model_name":"resnet18","state_dict":model.state_dict(),
                "class_names":["Apple___healthy","Apple___rust"],"num_classes":2,"image_size":224,
                "mean":(0.485,0.456,0.406),"std":(0.229,0.224,0.225)},path)

def image_bytes():
    image=Image.new("RGB",(64,64),"green"); buf=io.BytesIO(); image.save(buf,format="PNG"); return buf.getvalue()

def test_classifier_loads_self_contained_artifact(tmp_path):
    artifact=tmp_path/"model.pth"; make_artifact(artifact); classifier=LeafClassifier(str(artifact))
    raw,clean,confidence=classifier.predict(image_bytes())
    assert raw in {"Apple___healthy","Apple___rust"}; assert "Apple" in clean; assert 0<=confidence<=100

def test_missing_artifact_fails_clearly(tmp_path):
    try: LeafClassifier(str(tmp_path/"missing.pth")); assert False
    except FileNotFoundError as exc: assert "Model artifact not found" in str(exc)
