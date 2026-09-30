from pathlib import Path
import io
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image
from torchvision import models
from app.model import LeafClassifier

def create_artifact(path: Path):
    model=models.resnet18(weights=None); model.fc=nn.Linear(model.fc.in_features,2)
    torch.save({"artifact_version":1,"model_name":"resnet18","state_dict":model.state_dict(),
                "class_names":["Apple___healthy","Apple___rust"],"num_classes":2,"image_size":224,
                "mean":(0.485,0.456,0.406),"std":(0.229,0.224,0.225)},path)

def test_predict_endpoint(tmp_path):
    artifact=tmp_path/"model.pth"; create_artifact(artifact)
    from app import main
    main.classifier=LeafClassifier(str(artifact))
    client=TestClient(main.app)
    image=Image.new("RGB",(64,64),"green"); buf=io.BytesIO(); image.save(buf,format="PNG")
    response=client.post("/predict",files={"file":("leaf.png",buf.getvalue(),"image/png")})
    assert response.status_code==200
    payload=response.json(); assert set(payload)=={"class_name","clean_label","confidence"}
    assert 0<=payload["confidence"]<=100

def test_rejects_non_image():
    from app import main
    main.classifier=object()
    client=TestClient(main.app)
    response=client.post("/predict",files={"file":("notes.txt",b"hello","text/plain")})
    assert response.status_code==400
