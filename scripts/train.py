"""Reproducible ResNet18 training entry point."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from scripts.dataset import MEAN, STD, build_loaders

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def build_model(num_classes: int, pretrained: bool = True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    for parameter in model.parameters(): parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_names = build_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
        image_size=args.image_size, val_split=args.val_split, seed=args.seed)
    model = build_model(len(class_names), not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    history=[]; best=-1.0
    output=Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs+1):
        model.train(); train_loss=correct=total=0
        for images, labels in train_loader:
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad(set_to_none=True); logits=model(images)
            loss=criterion(logits,labels); loss.backward(); optimizer.step()
            train_loss += loss.item()*images.size(0); correct += (logits.argmax(1)==labels).sum().item(); total += labels.size(0)
        model.eval(); val_loss=val_correct=val_total=0
        with torch.inference_mode():
            for images,labels in val_loader:
                images,labels=images.to(device),labels.to(device); logits=model(images); loss=criterion(logits,labels)
                val_loss += loss.item()*images.size(0); val_correct += (logits.argmax(1)==labels).sum().item(); val_total += labels.size(0)
        metrics={"epoch":epoch,"train_loss":train_loss/total,"train_accuracy":correct/total,"val_loss":val_loss/val_total,"val_accuracy":val_correct/val_total}
        history.append(metrics); print(json.dumps(metrics))
        if metrics["val_accuracy"] > best:
            best=metrics["val_accuracy"]
            torch.save({"artifact_version":1,"model_name":"resnet18","state_dict":model.state_dict(),
                        "class_names":class_names,"num_classes":len(class_names),"image_size":args.image_size,
                        "mean":MEAN,"std":STD,"seed":args.seed,"dataset":"PlantVillage / emmarex/plantdisease",
                        "best_val_accuracy":best,"history":history}, output)
    Path(args.history).parent.mkdir(parents=True, exist_ok=True)
    Path(args.history).write_text(json.dumps(history,indent=2),encoding="utf-8")
    print(f"Saved model artifact: {output}")

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",default="data/raw"); p.add_argument("--output",default="models/plant_disease_resnet18.pth")
    p.add_argument("--history",default="reports/training_history.json"); p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--batch-size",type=int,default=32); p.add_argument("--learning-rate",type=float,default=1e-3)
    p.add_argument("--val-split",type=float,default=0.2); p.add_argument("--image-size",type=int,default=224)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--num-workers",type=int,default=2)
    p.add_argument("--no-pretrained",action="store_true"); return p.parse_args()
if __name__=="__main__": train(parse_args())
