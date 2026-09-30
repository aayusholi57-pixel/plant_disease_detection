"""Evaluate a trained artifact and write metrics."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from scripts.dataset import MEAN, STD

def evaluate(args):
    checkpoint=torch.load(args.model,map_location="cpu"); class_names=list(checkpoint["class_names"])
    image_size=int(checkpoint.get("image_size",224))
    transform=transforms.Compose([transforms.Resize((image_size,image_size)),transforms.ToTensor(),
                                  transforms.Normalize(checkpoint.get("mean",MEAN),checkpoint.get("std",STD))])
    dataset=datasets.ImageFolder(args.data_dir,transform=transform)
    generator=torch.Generator().manual_seed(int(checkpoint.get("seed",42)))
    indices=torch.randperm(len(dataset),generator=generator).tolist(); split=int(len(indices)*(1-args.val_split))
    loader=DataLoader(Subset(dataset,indices[split:]),batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
    from app.model import LeafClassifier
    classifier=LeafClassifier(args.model); y_true=[]; y_pred=[]
    with torch.inference_mode():
        for images,labels in loader:
            logits=classifier.model(images); y_true.extend(labels.tolist()); y_pred.extend(logits.argmax(1).tolist())
    metrics={"accuracy":accuracy_score(y_true,y_pred),"macro_f1":f1_score(y_true,y_pred,average="macro",zero_division=0),
             "weighted_f1":f1_score(y_true,y_pred,average="weighted",zero_division=0),
             "classification_report":classification_report(y_true,y_pred,target_names=class_names,output_dict=True,zero_division=0),
             "confusion_matrix":confusion_matrix(y_true,y_pred).tolist(),"num_validation_samples":len(y_true),
             "num_classes":len(class_names)}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in metrics.items() if k!="classification_report"},indent=2)); return metrics
def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",default="data/raw"); p.add_argument("--model",default="models/plant_disease_resnet18.pth")
    p.add_argument("--output",default="reports/evaluation.json"); p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--val-split",type=float,default=0.2); p.add_argument("--num-workers",type=int,default=2); return p.parse_args()
if __name__=="__main__": evaluate(parse_args())
