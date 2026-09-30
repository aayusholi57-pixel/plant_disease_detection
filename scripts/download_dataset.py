"""Download and unpack the public Kaggle PlantDisease dataset."""
from __future__ import annotations
import argparse, io, shutil, urllib.request, zipfile
from pathlib import Path
DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/emmarex/plantdisease"
def main(output_dir="data/raw"):
    target=Path(output_dir); target.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(DATASET_URL,timeout=120) as response:
        archive=zipfile.ZipFile(io.BytesIO(response.read()))
    temp=target.parent/"_dataset_extract"
    if temp.exists(): shutil.rmtree(temp)
    temp.mkdir(parents=True); archive.extractall(temp)
    candidates=[p for p in temp.rglob("*") if p.is_dir()]
    source=next((p for p in candidates if any(c.is_dir() for c in p.iterdir()) and
                 any(c.suffix.lower() in {".jpg",".jpeg",".png"} for c in p.rglob("*"))),None)
    if source is None: raise RuntimeError("Could not locate ImageFolder-compatible data.")
    for child in source.iterdir():
        dest=target/child.name
        if dest.exists(): shutil.rmtree(dest)
        shutil.copytree(child,dest)
    shutil.rmtree(temp); print(f"Dataset ready at {target}")
if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",default="data/raw")
    main(parser.parse_args().output_dir)
