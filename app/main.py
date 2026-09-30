"""FastAPI application for plant disease inference."""
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.model import LeafClassifier
from app.schemas import PredictionResponse

MODEL_PATH = os.getenv("MODEL_PATH", "models/plant_disease_resnet18.pth")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
ALLOWED_ORIGINS = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",") if x.strip()]
classifier: LeafClassifier | None = None

@asynccontextmanager
async def lifespan(_: FastAPI):
    global classifier
    classifier = LeafClassifier(MODEL_PATH)
    yield
    classifier = None

app = FastAPI(
    title="Plant Leaf Disease Detection API",
    description="ResNet18 image-classification API for plant leaf disease detection.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOWED_ORIGINS != ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/health", tags=["Health"])
def health_check() -> dict[str, object]:
    return {"status": "healthy" if classifier is not None else "degraded", "model_loaded": classifier is not None, "model_path": str(Path(MODEL_PATH))}

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_leaf_disease(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the upload limit.")
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    try:
        raw_label, clean_label, confidence = classifier.predict(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictionResponse(class_name=raw_label, clean_label=clean_label, confidence=confidence)
