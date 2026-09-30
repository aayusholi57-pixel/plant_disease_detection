from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import PredictionResponse
from app.model import LeafClassifier

app = FastAPI(
    title="Plant Leaf Disease Detection API",
    description="REST API for real-time classification of plant leaf diseases using a fine-tuned ResNet18 model.",
    version="1.0.0",
    docs_url="/docs",      # Swagger UI URL
    redoc_url="/redoc"     # ReDoc URL
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance initialized at startup
classifier = None

@app.on_event("startup")
def load_classifier():
    global classifier
    classifier = LeafClassifier(
        model_path="models/plant_disease_resnet18.pth",
        data_dir="data/raw"
    )

@app.get("/health", tags=["Health Check"])
def health_check():
    return {"status": "healthy", "model_loaded": classifier is not None}

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_leaf_disease(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()
    raw_label, clean_label, confidence = classifier.predict(contents)

    return PredictionResponse(
        class_name=raw_label,
        clean_label=clean_label,
        confidence=confidence
    )