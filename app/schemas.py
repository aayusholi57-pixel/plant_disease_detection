from pydantic import BaseModel

class PredictionResponse(BaseModel):
    class_name: str
    clean_label: str
    confidence: float