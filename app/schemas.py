"""API response schemas."""
from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    class_name: str = Field(..., description="Original dataset class name.")
    clean_label: str = Field(..., description="Human-readable disease label.")
    confidence: float = Field(..., ge=0, le=100, description="Top-1 confidence percentage.")
