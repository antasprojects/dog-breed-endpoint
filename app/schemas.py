from pydantic import BaseModel
from typing import List

class TopKItem(BaseModel):
    label: str
    prob: float

class PredictResponse(BaseModel):
    breed: str
    confidence: float
    top_k: List[TopKItem]
