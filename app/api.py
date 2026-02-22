from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from PIL import Image
import io

from app.schemas import PredictResponse, TopKItem

router = APIRouter()

@router.get("/")
def health():
    return {"status": "ok"}

@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/jpg"):
        raise HTTPException(status_code=415, detail="Unsupported image type")

    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")
    


    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model = request.app.state.model
    breed, conf, topk = model.predict(img, top_k=5)

    return PredictResponse(
        breed=breed,
        confidence=conf,
        top_k=[TopKItem(**x) for x in topk],
    )

