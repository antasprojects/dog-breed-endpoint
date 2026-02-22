from fastapi import FastAPI
from contextlib import asynccontextmanager
from pathlib import Path

from app.api import router
from app.inference.model import DogBreedModel



@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = DogBreedModel(
        model_path=Path("app/assets/efficientnet_b2_final.pth"),
        labels_path=Path("app/assets/labels.json"),
        device="cpu",
    )
    yield


app = FastAPI(title="Dog Breed Classifier", lifespan=lifespan)
app.include_router(router)