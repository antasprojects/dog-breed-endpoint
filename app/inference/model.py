import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2
from PIL import Image

from app.inference.preprocessing import prepare_image


class DogBreedModel:
    def __init__(self, model_path: Path, labels_path: Path, device: str = "cpu"):
        self.device = torch.device(device)

        with open(labels_path, "r", encoding="utf-8") as f:
            self.index_to_breed: Dict[str, str] = json.load(f)

        self.num_classes = len(self.index_to_breed)

        self.model = efficientnet_b2(weights=None)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features,
            self.num_classes
        )

        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)

        self.model.eval()
        self.model.to(self.device)

    @torch.inference_mode()
    def predict(self, img: Image.Image, top_k: int = 3) -> Tuple[str, float, List[Dict[str, Any]]]:

        x = prepare_image(img).to(self.device)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

        k = min(top_k, probs.numel())
        top_probs, top_idx = torch.topk(probs, k=k)

        top_probs = top_probs.cpu().tolist()
        top_idx = top_idx.cpu().tolist()

        topk = []
        for i, p in zip(top_idx, top_probs):
            breed = self.index_to_breed[str(i)]
            topk.append({"label": breed, "prob": float(p)})

        best = topk[0]
        return best["label"], best["prob"], topk