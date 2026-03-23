"""Base classes for VLM evaluation: EvalModel (similarity) and GenEvalModel (generative)."""

from abc import ABC
from typing import Any

import numpy as np
import torch


class EvalModel(ABC):
    """CLIP-style model: image/text features and similarity scores."""

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model = model.to(self.device)
        self.processor = processor
        self.get_image_features = getattr(model, "get_image_features", None)
        self.get_text_features = getattr(model, "get_text_features", None)
        self.get_similarity_scores = getattr(
            model, "get_similarity_scores",
            lambda **x: getattr(model(**x), "logits_per_image", None),
        )

    def get_all_image_feats(self, dataloader: Any) -> np.ndarray:
        all_feats = []
        with torch.no_grad():
            for d in dataloader:
                inputs = self.processor(images=d["images"], return_tensors="pt").to(self.device)
                feats = self.get_image_features(**inputs).detach().cpu().numpy()
                all_feats.append(feats)
        return np.concatenate(all_feats, axis=0)

    def get_all_text_feats(self, dataloader: Any) -> np.ndarray:
        all_feats = []
        with torch.no_grad():
            for d in dataloader:
                inputs = self.processor(
                    text=d["text"], return_tensors="pt", padding=True
                ).to(self.device)
                feats = self.get_text_features(**inputs).detach().cpu().numpy()
                all_feats.append(feats)
        return np.concatenate(all_feats, axis=0)

    def get_all_sim_scores(
        self, dataloader: Any, n_options: int | None = None
    ) -> np.ndarray:
        all_sims = []
        with torch.no_grad():
            for d in dataloader:
                images = d.get("images") or []
                if not images:
                    if n_options is not None:
                        all_sims.append(
                            np.full((1, n_options), np.nan, dtype=np.float64)
                        )
                    continue
                inputs = self.processor(
                    images=images,
                    text=d["text"],
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                sims = self.get_similarity_scores(**inputs)
                if sims is not None:
                    sims = sims.detach().cpu().numpy()
                all_sims.append(sims)
        return np.stack(all_sims, axis=0) if all_sims else np.array([]).reshape(0, 0)


class GenEvalModel(ABC):
    """LLaVA-style model: NTP/LL logits over options."""

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model = model.to(self.device)
        self.processor = processor

    def get_ntp_logits(self, image: Any, text: str) -> torch.Tensor:
        raise NotImplementedError

    def get_ll_logits(self, image: Any, text: str) -> torch.Tensor:
        raise NotImplementedError

    def get_all_sim_scores(self, dataloader: Any) -> np.ndarray:
        """Return per-trial, per-option scores (e.g. logits) for comparison scripts."""
        raise NotImplementedError
