"""CNN backbone, temperature scaling, and inference helpers."""
from __future__ import annotations

import numpy as np
import timm
import torch
import torch.nn as nn

from .data import CLASSES, to_model_input

BACKBONE = "efficientnet_b0"


def create_model(pretrained: bool = True) -> nn.Module:
    return timm.create_model(BACKBONE, pretrained=pretrained, num_classes=len(CLASSES))


def load_model(path, device="cpu") -> nn.Module:
    model = create_model(pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def logits_and_embeddings(model, x: torch.Tensor):
    """Return (logits, pooled embedding) for a normalized batch."""
    feats = model.forward_head(model.forward_features(x), pre_logits=True)
    return model.get_classifier()(feats), feats


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Learn one scalar T minimizing validation NLL of softmax(logits / T)."""
    lg = torch.as_tensor(logits, dtype=torch.float64)
    yy = torch.as_tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(lg / log_t.exp(), yy)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def softmax_np(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Predictor:
    """Wraps the CNN as a numpy function on (N, 224, 224, 3) images in [0, 1]."""

    def __init__(self, model, T: float, device, batch_size: int = 64):
        self.model, self.T, self.device, self.batch_size = model, T, device, batch_size

    @torch.no_grad()
    def logits(self, x_hwc: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(x_hwc), self.batch_size):
            x = to_model_input(x_hwc[i:i + self.batch_size]).to(self.device)
            out.append(self.model(x).float().cpu().numpy())
        return np.concatenate(out)

    def __call__(self, x_hwc: np.ndarray) -> np.ndarray:
        return softmax_np(self.logits(x_hwc), self.T)
