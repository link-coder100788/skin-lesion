"""Dataset loading, lesion-grouped split, and image transforms for HAM10000."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
URGENT = ["akiec", "bcc", "mel"]
URGENT_IDX = [CLASSES.index(c) for c in URGENT]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SEED = 42


def _default_img_size() -> int:
    """Input size of the deployed model: $IMG_SIZE, else models/train_log.json, else 224."""
    if os.environ.get("IMG_SIZE"):
        return int(os.environ["IMG_SIZE"])
    log = Path(__file__).resolve().parent.parent / "models/train_log.json"
    try:
        return int(json.load(open(log))["args"].get("img_size", 224))
    except (OSError, ValueError, KeyError):
        return 224


IMG_SIZE = _default_img_size()

# ISIC 2019 diagnoses -> HAM10000 classes. Invasive SCC joins akiec (which already holds SCC
# in situ / Bowen's); both are "urgent", so the triage definition is unchanged.
ISIC19_TO_HAM = {"NV": "nv", "MEL": "mel", "BCC": "bcc", "BKL": "bkl", "AK": "akiec",
                 "SCC": "akiec", "VASC": "vasc", "DF": "df"}

# Fitzpatrick17k labels that correspond to a HAM10000 class (invasive SCC is excluded:
# HAM10000's akiec only covers actinic keratosis / intraepithelial carcinoma).
FITZ_TO_HAM = {
    "melanoma": "mel", "superficial spreading melanoma ssm": "mel",
    "malignant melanoma": "mel", "lentigo maligna": "mel",
    "basal cell carcinoma": "bcc", "solid cystic basal cell carcinoma": "bcc",
    "basal cell carcinoma morpheiform": "bcc",
    "actinic keratosis": "akiec",
    "nevocytic nevus": "nv", "halo nevus": "nv", "congenital nevus": "nv",
    "seborrheic keratosis": "bkl",
    "dermatofibroma": "df",
    "pyogenic granuloma": "vasc", "granuloma pyogenic": "vasc",
}


def load_metadata(ham_dir: str | Path) -> pd.DataFrame:
    """Read HAM10000_metadata.csv and attach the image path for each row."""
    ham_dir = Path(ham_dir)
    df = pd.read_csv(ham_dir / "HAM10000_metadata.csv")
    paths = {p.stem: p for p in ham_dir.rglob("ISIC_*.jpg")}
    df["path"] = df["image_id"].map(lambda i: str(paths[i]))
    df["y"] = df["dx"].map(CLASSES.index)
    df["urgent"] = df["dx"].isin(URGENT).astype(int)
    return df


def lesion_split(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """70/15/15 train/val/test split, stratified by dx and grouped by lesion_id.

    Uses 20 stratified group folds: 3 for test, 3 for val, 14 for train, so no lesion
    appears in more than one split.
    """
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    split = pd.Series("train", index=df.index)
    for k, (_, idx) in enumerate(sgkf.split(df, df["dx"], groups=df["lesion_id"])):
        if k < 3:
            split.iloc[idx] = "test"
        elif k < 6:
            split.iloc[idx] = "val"
    return split


def load_image(path: str | Path, size: int | None = None) -> np.ndarray:
    """Load an RGB image, resized (full frame) to size x size (default IMG_SIZE), as uint8 HWC."""
    size = size or IMG_SIZE
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img)


def to_model_input(x_hwc: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) float in [0, 1] -> normalized (N, 3, H, W) tensor."""
    x = torch.as_tensor(x_hwc, dtype=torch.float32).permute(0, 3, 1, 2)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


class CachedImages(torch.utils.data.Dataset):
    """Holds pre-resized uint8 images in memory and applies tensor transforms."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2)  # N,3,H,W uint8
        self.labels = torch.tensor(np.asarray(labels), dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.images[i]
        if self.transform is not None:
            x = self.transform(x)
        x = x.float() / 255.0
        x = (x - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        return x, self.labels[i]


def train_transform(size: int | None = None):
    from torchvision.transforms import v2

    # Lesions have no "up": flips + 90-degree rotations. Colour jitter stays light
    # because colour carries diagnostic information.
    return v2.Compose([
        v2.RandomResizedCrop(size or IMG_SIZE, scale=(0.7, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomChoice([v2.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
    ])


def eval_transform(size: int | None = None):
    from torchvision.transforms import v2

    size = size or IMG_SIZE
    return v2.Resize((size, size), antialias=True)
