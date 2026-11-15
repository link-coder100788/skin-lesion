"""Fine-tune EfficientNet-B0 on HAM10000 and dump logits/embeddings for later stages.

Runs on a Kaggle GPU kernel (see kaggle/build_kernel.py) or locally for a smoke test:

    python -m src.train --ham data/ham --split splits/split.csv --out models --limit 300 --epochs 1
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

from .data import (CLASSES, FITZ_TO_HAM, ISIC19_TO_HAM, SEED, CachedImages, load_image,
                   load_metadata, train_transform)
from .model import create_model, get_device, logits_and_embeddings


def seed_everything(seed: int = SEED):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_images(paths, size: int) -> np.ndarray:
    with ThreadPoolExecutor(16) as ex:
        return np.stack(list(ex.map(lambda p: load_image(p, size), paths)))


def load_isic19(root, ham_ids: set) -> pd.DataFrame:
    """ISIC 2019 training images that are NOT HAM10000 images, mapped to HAM10000 classes."""
    root = Path(root)
    meta = next(p for p in root.rglob("train.csv") if "diagnosis" in open(p).readline())
    d = pd.read_csv(meta)
    d = d[~d["image_name"].isin(ham_ids) & d["diagnosis"].isin(ISIC19_TO_HAM)]
    files = {p.stem: p for p in root.rglob("*.jpg")}
    d = d[d["image_name"].isin(files)]
    out = pd.DataFrame({"image_id": d["image_name"].to_numpy(),
                        "path": [str(files[i]) for i in d["image_name"]],
                        "dx": d["diagnosis"].map(ISIC19_TO_HAM).to_numpy()})
    out["y"] = out["dx"].map(CLASSES.index)
    return out


def class_weights(y: np.ndarray) -> torch.Tensor:
    # weight = 1 / sqrt(freq): softer than inverse frequency, so the 115 df images do not
    # dominate the loss, but nv no longer swamps the minority classes.
    freq = np.maximum(np.bincount(y, minlength=len(CLASSES)), 1) / len(y)
    w = 1.0 / np.sqrt(freq)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, embs = [], []
    for x, _ in loader:
        # fp32 on purpose: under fp16 autocast EfficientNet-B3's activations overflowed to inf/NaN
        lg, em = logits_and_embeddings(model, x.to(device, non_blocking=True))
        logits.append(lg.float().cpu())
        embs.append(em.float().cpu())
    return torch.cat(logits).numpy(), torch.cat(embs).numpy()


def run_epoch(model, loader, criterion, opt, sched, scaler, device):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            loss = criterion(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if sched is not None:
            sched.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ham", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--fitz", default=None, help="Fitzpatrick17k dir (csv + images)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--head-epochs", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="subsample per split (smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--drop-path", type=float, default=0.0, help="stochastic depth rate")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--isic19", default=None,
                    help="ISIC 2019 dir (train.csv + jpgs): extra TRAINING images, HAM10000 duplicates removed")
    args = ap.parse_args()

    seed_everything()
    device = get_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("device:", device, flush=True)

    df = load_metadata(args.ham)
    split = pd.read_csv(args.split).set_index("image_id")["split"]
    df["split"] = df["image_id"].map(split)
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    if args.limit:
        parts = {s: d.sample(min(args.limit, len(d)), random_state=SEED) for s, d in parts.items()}

    # Extra training data: only the training split grows; val/test stay the HAM10000 splits.
    train_df = parts["train"]
    if args.isic19:
        extra = load_isic19(args.isic19, set(df["image_id"]))
        if args.limit:
            extra = extra.sample(min(args.limit, len(extra)), random_state=SEED)
        train_df = pd.concat([train_df, extra], ignore_index=True)
        print(f"isic2019: +{len(extra)} training images {extra['dx'].value_counts().to_dict()}", flush=True)

    t0 = time.time()
    size = args.img_size
    tr_imgs = load_images(train_df["path"], round(size * 8 / 7))  # crop source: 256 for 224
    ev_imgs = {s: load_images(parts[s]["path"], size) for s in ("train", "val", "test")}
    print(f"loaded images in {time.time() - t0:.0f}s", flush=True)

    kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda",
              persistent_workers=args.workers > 0)
    y_tr = train_df["y"].to_numpy()
    train_loader = torch.utils.data.DataLoader(
        CachedImages(tr_imgs, y_tr, train_transform(size)), shuffle=True, drop_last=True, **kw)
    eval_loaders = {s: torch.utils.data.DataLoader(
        CachedImages(ev_imgs[s], parts[s]["y"].to_numpy()), shuffle=False, **kw)
        for s in ev_imgs}

    model = create_model(pretrained=True, backbone=args.backbone, drop_path=args.drop_path).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_tr).to(device),
                                    label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    history, best = [], -1.0

    def evaluate(epoch, phase, loss):
        nonlocal best
        lg, _ = predict(model, eval_loaders["val"], device)
        bacc = balanced_accuracy_score(parts["val"]["y"], lg.argmax(1))
        history.append(dict(epoch=epoch, phase=phase, train_loss=loss, val_bacc=bacc))
        print(f"[{phase}] epoch {epoch} loss {loss:.4f} val_bacc {bacc:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if bacc > best:  # checkpoint on balanced accuracy, not plain accuracy
            best = bacc
            torch.save(model.state_dict(), out / "cnn.pt")

    # Phase 1: frozen backbone, train only the classifier head.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.get_classifier().parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(args.head_epochs):
        evaluate(ep, "head", run_epoch(model, train_loader, criterion, opt, None, scaler, device))

    # Phase 2: unfreeze everything, discriminative LRs, cosine schedule.
    for p in model.parameters():
        p.requires_grad = True
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": backbone, "lr": 1e-4}, {"params": head, "lr": 3e-4}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[1e-4, 3e-4], total_steps=args.epochs * len(train_loader),
        pct_start=0.1, anneal_strategy="cos")
    for ep in range(args.epochs):
        evaluate(args.head_epochs + ep, "full",
                 run_epoch(model, train_loader, criterion, opt, sched, scaler, device))

    # Reload the best checkpoint and dump logits + embeddings for every split.
    model.load_state_dict(torch.load(out / "cnn.pt", map_location=device))
    dumps = {}
    for s, loader in eval_loaders.items():
        lg, em = predict(model, loader, device)
        dumps[f"{s}_logits"], dumps[f"{s}_emb"] = lg, em.astype(np.float16)
        dumps[f"{s}_ids"] = parts[s]["image_id"].to_numpy()

    if args.fitz:
        fdir = Path(args.fitz)
        fmeta = pd.read_csv(next(fdir.rglob("fitzpatrick17k.csv")))
        fmeta = fmeta[fmeta["label"].isin(FITZ_TO_HAM)]
        files = {p.stem: p for p in fdir.rglob("*.jpg")}
        fmeta = fmeta[fmeta["md5hash"].isin(files)]
        if args.limit:
            fmeta = fmeta.sample(min(args.limit, len(fmeta)), random_state=SEED)
        fimgs = load_images([files[h] for h in fmeta["md5hash"]], size)
        fy = fmeta["label"].map(FITZ_TO_HAM).map(CLASSES.index).to_numpy()
        lg, em = predict(model, torch.utils.data.DataLoader(CachedImages(fimgs, fy), shuffle=False, **kw),
                         device)
        dumps["fitz_logits"], dumps["fitz_emb"] = lg, em.astype(np.float16)
        dumps["fitz_ids"] = fmeta["md5hash"].to_numpy()
        print(f"fitzpatrick17k: {len(fmeta)} matched images", flush=True)

    np.savez_compressed(out / "outputs.npz", **dumps)
    json.dump(dict(history=history, best_val_bacc=best, args=vars(args),
                   device=str(device), torch=torch.__version__),
              open(out / "train_log.json", "w"), indent=2)
    print(f"done in {time.time() - t0:.0f}s, best val bacc {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
