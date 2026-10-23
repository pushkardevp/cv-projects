"""Tomato leaf disease classification with transfer learning (ResNet-18, ImageNet weights).

Data layout (Kaggle "tomatoleaf"): <data>/train/<class>/*.jpg (10 x 1000) and <data>/val/<class>/*.jpg (10 x 100).
We split the provided train folder 90/10 into train / validation (model selection) and keep the provided
"val" folder untouched as the TEST set (reported numbers).

Two modes (compare them):
  --mode linear    : freeze the pretrained backbone, train only the final layer (feature extraction)
  --mode finetune  : train all layers (fine-tuning)

  --robust-aug     : also augment with Gaussian blur and noise (more robust to poor field photos)

Usage: python train.py --data ../../data/tomato/tomato --mode finetune --epochs 5 --out ../results/finetune
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class AddGaussianNoise:
    def __init__(self, std_max=0.12, p=0.3):
        self.std_max, self.p = std_max, p

    def __call__(self, t):
        if torch.rand(1).item() < self.p:
            t = (t + torch.randn_like(t) * torch.rand(1).item() * self.std_max).clamp(0, 1)
        return t


def loaders(data, batch, seed=0, robust_aug=False):
    aug = [transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
           transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
           transforms.ColorJitter(0.2, 0.2, 0.2)]
    if robust_aug:  # simulate field-camera degradations: defocus blur and sensor noise
        aug.append(transforms.RandomApply([transforms.GaussianBlur(9, sigma=(0.5, 3.0))], p=0.3))
    post = [transforms.ToTensor()] + ([AddGaussianNoise()] if robust_aug else []) + [transforms.Normalize(MEAN, STD)]
    train_tf = transforms.Compose(aug + post)
    eval_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                  transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    full_train = datasets.ImageFolder(os.path.join(data, "train"), train_tf)
    full_eval = datasets.ImageFolder(os.path.join(data, "train"), eval_tf)
    test = datasets.ImageFolder(os.path.join(data, "val"), eval_tf)
    idx = np.random.RandomState(seed).permutation(len(full_train))
    n_val = len(idx) // 10
    tr, va = Subset(full_train, idx[n_val:]), Subset(full_eval, idx[:n_val])
    kw = dict(batch_size=batch, num_workers=8, pin_memory=True)
    return (DataLoader(tr, shuffle=True, **kw), DataLoader(va, **kw), DataLoader(test, **kw), full_train.classes)


def build(n_classes, mode):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if mode == "linear":
        for p in m.parameters():
            p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x.to(dev, non_blocking=True))
        ps.append(out.float().softmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../data/tomato/tomato")
    ap.add_argument("--mode", choices=["linear", "finetune"], default="finetune")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="../results/finetune")
    ap.add_argument("--robust-aug", action="store_true", help="add blur + noise augmentation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr, va, te, classes = loaders(a.data, a.batch, robust_aug=a.robust_aug)
    model = build(len(classes), a.mode).to(dev)
    lr = a.lr or (1e-3 if a.mode == "linear" else 3e-4)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=a.epochs * len(tr))
    scaler = torch.amp.GradScaler()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    hist, best = [], -1
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        tot, correct, n = 0.0, 0, 0
        for x, y in tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
                loss = loss_fn(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
        yv, pv = predict(model, va, dev)
        val_acc = float((pv.argmax(1) == yv).mean())
        hist.append(dict(epoch=ep + 1, train_loss=tot / n, train_acc=correct / n, val_acc=val_acc, time_s=time.time() - t0))
        print(hist[-1], flush=True)
        if val_acc > best:
            best = val_acc
            torch.save(dict(state=model.state_dict(), classes=classes, mode=a.mode), os.path.join(a.out, "model.pt"))

    ck = torch.load(os.path.join(a.out, "model.pt"))
    model.load_state_dict(ck["state"])
    yt, pt = predict(model, te, dev)
    np.savez(os.path.join(a.out, "test_predictions.npz"), y=yt, p=pt)
    json.dump(dict(classes=classes, history=hist, mode=a.mode, lr=lr, epochs=a.epochs,
                   train_time_s=time.time() - t0,
                   trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad)),
              open(os.path.join(a.out, "train_log.json"), "w"), indent=2)
    print("test acc", float((pt.argmax(1) == yt).mean()))


if __name__ == "__main__":
    main()
