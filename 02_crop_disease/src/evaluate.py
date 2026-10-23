"""Evaluate trained models on the held-out test set and explain predictions with Grad-CAM.

Produces in ../results:
  comparison.md            accuracy / macro-F1 of linear-probe vs fine-tuned model
  classification_report.txt per-class precision / recall / F1 (fine-tuned)
  confusion_matrix.png, training_curves.png
  robustness.md            accuracy under blur / brightness / noise / JPEG (lab images -> field conditions)
  gradcam_leaf_focus.json  fraction of Grad-CAM activation that falls on the leaf (not the background)
and in ../samples: gradcam_correct.png, gradcam_errors.png

Usage: python evaluate.py --data ../../data/tomato/tomato
"""
import argparse
import io
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torchvision import datasets, models, transforms

sys.path.insert(0, os.path.dirname(__file__))
from train import MEAN, STD

EVAL_TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                              transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def short(c):
    return c.replace("Tomato___", "").replace("_", " ").replace("Spider mites Two-spotted spider mite", "Spider mites")


def load(path, dev):
    ck = torch.load(path, map_location=dev)
    m = models.resnet18()
    m.fc = torch.nn.Linear(m.fc.in_features, len(ck["classes"]))
    m.load_state_dict(ck["state"])
    return m.to(dev).eval(), ck["classes"]


@torch.no_grad()
def predict_pil(model, imgs, dev, bs=64):
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.stack([EVAL_TF(im) for im in imgs[i:i + bs]]).to(dev)
        out.append(model(x).argmax(1).cpu())
    return torch.cat(out).numpy()


def perturb(im, kind):
    if kind == "blur":
        return im.filter(ImageFilter.GaussianBlur(3))
    if kind == "dark":
        return ImageEnhance.Brightness(im).enhance(0.5)
    if kind == "bright":
        return ImageEnhance.Brightness(im).enhance(1.6)
    if kind == "noise":
        a = np.asarray(im).astype(np.float32) + np.random.RandomState(0).normal(0, 25, (*im.size[::-1], 3))
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if kind == "jpeg_q10":
        b = io.BytesIO()
        im.save(b, "JPEG", quality=10)
        return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
    return im


def leaf_mask(rgb224):
    """Rough leaf segmentation: PlantVillage backgrounds are low-saturation grey; leaves are saturated."""
    hsv = cv2.cvtColor(rgb224, cv2.COLOR_RGB2HSV)
    m = (hsv[..., 1] > 50) & (hsv[..., 2] > 30)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return m.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../data/tomato/tomato")
    ap.add_argument("--results", default="../results")
    ap.add_argument("--samples", default="../samples")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    test = datasets.ImageFolder(os.path.join(a.data, "val"))
    pil = [Image.open(p).convert("RGB") for p, _ in test.samples]
    y = np.array(test.targets)

    # 1. comparison of the two training modes
    rows = ["| model | trainable params | epochs | train time (s) | test accuracy | test macro-F1 |", "|---|---|---|---|---|---|"]
    labels = {"linear": "linear probe (frozen backbone)", "finetune": "full fine-tune",
              "finetune_robust": "full fine-tune + blur/noise augmentation"}
    for mode in labels:
        d = os.path.join(a.results, mode)
        if not os.path.exists(os.path.join(d, "test_predictions.npz")):
            continue
        z = np.load(os.path.join(d, "test_predictions.npz"))
        log = json.load(open(os.path.join(d, "train_log.json")))
        p = z["p"].argmax(1)
        rows.append(f"| ResNet-18 {labels[mode]} | "
                    f"{log['trainable_params']:,} | {log['epochs']} | {log['train_time_s']:.0f} | "
                    f"{(p == z['y']).mean() * 100:.1f}% | {f1_score(z['y'], p, average='macro') * 100:.1f}% |")
    open(os.path.join(a.results, "comparison.md"), "w").write("\n".join(rows) + "\n")
    print("\n".join(rows))

    model, classes = load(os.path.join(a.results, "finetune", "model.pt"), dev)
    names = [short(c) for c in classes]
    pred = predict_pil(model, pil, dev)
    rep = classification_report(y, pred, target_names=names, digits=3)
    open(os.path.join(a.results, "classification_report.txt"), "w").write(rep)
    print(rep)

    # 2. confusion matrix
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > 50 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix, test set (n={len(y)}), accuracy {(pred == y).mean() * 100:.1f}%")
    plt.tight_layout()
    plt.savefig(os.path.join(a.results, "confusion_matrix.png"), dpi=90)
    plt.close()

    # 3. training curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for mode in labels:
        f = os.path.join(a.results, mode, "train_log.json")
        if os.path.exists(f):
            h = json.load(open(f))["history"]
            e = [r["epoch"] for r in h]
            ax[0].plot(e, [r["train_loss"] for r in h], "o-", label=mode)
            ax[1].plot(e, [r["val_acc"] * 100 for r in h], "o-", label=mode)
    ax[0].set_title("training loss")
    ax[1].set_title("validation accuracy (%)")
    for x in ax:
        x.set_xlabel("epoch")
        x.legend()
        x.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(a.results, "training_curves.png"), dpi=90)
    plt.close()

    # 4. robustness to image degradations
    rob_models = {"finetune": model}
    if os.path.exists(os.path.join(a.results, "finetune_robust", "model.pt")):
        rob_models["finetune_robust"] = load(os.path.join(a.results, "finetune_robust", "model.pt"), dev)[0]
    rows = ["| condition | " + " | ".join(rob_models) + " |", "|---" * (len(rob_models) + 1) + "|"]
    rob = {}
    for kind in ("clean", "blur", "dark", "bright", "noise", "jpeg_q10"):
        imgs = [perturb(im, kind) for im in pil]
        accs = [float((predict_pil(m, imgs, dev) == y).mean() * 100) for m in rob_models.values()]
        rob[kind] = dict(zip(rob_models, accs))
        rows.append(f"| {kind} | " + " | ".join(f"{v:.1f}%" for v in accs) + " |")
    open(os.path.join(a.results, "robustness.md"), "w").write("\n".join(rows) + "\n")
    print("\n".join(rows))

    # 5. Grad-CAM: on which pixels does the decision rely?
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    focus, vis_ok, vis_err = [], [], []
    rng = np.random.RandomState(0)
    order = rng.permutation(len(pil))
    for i in order:
        x = EVAL_TF(pil[i]).unsqueeze(0).to(dev)
        g = cam(input_tensor=x, targets=None)[0]  # 224x224 in [0,1], for the predicted class
        rgb = np.asarray(transforms.CenterCrop(224)(transforms.Resize(256)(pil[i]))).astype(np.float32) / 255
        m = leaf_mask((rgb * 255).astype(np.uint8))
        if 0.05 < m.mean() < 0.95:
            focus.append(dict(on_leaf=float(g[m].sum() / (g.sum() + 1e-8)), leaf_area=float(m.mean())))
        item = (show_cam_on_image(rgb, g, use_rgb=True), rgb, names[y[i]], names[pred[i]])
        if pred[i] == y[i] and len(vis_ok) < 8 and all(t[2] != names[y[i]] for t in vis_ok):
            vis_ok.append(item)
        if pred[i] != y[i] and len(vis_err) < 8:
            vis_err.append(item)
        if len(focus) >= 300 and len(vis_ok) >= 8:
            # errors are rare: keep scanning only for them
            if len(vis_err) >= min(8, int((pred != y).sum())):
                break
    f = np.array([r["on_leaf"] for r in focus])
    la = np.array([r["leaf_area"] for r in focus])
    lf = dict(images=len(f), mean_cam_on_leaf=float(f.mean()), mean_leaf_area=float(la.mean()),
              pct_images_cam_mostly_on_leaf=float((f > 0.5).mean() * 100))
    json.dump(dict(robustness=rob, gradcam=lf), open(os.path.join(a.results, "gradcam_leaf_focus.json"), "w"), indent=2)
    print(lf)

    for items, fn, ttl in ((vis_ok, "gradcam_correct.png", "correct"), (vis_err, "gradcam_errors.png", "misclassified")):
        if not items:
            continue
        fig, ax = plt.subplots(2, len(items), figsize=(2.6 * len(items), 5.6))
        ax = ax.reshape(2, -1)
        for k, (c, rgb, t, p) in enumerate(items):
            ax[0, k].imshow(rgb)
            ax[0, k].set_title(f"true: {t}", fontsize=8)
            ax[1, k].imshow(c)
            ax[1, k].set_title(f"pred: {p}", fontsize=8, color="green" if t == p else "red")
            ax[0, k].axis("off")
            ax[1, k].axis("off")
        fig.suptitle(f"Grad-CAM ({ttl}): red = regions that drove the prediction")
        plt.tight_layout()
        plt.savefig(os.path.join(a.samples, fn), dpi=90)
        plt.close()


if __name__ == "__main__":
    main()
