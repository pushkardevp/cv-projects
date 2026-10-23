"""Classify one leaf photo and save a Grad-CAM explanation.

Usage: python predict.py --image leaf.jpg [--model ../results/finetune_robust/model.pt] [--out gradcam.png]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import EVAL_TF, load, short

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "..", "results", "finetune_robust", "model.pt"))
ap.add_argument("--out", default="gradcam.png")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
model, classes = load(a.model, dev)
im = Image.open(a.image).convert("RGB")
x = EVAL_TF(im).unsqueeze(0).to(dev)
with torch.no_grad():
    prob = model(x).softmax(1)[0].cpu().numpy()
top = np.argsort(-prob)[:3]
for i in top:
    print(f"{short(classes[i]):30s} {prob[i] * 100:5.1f}%")
g = GradCAM(model=model, target_layers=[model.layer4[-1]])(input_tensor=x, targets=None)[0]
rgb = np.asarray(transforms.CenterCrop(224)(transforms.Resize(256)(im))).astype(np.float32) / 255
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(rgb)
ax[0].set_title("input")
ax[1].imshow(show_cam_on_image(rgb, g, use_rgb=True))
ax[1].set_title(f"{short(classes[top[0]])} ({prob[top[0]] * 100:.1f}%)")
for x_ in ax:
    x_.axis("off")
plt.tight_layout()
plt.savefig(a.out, dpi=100)
print("saved", a.out)
