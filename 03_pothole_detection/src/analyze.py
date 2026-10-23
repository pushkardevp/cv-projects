"""Error analysis of the trained pothole detector on the TEST split.

  - recall by pothole size (small / medium / large box area) at IoU >= 0.5, conf >= 0.25
  - false positives per image
  - sample figure: ground truth (green) vs prediction (red) for good and bad test images

Usage: python analyze.py --weights ../results/runs/yolov8n/weights/best.pt --data ../../data/pothole_yolo
"""
import argparse
import glob
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


def iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None] - inter + 1e-9)


def load_gt(lbl, w, h):
    if not os.path.exists(lbl) or os.path.getsize(lbl) == 0:
        return np.zeros((0, 4))
    a = np.loadtxt(lbl, ndmin=2)[:, 1:]
    return np.stack([(a[:, 0] - a[:, 2] / 2) * w, (a[:, 1] - a[:, 3] / 2) * h,
                     (a[:, 0] + a[:, 2] / 2) * w, (a[:, 1] + a[:, 3] / 2) * h], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="../results/runs/yolov8n/weights/best.pt")
    ap.add_argument("--data", default="../../data/pothole_yolo")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="../results")
    ap.add_argument("--samples", default="../samples")
    a = ap.parse_args()
    model = YOLO(a.weights)
    imgs = sorted(glob.glob(os.path.join(a.data, "images", "test", "*")))
    buckets = {"small (<1% of image)": [0, 0], "medium (1-5%)": [0, 0], "large (>5%)": [0, 0]}
    fp_total, per_img = 0, []
    for p in imgs:
        im = cv2.imread(p)
        h, w = im.shape[:2]
        gt = load_gt(p.replace("images", "labels").rsplit(".", 1)[0] + ".txt", w, h)
        r = model.predict(im, conf=a.conf, verbose=False)[0]
        pr = r.boxes.xyxy.cpu().numpy()
        matched_g, matched_p = set(), set()
        if len(gt) and len(pr):
            m = iou(gt, pr)
            for gi, pi in sorted(zip(*np.where(m >= 0.5)), key=lambda t: -m[t]):
                if gi not in matched_g and pi not in matched_p:
                    matched_g.add(gi)
                    matched_p.add(pi)
        for gi, g in enumerate(gt):
            frac = (g[2] - g[0]) * (g[3] - g[1]) / (w * h)
            k = "small (<1% of image)" if frac < 0.01 else "medium (1-5%)" if frac < 0.05 else "large (>5%)"
            buckets[k][0] += 1
            buckets[k][1] += gi in matched_g
        fp = len(pr) - len(matched_p)
        fp_total += fp
        per_img.append(dict(path=p, gt=gt, pr=pr, tp=len(matched_g), fn=len(gt) - len(matched_g), fp=fp, im=im))

    res = dict(conf=a.conf, images=len(imgs), false_positives_per_image=fp_total / len(imgs),
               recall_by_size={k: dict(gt=v[0], detected=v[1], recall=v[1] / max(v[0], 1)) for k, v in buckets.items()})
    json.dump(res, open(os.path.join(a.out, "error_analysis.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))

    good = sorted([r for r in per_img if r["fn"] == 0 and r["fp"] == 0 and r["tp"] > 0], key=lambda r: -r["tp"])[:4]
    bad = sorted(per_img, key=lambda r: -(r["fn"] + r["fp"]))[:4]
    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    for x, r, kind in zip(ax.flat, good + bad, ["good"] * 4 + ["hard"] * 4):
        im = r["im"].copy()
        for b in r["gt"].astype(int):
            cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (0, 255, 0), 2)
        for b in r["pr"].astype(int):
            cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (0, 0, 255), 2)
        x.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        x.set_title(f"{kind}: TP {r['tp']}  FN {r['fn']}  FP {r['fp']}", fontsize=10)
        x.axis("off")
    plt.suptitle("Test images. Green = ground truth, red = YOLOv8 prediction (conf >= 0.25)")
    plt.tight_layout()
    plt.savefig(os.path.join(a.samples, "test_predictions.png"), dpi=80)
    plt.close()


if __name__ == "__main__":
    main()
