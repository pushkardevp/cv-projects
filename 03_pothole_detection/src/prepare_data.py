"""Convert the Kaggle pothole dataset (Pascal VOC XML boxes) to YOLO format and split 70/15/15.

Input : <src>/images/*.png, <src>/annotations/*.xml
Output: <dst>/{images,labels}/{train,val,test}/..., <dst>/data.yaml, <dst>/split_stats.json

YOLO label line: class x_center y_center width height  (all normalised to 0..1)

Usage: python prepare_data.py --src ../../data/pothole --dst ../../data/pothole_yolo
"""
import argparse
import glob
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET


def convert(xml_path):
    r = ET.parse(xml_path).getroot()
    W = float(r.find("size/width").text)
    H = float(r.find("size/height").text)
    lines = []
    for o in r.findall("object"):
        b = o.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return r.find("filename").text, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../../data/pothole")
    ap.add_argument("--dst", default="../../data/pothole_yolo")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    xmls = sorted(glob.glob(os.path.join(a.src, "annotations", "*.xml")))
    random.Random(a.seed).shuffle(xmls)
    n = len(xmls)
    splits = {"train": xmls[:int(.7 * n)], "val": xmls[int(.7 * n):int(.85 * n)], "test": xmls[int(.85 * n):]}
    stats = {}
    for s, files in splits.items():
        os.makedirs(os.path.join(a.dst, "images", s), exist_ok=True)
        os.makedirs(os.path.join(a.dst, "labels", s), exist_ok=True)
        boxes = 0
        for x in files:
            fn, lines = convert(x)
            stem = os.path.splitext(fn)[0]
            shutil.copy(os.path.join(a.src, "images", fn), os.path.join(a.dst, "images", s, fn))
            with open(os.path.join(a.dst, "labels", s, stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            boxes += len(lines)
        stats[s] = dict(images=len(files), boxes=boxes)
    with open(os.path.join(a.dst, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(a.dst)}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: pothole\n")
    json.dump(stats, open(os.path.join(a.dst, "split_stats.json"), "w"), indent=2)
    print(stats)


if __name__ == "__main__":
    main()
