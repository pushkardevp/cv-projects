"""Run the face landmarker on every DDD image and save EAR / MAR / blink score to a CSV.

Usage: python extract_features.py --data "../../data/ddd/Driver Drowsiness Dataset (DDD)" --out ../results/features.csv
"""
import argparse
import csv
import glob
import os
import re
import sys
from multiprocessing import Pool

import cv2
import mediapipe as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drowsiness as dz

_lm = None


def work(path):
    global _lm
    if _lm is None:
        _lm = dz.make_landmarker()
    img = cv2.imread(path)
    h, w = img.shape[:2]
    res = _lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    f = dz.features(res, w, h)
    name = os.path.basename(path)
    m = re.match(r"([A-Za-z]+)(\d+)", name)
    label = 1 if os.path.basename(os.path.dirname(path)) == "Drowsy" else 0
    if f is None:
        return [name, label, m.group(1), int(m.group(2)), 0, "", "", ""]
    return [name, label, m.group(1), int(m.group(2)), 1, f"{f['ear']:.5f}", f"{f['mar']:.5f}",
            "" if f["blink"] is None else f"{f['blink']:.5f}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../data/ddd/Driver Drowsiness Dataset (DDD)")
    ap.add_argument("--out", default="../results/features.csv")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.data, "Drowsy", "*.png")) + glob.glob(os.path.join(a.data, "Non Drowsy", "*.png")))
    with Pool(a.workers) as p, open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "drowsy", "group", "frame", "face_found", "ear", "mar", "blink"])
        for i, row in enumerate(p.imap(work, files, chunksize=64)):
            w.writerow(row)
            if i % 5000 == 0:
                print(i, len(files), flush=True)
    print("done", len(files))


if __name__ == "__main__":
    main()
