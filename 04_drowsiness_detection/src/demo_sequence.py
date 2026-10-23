"""Play the consecutive DDD frames of one person through the live DrowsinessMonitor and save an annotated GIF/MP4.
Shows the temporal logic (consecutive closed frames, PERCLOS) that a single image cannot.

Usage: python demo_sequence.py --group M --start 1 --n 150 --out ../samples/demo_drowsy_M.gif
"""
import argparse
import glob
import os
import re
import sys

import cv2
import mediapipe as mp
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drowsiness as dz

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../../data/ddd/Driver Drowsiness Dataset (DDD)")
ap.add_argument("--group", required=True, help="file prefix, e.g. M (drowsy set) or b (non-drowsy set)")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--out", required=True)
a = ap.parse_args()

sub = "Drowsy" if a.group[0].isupper() else "Non Drowsy"
files = sorted(glob.glob(os.path.join(a.data, sub, f"{a.group}*.png")),
               key=lambda p: int(re.match(r"[A-Za-z]+(\d+)", os.path.basename(p)).group(1)))
files = [f for f in files if re.fullmatch(rf"{a.group}\d+\.png", os.path.basename(f))][a.start:a.start + a.n]
ear_t, mar_t = dz.load_thresholds()
mon = dz.DrowsinessMonitor(ear_t, mar_t)
lm = dz.make_landmarker()
frames = []
alerts = 0
for p in files:
    img = cv2.imread(p)
    img = cv2.resize(img, (454, 454))
    h, w = img.shape[:2]
    f = dz.features(lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))), w, h)
    st = mon.update(f)
    alerts += st["alert"]
    vis = dz.draw(img, f, st)
    cv2.putText(vis, f"{'drowsy' if sub == 'Drowsy' else 'alert'} set, person {a.group}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    frames.append(Image.fromarray(cv2.cvtColor(cv2.resize(vis, (256, 256)), cv2.COLOR_BGR2RGB)))
frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)
print(f"{len(frames)} frames, alert in {alerts} -> {a.out}")
