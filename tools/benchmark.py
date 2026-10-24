"""Measure per-frame inference speed of each project's model/pipeline (batch 1, after warm-up).

Usage (from the repository root, datasets downloaded): python tools/benchmark.py
"""
import glob
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 200


def bench(name, fn, frames, hw):
    for f in frames[:10]:
        fn(f)  # warm-up
    t = time.perf_counter()
    for i in range(N):
        fn(frames[i % len(frames)])
    ms = (time.perf_counter() - t) / N * 1000
    print(f"| {name} | {hw} | {ms:.1f} ms | {1000 / ms:.0f} FPS |", flush=True)


def main():
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("| project | hardware | latency per frame | throughput |\n|---|---|---|---|")

    sys.path.insert(0, os.path.join(ROOT, "01_lane_detection", "src"))
    import lane_detection as ld
    cap = cv2.VideoCapture(os.path.join(ROOT, "data", "lane", "challenge.mp4"))
    fr = [cap.read()[1] for _ in range(60)]
    M = ld.birdseye_matrix(fr[0].shape[1], fr[0].shape[0])
    tr = ld.LaneTracker()
    bench("01 lane detection (full pipeline, 1280x720)", lambda f: ld.process_frame(f, tr, M), fr, "CPU")

    sys.path.insert(0, os.path.join(ROOT, "02_crop_disease", "src"))
    from PIL import Image
    from evaluate import EVAL_TF, load
    model, _ = load(os.path.join(ROOT, "02_crop_disease", "results", "finetune_robust", "model.pt"), "cuda")
    ims = [Image.open(p).convert("RGB") for p in sorted(glob.glob(os.path.join(ROOT, "data", "tomato", "tomato", "val", "*", "*")))[:60]]

    def crop(im):
        with torch.no_grad():
            model(EVAL_TF(im).unsqueeze(0).cuda()).argmax(1).item()
    bench("02 crop disease (ResNet-18, preprocess + forward)", crop, ims, gpu)

    from ultralytics import YOLO
    yolo = YOLO(os.path.join(ROOT, "03_pothole_detection", "models", "pothole_yolov8n.pt"))
    pims = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(ROOT, "data", "pothole_yolo", "images", "test", "*")))[:60]]
    bench("03 pothole (YOLOv8n, 640 px, end-to-end)", lambda f: yolo.predict(f, verbose=False), pims, gpu)

    sys.path.insert(0, os.path.join(ROOT, "04_drowsiness_detection", "src"))
    import mediapipe as mp
    import drowsiness as dz
    lm = dz.make_landmarker()
    dims = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in
            sorted(glob.glob(os.path.join(ROOT, "data", "ddd", "Driver Drowsiness Dataset (DDD)", "Drowsy", "*.png")))[:60]]
    mon = dz.DrowsinessMonitor()
    bench("04 drowsiness (face landmarks + EAR + temporal logic, 227 px)",
          lambda f: mon.update(dz.features(lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=f)), 227, 227)), dims, "CPU")


if __name__ == "__main__":
    main()
