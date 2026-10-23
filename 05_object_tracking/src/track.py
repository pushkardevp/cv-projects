"""Multi-object (pedestrian) tracking = pretrained YOLOv8 person detector + ByteTrack / BoT-SORT association.

Tracking-by-detection: each frame the detector gives person boxes; the tracker predicts every existing track
forward with a Kalman filter, matches predictions to detections by IoU (Hungarian algorithm), keeps unmatched
tracks alive for a while (occlusions) and starts new IDs for unmatched detections. ByteTrack additionally
re-uses LOW-confidence detections in a second matching round, which recovers partly occluded people.
BoT-SORT adds camera-motion compensation and (optionally) appearance features.

Writes MOT-challenge format: frame,id,x,y,w,h,conf,-1,-1,-1

Usage:
  python track.py --seq ../../data/mot17/MOT17/train/MOT17-02-FRCNN --model yolov8s.pt --tracker bytetrack.yaml \
                  --out ../results/tracks/yolov8s_bytetrack/MOT17-02.txt [--video out.mp4]
"""
import argparse
import colorsys
import configparser
import glob
import os
import time

import cv2
from ultralytics import YOLO


def color(i):
    r, g, b = colorsys.hsv_to_rgb((i * 0.618) % 1, 0.8, 1)
    return int(b * 255), int(g * 255), int(r * 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="MOT17 sequence folder (with img1/ and seqinfo.ini) or a video file")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.1, help="low threshold: ByteTrack uses low-score boxes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", help="optional annotated output video")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    if os.path.isdir(a.seq):
        frames = sorted(glob.glob(os.path.join(a.seq, "img1", "*.jpg")))
        ini = configparser.ConfigParser()
        ini.read(os.path.join(a.seq, "seqinfo.ini"))
        fps = float(ini["Sequence"]["frameRate"])
    else:
        frames, fps = None, cv2.VideoCapture(a.seq).get(cv2.CAP_PROP_FPS)
    model = YOLO(a.model)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    wr = None
    lines = []
    t0 = time.time()
    n = 0
    if frames is not None:
        if a.max_frames:
            frames = frames[:a.max_frames]
        reader = (cv2.imread(f) for f in frames)
    else:
        cap = cv2.VideoCapture(a.seq)
        reader = iter(lambda: cap.read()[1], None)
    # one frame at a time; persist=True keeps the tracker state (and IDs) between calls
    for fi, frame in enumerate(reader, start=1):
        r = model.track(frame, tracker=a.tracker, classes=[0], conf=a.conf, imgsz=a.imgsz, persist=True, verbose=False)[0]
        n += 1
        b = r.boxes
        img = r.orig_img if a.video else None
        if b is not None and b.id is not None:
            for (x1, y1, x2, y2), tid, c in zip(b.xyxy.tolist(), b.id.int().tolist(), b.conf.tolist()):
                lines.append(f"{fi},{tid},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{c:.3f},-1,-1,-1")
                if img is not None:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color(tid), 2)
                    cv2.putText(img, str(tid), (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color(tid), 2)
        if img is not None:
            cv2.putText(img, f"frame {fi}  tracks {0 if b is None or b.id is None else len(b.id)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            if wr is None:
                h, w = img.shape[:2]
                scale = 960 / w
                size = (960, int(h * scale))
                wr = cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            wr.write(cv2.resize(img, size))
    if wr:
        wr.release()
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{n} frames, {len(lines)} boxes, {n / (time.time() - t0):.1f} FPS -> {a.out}")
    with open(a.out.replace(".txt", "_fps.txt"), "w") as f:
        f.write(f"{n / (time.time() - t0):.2f}\n")


if __name__ == "__main__":
    main()
