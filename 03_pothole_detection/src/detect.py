"""Run the trained pothole detector on your own image, folder or video.

Usage:
  python detect.py --source my_road.jpg
  python detect.py --source road_video.mp4 --conf 0.3
Results are saved to ../results/predict/
"""
import argparse
import os

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
ap.add_argument("--weights", default=os.path.join(HERE, "..", "models", "pothole_yolov8n.pt"))
ap.add_argument("--conf", type=float, default=0.25)
a = ap.parse_args()
YOLO(a.weights).predict(a.source, conf=a.conf, save=True, project=os.path.join(HERE, "..", "results"),
                        name="predict", exist_ok=True)
print("saved to", os.path.join(HERE, "..", "results", "predict"))
