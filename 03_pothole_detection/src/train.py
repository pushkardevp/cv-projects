"""Fine-tune a COCO-pretrained YOLOv8 detector on the pothole dataset, then evaluate on the held-out test split.

Usage:
  python train.py --data ../../data/pothole_yolo/data.yaml --model yolov8n.pt --epochs 50 --name yolov8n
  python train.py --data ../../data/pothole_yolo/data.yaml --model yolov8s.pt --epochs 50 --name yolov8s
Outputs go to ../results/runs/<name>/ (training curves, PR curve, confusion matrix, best.pt) and
../results/<name>_test_metrics.json (mAP on the test split, which is never used for training or model selection).
"""
import argparse
import json
import os

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../data/pothole_yolo/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="yolov8n")
    a = ap.parse_args()
    project = os.path.abspath("../results/runs")

    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, project=project, name=a.name,
                exist_ok=True, seed=0, patience=20, workers=8, plots=True, verbose=False)

    best = YOLO(os.path.join(project, a.name, "weights", "best.pt"))
    m = best.val(data=a.data, split="test", imgsz=a.imgsz, project=project, name=f"{a.name}_test",
                 exist_ok=True, plots=True, verbose=False)
    res = dict(model=a.model, epochs=a.epochs, imgsz=a.imgsz,
               precision=float(m.box.mp), recall=float(m.box.mr),
               mAP50=float(m.box.map50), mAP50_95=float(m.box.map),
               speed_ms_per_image=m.speed)
    with open(f"../results/{a.name}_test_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(res)


if __name__ == "__main__":
    main()
