"""Quantitative test of the departure warning using simulated lateral drift.

The sample videos contain no lane departures and no labels, so we create them:
for a frame, a homography that shifts the ROAD PLANE sideways by d metres
(warp to bird's-eye, translate, warp back) simulates the car drifting d metres.
The true offset of the shifted frame = offset of the original frame + d, so we
know the ground truth and can measure:
  - offset estimation error (MAE, metres)
  - warning accuracy / precision / recall vs the true |offset| > threshold

Usage: python drift_test.py --data ../../data/lane --out ../results --samples ../samples
"""
import argparse
import glob
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import lane_detection as ld

SHIFTS_M = np.round(np.arange(-1.2, 1.21, 0.1), 2)


def estimate(img, lane_w=None):
    """Single-frame estimate. lane_w = lane width remembered from earlier frames (as the video tracker would)."""
    h, w = img.shape[:2]
    tr = ld.LaneTracker(smooth=False)
    tr.lane_w = lane_w
    _, info = ld.process_frame(img, tr, ld.birdseye_matrix(w, h))
    return info


def shifted(img, d_m, px_per_m):
    h, w = img.shape[:2]
    M = ld.birdseye_matrix(w, h)
    T = np.array([[1, 0, -d_m * px_per_m], [0, 1, 0], [0, 0, 1]], np.float64)  # car moves right = scene moves left
    H = np.linalg.inv(M) @ T @ M
    return cv2.warpPerspective(img, H, (w, h))


def frames(data, every=25):
    for p in sorted(glob.glob(os.path.join(data, "*.jpg"))):
        yield os.path.basename(p), cv2.imread(p)
    for v in sorted(glob.glob(os.path.join(data, "*.mp4"))):
        cap = cv2.VideoCapture(v)
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i % every == 0:
                yield f"{os.path.basename(v)}#{i}", f
            i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../data/lane")
    ap.add_argument("--out", default="../results")
    ap.add_argument("--samples", default="../samples")
    a = ap.parse_args()
    thr = ld.DEPARTURE_THRESHOLD * ld.LANE_WIDTH_M
    recs = []
    demo = None
    for name, img in frames(a.data):
        base = estimate(img)
        if base["lane_w"] is None:  # need both lines in the original frame to know the truth
            continue
        px_per_m = base["lane_w"] / ld.LANE_WIDTH_M
        base_off = base["offset"] * ld.LANE_WIDTH_M
        for d in SHIFTS_M:
            s = shifted(img, d, px_per_m)
            info = estimate(s, base["lane_w"])
            true = base_off + d
            est = info["offset"] * ld.LANE_WIDTH_M if info["offset"] is not None else None
            recs.append(dict(frame=name, shift=float(d), true=float(true), est=est,
                             true_warn=bool(abs(true) > thr), pred_warn=bool(info["warn"])))
        if name.startswith("solidWhiteCurve"):
            tr = ld.LaneTracker(False)
            tr.lane_w = base["lane_w"]
            demo = [cv2.cvtColor(ld.process_frame(shifted(img, d, px_per_m), tr,
                                                  ld.birdseye_matrix(img.shape[1], img.shape[0]))[0], cv2.COLOR_BGR2RGB)
                    for d in (-1.0, 0.0, 1.2)]

    det = [r for r in recs if r["est"] is not None]
    err = np.array([abs(r["est"] - r["true"]) for r in det])
    tp = sum(r["true_warn"] and r["pred_warn"] for r in recs)
    fp = sum(not r["true_warn"] and r["pred_warn"] for r in recs)
    fn = sum(r["true_warn"] and not r["pred_warn"] for r in recs)
    tn = sum(not r["true_warn"] and not r["pred_warn"] for r in recs)
    summary = dict(n_frames=len({r["frame"] for r in recs}), n_tests=len(recs), detected_pct=100 * len(det) / len(recs),
                   offset_mae_m=float(err.mean()), offset_p90_err_m=float(np.percentile(err, 90)),
                   warning_accuracy=(tp + tn) / len(recs), warning_precision=tp / max(tp + fp, 1),
                   warning_recall=tp / max(tp + fn, 1), confusion=dict(tp=tp, fp=fp, fn=fn, tn=tn),
                   threshold_m=thr)
    with open(os.path.join(a.out, "drift_test.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # error vs applied shift
    plt.figure(figsize=(7, 3.5))
    by = {d: [abs(r["est"] - r["true"]) for r in det if r["shift"] == d] for d in SHIFTS_M}
    plt.plot(SHIFTS_M, [np.mean(by[d]) if by[d] else np.nan for d in SHIFTS_M], "o-")
    plt.xlabel("simulated lateral drift (m)")
    plt.ylabel("mean |offset error| (m)")
    plt.title("Offset estimation error vs simulated drift")
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "drift_error.png"), dpi=90)
    plt.close()

    if demo:
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.4))
        for x, im, t in zip(ax, demo, ("simulated drift 1.0 m left", "original frame", "simulated drift 1.2 m right")):
            x.imshow(im)
            x.set_title(t)
            x.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(a.samples, "departure_warning_demo.png"), dpi=90)
        plt.close()


if __name__ == "__main__":
    main()
