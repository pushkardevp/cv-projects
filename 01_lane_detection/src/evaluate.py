"""Run the pipeline on all videos and images under three configurations and compute metrics.

There are no ground-truth lane labels in the Udacity sample videos, so we use label-free metrics:
  - detection rate: % of frames where the raw Hough fit found the left / right / both lines
  - jitter: mean frame-to-frame change (px) of the output lane line x-position at the bottom and top rows
  - lane-width stability: coefficient of variation of the bird's-eye lane width (a real lane has constant width)
  - warning rate: % of frames with a departure warning (should be ~0 here, the car stays in lane)

Usage: python evaluate.py --data ../../data/lane --out ../results
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
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import lane_detection as ld

CONFIGS = {"full": dict(color=True, smooth=True),
           "no_color_mask": dict(color=False, smooth=True),
           "no_smoothing": dict(color=True, smooth=False)}


def run_video(path, color, smooth, writer_path=None, gif_path=None):
    cap = cv2.VideoCapture(path)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    M = ld.birdseye_matrix(w, h)
    tr = ld.LaneTracker(smooth)
    wr = cv2.VideoWriter(writer_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) if writer_path else None
    gif = []
    rows = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        vis, info = ld.process_frame(f, tr, M, color)
        if wr:
            wr.write(vis)
        if gif_path and i % 3 == 0 and len(gif) < 60:
            gif.append(Image.fromarray(cv2.cvtColor(cv2.resize(vis, (480, int(480 * h / w))), cv2.COLOR_BGR2RGB)))
        rows.append(info)
        i += 1
    if wr:
        wr.release()
    if gif:
        gif[0].save(gif_path, save_all=True, append_images=gif[1:], duration=120, loop=0, optimize=True)
    return rows, w, h


def metrics(rows, w, h):
    yb, yt = ld.Y_BOTTOM * h, ld.Y_TOP * h
    n = len(rows)
    L = np.mean([r["left_found"] for r in rows]) * 100
    R = np.mean([r["right_found"] for r in rows]) * 100
    B = np.mean([r["left_found"] and r["right_found"] for r in rows]) * 100
    jit = []
    for side in ("left", "right"):
        xs = np.array([[ld.x_at(r[side], yb), ld.x_at(r[side], yt)] if r[side] is not None else [np.nan, np.nan]
                       for r in rows])
        jit.append(np.nanmean(np.abs(np.diff(xs, axis=0)), axis=0))
    jit = np.mean(jit, axis=0)
    lw = np.array([r["lane_w"] for r in rows if r["lane_w"] is not None])
    warn = np.mean([r["warn"] for r in rows]) * 100
    off = np.array([r["offset"] for r in rows if r["offset"] is not None])
    return dict(frames=n, left_det_pct=L, right_det_pct=R, both_det_pct=B,
                jitter_bottom_px=float(jit[0]), jitter_top_px=float(jit[1]),
                lane_width_cv_pct=float(lw.std() / lw.mean() * 100) if len(lw) else None,
                warning_pct=warn, mean_abs_offset_m=float(np.mean(np.abs(off)) * ld.LANE_WIDTH_M) if len(off) else None)


def pipeline_figure(img_path, out_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    M = ld.birdseye_matrix(w, h)
    cm = ld.color_mask(img)
    e = ld.edges(img, True)
    r = ld.roi_mask(e)
    segs = cv2.HoughLinesP(r, 1, np.pi / 180, ld.HOUGH_THRESHOLD, minLineLength=ld.HOUGH_MIN_LEN, maxLineGap=ld.HOUGH_MAX_GAP)
    hough = img.copy()
    if segs is not None:
        for x1, y1, x2, y2 in segs.reshape(-1, 4):
            cv2.line(hough, (x1, y1), (x2, y2), (0, 0, 255), 3)
    poly = np.array([[(int(x * w), int(y * h)) for x, y in ld.ROI]], np.int32)
    cv2.polylines(hough, poly, True, (0, 255, 0), 2)
    bird = cv2.warpPerspective(img, M, (w, h))
    final, _ = ld.process_frame(img, ld.LaneTracker(), M)
    panels = [("1. input", img), ("2. colour mask (HLS)", cm), ("3. Canny edges (masked)", e),
              ("4. Hough segments + ROI", hough), ("5. bird's-eye warp", bird), ("6. output", final)]
    fig, ax = plt.subplots(2, 3, figsize=(15, 6.5))
    for a, (t, im) in zip(ax.flat, panels):
        a.imshow(im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2RGB), cmap="gray")
        a.set_title(t)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=90)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../data/lane")
    ap.add_argument("--out", default="../results")
    ap.add_argument("--samples", default="../samples")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.samples, exist_ok=True)

    results = {}
    for vid in sorted(glob.glob(os.path.join(a.data, "*.mp4"))):
        name = os.path.splitext(os.path.basename(vid))[0]
        for cfg, kw in CONFIGS.items():
            full = cfg == "full"
            rows, w, h = run_video(vid, kw["color"], kw["smooth"],
                                   writer_path=os.path.join(a.out, f"{name}_out.mp4") if full else None,
                                   gif_path=os.path.join(a.samples, f"{name}.gif") if full else None)
            results.setdefault(name, {})[cfg] = metrics(rows, w, h)
            if full:  # offset trace plot
                off = [r["offset"] * ld.LANE_WIDTH_M if r["offset"] is not None else np.nan for r in rows]
                plt.figure(figsize=(8, 2.8))
                plt.plot(off)
                for s in (1, -1):
                    plt.axhline(s * ld.DEPARTURE_THRESHOLD * ld.LANE_WIDTH_M, color="r", ls="--")
                plt.xlabel("frame")
                plt.ylabel("lateral offset (m)")
                plt.title(f"{name}: estimated offset from lane centre (red = warning threshold)")
                plt.tight_layout()
                plt.savefig(os.path.join(a.out, f"{name}_offset.png"), dpi=90)
                plt.close()
            print(name, cfg, results[name][cfg])

    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # markdown table
    lines = ["| video | config | frames | left % | right % | both % | jitter bottom (px) | jitter top (px) | lane width CV % | warnings % |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for v, d in results.items():
        for c, m in d.items():
            lines.append(f"| {v} | {c} | {m['frames']} | {m['left_det_pct']:.1f} | {m['right_det_pct']:.1f} | {m['both_det_pct']:.1f} | "
                         f"{m['jitter_bottom_px']:.1f} | {m['jitter_top_px']:.1f} | {m['lane_width_cv_pct']:.1f} | {m['warning_pct']:.1f} |")
    with open(os.path.join(a.out, "metrics.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # still images
    for img in sorted(glob.glob(os.path.join(a.data, "*.jpg"))):
        name = os.path.splitext(os.path.basename(img))[0]
        im = cv2.imread(img)
        vis, _ = ld.process_frame(im, ld.LaneTracker(), ld.birdseye_matrix(im.shape[1], im.shape[0]))
        cv2.imwrite(os.path.join(a.samples, f"{name}_out.jpg"), vis)
    first = sorted(glob.glob(os.path.join(a.data, "*.jpg")))[0]
    pipeline_figure(first, os.path.join(a.samples, "pipeline_stages.png"))


if __name__ == "__main__":
    main()
