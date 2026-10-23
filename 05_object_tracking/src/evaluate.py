"""Evaluate tracker outputs against MOT17 ground truth (CLEAR-MOT + identity metrics, py-motmetrics).

MOT17 protocol, simplified like the official devkit:
  - GT boxes of class 1 (pedestrian) with the "consider" flag = 1 are the targets.
  - Tracker boxes that match (IoU >= 0.5) a GT box of a distractor class (2 person on vehicle, 7 static person,
    8 distractor, 12 reflection) or a non-considered box are removed before scoring (not counted as FP).
  - Matching threshold IoU >= 0.5.

Metrics: MOTA = 1 - (FN + FP + IDSW) / GT,  IDF1 = identity F1,  MT/ML = mostly tracked / lost,  ID switches, FPS.

Usage: python evaluate.py --mot ../../data/mot17/MOT17/train --tracks ../results/tracks --out ../results
"""
import argparse
import glob
import json
import os

import motmetrics as mm
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

DISTRACTORS = {2, 7, 8, 12}


def load(path, gt=False):
    cols = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"] if gt else ["frame", "id", "x", "y", "w", "h", "conf", "a", "b", "c"]
    d = pd.read_csv(path, header=None, names=cols)
    return d


def iou_matrix(a, b):
    """a, b: (N,4) x,y,w,h"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:4].prod(1)[:, None] + b[:, 2:4].prod(1)[None] - inter + 1e-9)


def evaluate_seq(gt_path, tr_path, name, min_conf=0.0):
    gt = load(gt_path, gt=True)
    tr = load(tr_path)
    tr = tr[tr.conf >= min_conf]
    acc = mm.MOTAccumulator(auto_id=False)
    for f in sorted(gt.frame.unique()):
        g = gt[gt.frame == f]
        t = tr[tr.frame == f]
        keep_g = g[(g.cls == 1) & (g.conf == 1)]
        ign_g = g[~((g.cls == 1) & (g.conf == 1)) & (g.cls.isin(DISTRACTORS | {1}))]
        tb = t[["x", "y", "w", "h"]].values
        # drop tracker boxes that cover ignored / distractor GT (matched one-to-one first to valid GT)
        if len(tb) and len(ign_g):
            gb_all = np.vstack([keep_g[["x", "y", "w", "h"]].values, ign_g[["x", "y", "w", "h"]].values])
            iou = iou_matrix(gb_all, tb)
            r, c = linear_sum_assignment(-iou)
            drop = [cc for rr, cc in zip(r, c) if rr >= len(keep_g) and iou[rr, cc] >= 0.5]
            t = t.drop(t.index[drop])
            tb = t[["x", "y", "w", "h"]].values
        # distance = 1 - IoU, pairs below IoU 0.5 cannot match (own helper: motmetrics' version needs NumPy < 2)
        iou = iou_matrix(keep_g[["x", "y", "w", "h"]].values.astype(float), tb.astype(float))
        dist = np.where(iou >= 0.5, 1 - iou, np.nan)
        acc.update(keep_g.id.values, t.id.values, dist, frameid=f)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot", default="../../data/mot17/MOT17/train")
    ap.add_argument("--tracks", default="../results/tracks")
    ap.add_argument("--out", default="../results")
    ap.add_argument("--sweep", action="store_true", help="also sweep the output confidence threshold")
    a = ap.parse_args()
    mh = mm.metrics.create()
    metrics = ["mota", "idf1", "precision", "recall", "num_switches", "mostly_tracked", "mostly_lost",
               "num_false_positives", "num_misses", "num_unique_objects"]
    rows, allsum = [], {}
    for cfg in sorted(os.listdir(a.tracks)):
        accs, names, fps = [], [], []
        for tr in sorted(glob.glob(os.path.join(a.tracks, cfg, "MOT17-*.txt"))):
            if tr.endswith("_fps.txt"):
                continue
            seq = os.path.basename(tr)[:-4]
            accs.append(evaluate_seq(os.path.join(a.mot, f"{seq}-FRCNN", "gt", "gt.txt"), tr, seq))
            names.append(seq)
            fps.append(float(open(tr.replace(".txt", "_fps.txt")).read()))
        if not accs:
            continue
        s = mh.compute_many(accs, metrics=metrics, names=names, generate_overall=True)
        s["fps"] = fps + [float(np.mean(fps))]
        allsum[cfg] = s.reset_index().to_dict(orient="records")
        for seqname, r in s.iterrows():
            rows.append(dict(config=cfg, sequence=seqname, **r.to_dict()))
        print(cfg)
        print(mm.io.render_summary(s, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "metrics.csv"), index=False)
    lines = ["| detector + tracker | sequence | MOTA % | IDF1 % | precision % | recall % | ID switches | MT | ML | FP | FN | GT IDs | FPS |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        b = "**" if r.sequence == "OVERALL" else ""
        lines.append(f"| {b}{r.config}{b} | {b}{r.sequence}{b} | {r.mota * 100:.1f} | {r.idf1 * 100:.1f} | {r.precision * 100:.1f} | "
                     f"{r.recall * 100:.1f} | {int(r.num_switches)} | {int(r.mostly_tracked)} | {int(r.mostly_lost)} | "
                     f"{int(r.num_false_positives)} | {int(r.num_misses)} | {int(r.num_unique_objects)} | {r.fps:.1f} |")
    open(os.path.join(a.out, "metrics.md"), "w").write("\n".join(lines) + "\n")
    json.dump(allsum, open(os.path.join(a.out, "metrics.json"), "w"), indent=2, default=float)

    if a.sweep:  # post-hoc: drop output boxes below a confidence threshold (trades FP against FN)
        lines = ["| detector + tracker | min box confidence | MOTA % | IDF1 % | precision % | recall % | ID switches |", "|---|---|---|---|---|---|---|"]
        for cfg in sorted(os.listdir(a.tracks)):
            trs = [t for t in sorted(glob.glob(os.path.join(a.tracks, cfg, "MOT17-*.txt"))) if not t.endswith("_fps.txt")]
            for mc in (0.0, 0.3, 0.4, 0.5, 0.6):
                accs = [evaluate_seq(os.path.join(a.mot, f"{os.path.basename(t)[:-4]}-FRCNN", "gt", "gt.txt"), t, "", mc) for t in trs]
                s = mh.compute_many(accs, metrics=metrics, names=[os.path.basename(t)[:-4] for t in trs], generate_overall=True).loc["OVERALL"]
                lines.append(f"| {cfg} | {mc:.1f} | {s.mota * 100:.1f} | {s.idf1 * 100:.1f} | {s.precision * 100:.1f} | {s.recall * 100:.1f} | {int(s.num_switches)} |")
                print(lines[-1], flush=True)
        open(os.path.join(a.out, "conf_sweep.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
