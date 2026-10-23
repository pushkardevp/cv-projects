"""Evaluate rule-based drowsiness detectors on DDD with PERSON-WISE splits.

DDD file prefixes (A, B, ..., ZC drowsy; a, b, ..., zc non-drowsy) are recording sessions of different people,
so we split by prefix: thresholds are tuned on 40% of the people and tested on the other 60%; repeated over
5 random splits (mean +- std). No image of a test person is ever used for tuning.

Detectors (each is ONE threshold, tuned for balanced accuracy on the calibration people):
  ear          : drowsy if EAR < t                          (classic geometric rule)
  blink        : drowsy if MediaPipe eyeBlink score > t     (learned blendshape, same landmarker)
  ear_or_mar   : drowsy if EAR < t1 or MAR > t2             (eyes + yawning)
  ear_relative : drowsy if EAR / (person's 90th-percentile EAR) < t  (per-person calibration, label-free)
  perclos_30   : temporal: windows of 30 consecutive frames, drowsy if % frames with EAR < t_ear >= t_p

Usage: python evaluate.py --features ../results/features.csv --data "../../data/ddd/Driver Drowsiness Dataset (DDD)"
"""
import argparse
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drowsiness as dz

WIN = 30


def best_threshold(score, y):
    """score: higher = more drowsy. Return threshold maximising balanced accuracy."""
    qs = np.unique(np.quantile(score, np.linspace(0.01, 0.99, 199)))
    best, bt = -1, None
    for t in qs:
        p = score >= t
        ba = (p[y == 1].mean() + (~p[y == 0]).mean()) / 2
        if ba > best:
            best, bt = ba, t
    return bt


def metrics(y, pred, score=None):
    m = dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
             recall=recall_score(y, pred), f1=f1_score(y, pred))
    if score is not None:
        m["auc"] = roc_auc_score(y, score)
    return m


def add_scores(d):
    d = d.copy()
    d["s_ear"] = -d.ear
    d["s_blink"] = d.blink
    d["s_mar"] = d.mar
    d["s_rel"] = -d.ear / d.groupby("group").ear.transform(lambda e: e.quantile(0.9))
    return d


def windows(d, ear_t):
    rows = []
    for g, gd in d.sort_values("frame").groupby("group"):
        closed = (gd.ear.values < ear_t).astype(float)
        for i in range(0, len(gd) - WIN + 1, WIN):
            rows.append(dict(group=g, drowsy=gd.drowsy.iloc[0], perclos=closed[i:i + WIN].mean()))
    return pd.DataFrame(rows)


def run_split(d, cal_groups):
    cal, te = d[d.group.isin(cal_groups)], d[~d.group.isin(cal_groups)]
    out, thr = {}, {}
    for name, col in (("ear", "s_ear"), ("blink", "s_blink"), ("ear_relative", "s_rel")):
        t = best_threshold(cal[col].values, cal.drowsy.values)
        thr[name] = t
        out[name] = metrics(te.drowsy, te[col] >= t, te[col])
    # EAR OR MAR: tune MAR threshold with the EAR threshold fixed
    te_ = thr["ear"]
    best, bm = -1, None
    for tm in np.quantile(cal.mar, [0.9, 0.95, 0.98, 0.99, 0.995, 1.0]):
        p = (cal.s_ear >= te_) | (cal.mar > tm)
        ba = (p[cal.drowsy == 1].mean() + (~p[cal.drowsy == 0]).mean()) / 2
        if ba > best:
            best, bm = ba, tm
    thr["mar"] = bm
    out["ear_or_mar"] = metrics(te.drowsy, (te.s_ear >= te_) | (te.mar > bm))
    # temporal PERCLOS over 30-frame windows
    wc, wt = windows(cal, -te_), windows(te, -te_)
    tp = best_threshold(wc.perclos.values, wc.drowsy.values)
    thr["perclos"] = tp
    out["perclos_30"] = metrics(wt.drowsy, wt.perclos >= tp, wt.perclos)
    return out, thr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="../results/features.csv")
    ap.add_argument("--data", default="../../data/ddd/Driver Drowsiness Dataset (DDD)")
    ap.add_argument("--results", default="../results")
    ap.add_argument("--samples", default="../samples")
    a = ap.parse_args()
    raw = pd.read_csv(a.features)
    face_rate = raw.face_found.mean()
    d = add_scores(raw[raw.face_found == 1])
    groups = {k: sorted(d[d.drowsy == k].group.unique()) for k in (0, 1)}

    allres, allthr = [], []
    for seed in range(5):
        rng = np.random.RandomState(seed)
        cal = [g for k in (0, 1) for g in rng.choice(groups[k], int(0.4 * len(groups[k])), replace=False)]
        res, thr, te = run_split(d, cal)
        allres.append(res)
        allthr.append(thr)
        if seed == 0:
            te0, thr0 = te, thr

    names = ["ear", "blink", "ear_or_mar", "ear_relative", "perclos_30"]
    desc = {"ear": "EAR < t (frame)", "blink": "MediaPipe blink score > t (frame)", "ear_or_mar": "EAR < t or MAR > t (frame)",
            "ear_relative": "EAR / person's p90 EAR < t (frame)", "perclos_30": "PERCLOS over 30 frames (temporal)"}
    summary = {}
    lines = ["| detector | accuracy | precision | recall | F1 | ROC-AUC |", "|---|---|---|---|---|---|"]
    for n in names:
        s = {}
        for k in ("accuracy", "precision", "recall", "f1", "auc"):
            v = [r[n][k] for r in allres if k in r[n]]
            if v:
                s[k] = (float(np.mean(v)), float(np.std(v)))
        summary[n] = s
        fmt = lambda k: f"{s[k][0] * 100:.1f} ± {s[k][1] * 100:.1f}" if k in s else "–"
        lines.append(f"| {desc[n]} | {fmt('accuracy')} | {fmt('precision')} | {fmt('recall')} | {fmt('f1')} | {fmt('auc')} |")
    table = "\n".join(lines)
    print(table)
    open(os.path.join(a.results, "metrics.md"), "w").write(
        f"Face found in {face_rate * 100:.2f}% of {len(raw)} images. Person-wise splits, 5 seeds, mean ± std (%).\n\n" + table + "\n")
    thr_mean = {k: float(np.mean([t[k] for t in allthr])) for k in allthr[0]}
    json.dump(dict(face_found_rate=face_rate, n_images=len(raw), n_people_drowsy=len(groups[1]),
                   n_people_alert=len(groups[0]), summary=summary, thresholds_mean=thr_mean,
                   note="ear threshold stored as -EAR score; see thresholds.json for the deployable values"),
              open(os.path.join(a.results, "metrics.json"), "w"), indent=2)
    json.dump(dict(ear_threshold=-thr_mean["ear"], mar_threshold=thr_mean["mar"], perclos_threshold=thr_mean["perclos"],
                   source="mean of 5 person-wise calibration splits on DDD"),
              open(os.path.join(a.results, "thresholds.json"), "w"), indent=2)

    # plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for k, lab in ((0, "non-drowsy"), (1, "drowsy")):
        ax[0].hist(d[d.drowsy == k].ear, bins=80, alpha=.6, density=True, label=lab)
    ax[0].axvline(-thr_mean["ear"], color="k", ls="--", label=f"threshold {-thr_mean['ear']:.3f}")
    ax[0].set_xlabel("EAR")
    ax[0].set_title("EAR distribution (all images)")
    ax[0].legend()
    for n, col in (("EAR", "s_ear"), ("blink score", "s_blink"), ("relative EAR", "s_rel"), ("MAR", "s_mar")):
        fpr, tpr, _ = roc_curve(te0.drowsy, te0[col])
        ax[1].plot(fpr, tpr, label=f"{n} (AUC {roc_auc_score(te0.drowsy, te0[col]):.2f})")
    ax[1].plot([0, 1], [0, 1], "k:")
    ax[1].set_xlabel("false positive rate")
    ax[1].set_ylabel("true positive rate")
    ax[1].set_title("ROC, test people (split 0)")
    ax[1].legend(fontsize=8)
    med = d.groupby(["group", "drowsy"]).ear.median().reset_index().sort_values("ear")
    ax[2].bar(range(len(med)), med.ear, color=["tab:orange" if v else "tab:blue" for v in med.drowsy])
    ax[2].set_title("median EAR per person (orange = drowsy set)")
    ax[2].set_xlabel("person (sorted)")
    ax[2].set_ylabel("median EAR")
    plt.tight_layout()
    plt.savefig(os.path.join(a.results, "ear_analysis.png"), dpi=90)
    plt.close()

    # annotated sample images: correct and wrong frame-level EAR decisions on test people
    lm = dz.make_landmarker()
    t_ear = -thr_mean["ear"]
    picks = []
    rng = np.random.RandomState(1)
    for k in (1, 0):
        for correct in (True, False):
            sub = te0[(te0.drowsy == k) & (((te0.ear < t_ear) == bool(k)) == correct)]
            picks += [(r, correct) for _, r in sub.iloc[rng.choice(len(sub), 3, replace=False)].iterrows()]
    fig, ax = plt.subplots(2, 6, figsize=(16, 6))
    for x, (r, correct) in zip(ax.flat, picks):
        img = cv2.imread(os.path.join(a.data, "Drowsy" if r.drowsy else "Non Drowsy", r.file))
        h, w = img.shape[:2]
        f = dz.features(lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))), w, h)
        if f:
            for idx in (dz.RIGHT_EYE, dz.LEFT_EYE):
                pts = dz._pts(f["landmarks"], idx, w, h).astype(np.int32)
                cv2.polylines(img, [pts], True, (0, 255, 255), 1)
        x.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pred = "drowsy" if r.ear < t_ear else "alert"
        x.set_title(f"label {'drowsy' if r.drowsy else 'alert'} | pred {pred}\nEAR {r.ear:.3f}", fontsize=9,
                    color="green" if correct else "red")
        x.axis("off")
    plt.suptitle(f"EAR rule on unseen people (threshold {t_ear:.3f}); green = correct, red = wrong")
    plt.tight_layout()
    plt.savefig(os.path.join(a.samples, "ear_predictions.png"), dpi=90)
    plt.close()


if __name__ == "__main__":
    main()
