"""Driver drowsiness detection from facial landmarks (MediaPipe Face Landmarker, 478 points).

Per frame:
  EAR (eye aspect ratio)   = (|p2-p6| + |p3-p5|) / (2 |p1-p4|), averaged over both eyes.
                             Open eye ~0.25-0.35, closed eye < ~0.15 (Soukupova & Cech, 2016).
  MAR (mouth aspect ratio) = mean vertical lip opening / mouth width. Yawning -> large MAR.
Over time (DrowsinessMonitor):
  - eyes closed for >= CLOSED_FRAMES consecutive frames      -> ALERT "eyes closed"
  - PERCLOS (% of the last WINDOW frames with eyes closed) >= PERCLOS_ALERT -> ALERT "drowsy"
  - mouth open (MAR > threshold) for >= YAWN_FRAMES frames  -> yawn counted

Run live (external webcam = index 1, laptop camera usually 0):
  python drowsiness.py --source 1 --calibrate
Run on a video file:
  python drowsiness.py --source video.mp4 --output out.mp4
"""
import argparse
import collections
import json
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "models", "face_landmarker.task")
THRESH_FILE = os.path.join(HERE, "..", "results", "thresholds.json")

# landmark indices (MediaPipe face mesh topology)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]   # p1..p6: corner, top, top, corner, bottom, bottom
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = dict(left=61, right=291, pairs=[(81, 178), (13, 14), (311, 402)])

# defaults; evaluate.py writes tuned values to results/thresholds.json
EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.60
CLOSED_FRAMES = 15        # ~0.5 s at 30 fps
WINDOW = 90               # ~3 s at 30 fps
PERCLOS_ALERT = 0.40
YAWN_FRAMES = 20


def make_landmarker(video=False):
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL), num_faces=1, output_face_blendshapes=True,
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE)
    return vision.FaceLandmarker.create_from_options(opts)


def _pts(lms, idx, w, h):
    return np.array([[lms[i].x * w, lms[i].y * h] for i in idx])


def eye_aspect_ratio(p):
    return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * np.linalg.norm(p[0] - p[3]) + 1e-6)


def features(result, w, h):
    """Return dict(ear, mar, blink, landmarks) or None if no face."""
    if not result.face_landmarks:
        return None
    lms = result.face_landmarks[0]
    ear = (eye_aspect_ratio(_pts(lms, RIGHT_EYE, w, h)) + eye_aspect_ratio(_pts(lms, LEFT_EYE, w, h))) / 2
    width = np.linalg.norm(_pts(lms, [MOUTH["left"]], w, h) - _pts(lms, [MOUTH["right"]], w, h))
    opening = np.mean([np.linalg.norm(_pts(lms, [a], w, h) - _pts(lms, [b], w, h)) for a, b in MOUTH["pairs"]])
    blink = None
    if result.face_blendshapes:
        bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
        blink = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2
    return dict(ear=float(ear), mar=float(opening / (width + 1e-6)), blink=blink, landmarks=lms)


class DrowsinessMonitor:
    def __init__(self, ear_t=EAR_THRESHOLD, mar_t=MAR_THRESHOLD):
        self.ear_t, self.mar_t = ear_t, mar_t
        self.closed_run = 0
        self.yawn_run = 0
        self.yawns = 0
        self.hist = collections.deque(maxlen=WINDOW)

    def update(self, f):
        if f is None:  # no face: count as not-closed but do not reset alerts abruptly
            self.hist.append(0)
            return dict(state="NO FACE", perclos=float(np.mean(self.hist)), alert=False)
        closed = f["ear"] < self.ear_t
        self.closed_run = self.closed_run + 1 if closed else 0
        self.hist.append(int(closed))
        if f["mar"] > self.mar_t:
            self.yawn_run += 1
            if self.yawn_run == YAWN_FRAMES:
                self.yawns += 1
        else:
            self.yawn_run = 0
        perclos = float(np.mean(self.hist))
        alert, state = False, "AWAKE"
        if self.closed_run >= CLOSED_FRAMES:
            alert, state = True, "EYES CLOSED!"
        elif len(self.hist) >= WINDOW // 2 and perclos >= PERCLOS_ALERT:
            alert, state = True, "DROWSY"
        elif self.yawn_run >= YAWN_FRAMES:
            state = "YAWNING"
        return dict(state=state, perclos=perclos, alert=alert, closed=closed)


def draw(frame, f, st):
    h, w = frame.shape[:2]
    if f is not None:
        for idx in (RIGHT_EYE, LEFT_EYE, [MOUTH["left"], MOUTH["right"]] + [i for p in MOUTH["pairs"] for i in p]):
            for x, y in _pts(f["landmarks"], idx, w, h).astype(int):
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        cv2.putText(frame, f"EAR {f['ear']:.3f}  MAR {f['mar']:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"PERCLOS {st['perclos'] * 100:.0f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    col = (0, 0, 255) if st["alert"] else (0, 200, 0)
    cv2.putText(frame, st["state"], (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
    if st["alert"]:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
    return frame


def load_thresholds():
    if os.path.exists(THRESH_FILE):
        t = json.load(open(THRESH_FILE))
        return t["ear_threshold"], t["mar_threshold"]
    return EAR_THRESHOLD, MAR_THRESHOLD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index (0 laptop, 1 external) or video path")
    ap.add_argument("--output", help="optional output video")
    ap.add_argument("--ear", type=float, help="override EAR threshold")
    ap.add_argument("--calibrate", action="store_true",
                    help="first 3 s: keep eyes open and look at the camera; threshold = 0.75 x your open-eye EAR")
    a = ap.parse_args()
    ear_t, mar_t = load_thresholds()
    if a.ear:
        ear_t = a.ear
    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {a.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    lm = make_landmarker(video=True)
    mon = DrowsinessMonitor(ear_t, mar_t)
    wr = None
    t_ms = 0
    calib, n_frames = [], 0
    print(f"EAR threshold {ear_t:.3f}, MAR threshold {mar_t:.2f}. Press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        t_ms += int(1000 / fps)
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), t_ms)
        f = features(res, w, h)
        n_frames += 1
        if a.calibrate and n_frames <= 3 * fps:
            if f is not None:
                calib.append(f["ear"])
            vis = frame
            cv2.putText(vis, "CALIBRATING: eyes open, look at camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if n_frames >= 3 * fps and calib:
                mon.ear_t = 0.75 * float(np.median(calib))
                print(f"calibrated: open-eye EAR {np.median(calib):.3f} -> threshold {mon.ear_t:.3f}")
        else:
            st = mon.update(f)
            vis = draw(frame, f, st)
        if a.output:
            if wr is None:
                wr = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            wr.write(vis)
        if isinstance(src, int):
            cv2.imshow("drowsiness", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    if wr:
        wr.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
