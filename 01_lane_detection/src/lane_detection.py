"""Classical lane detection + lane departure warning.

Pipeline per frame:
  1. Colour mask (HLS): keep white and yellow paint, drop grey asphalt/shadows.
  2. Grayscale + Gaussian blur + Canny edges, restricted to the colour mask.
  3. Trapezoid region of interest (road area in front of the car).
  4. Probabilistic Hough transform -> line segments.
  5. Split segments into left/right by slope, fit one line per side (length-weighted).
  6. Temporal smoothing (exponential moving average) across frames.
  7. Perspective warp to a bird's-eye view: measure lane width and the car's
     lateral offset from the lane centre -> departure warning.

Usage:
  python lane_detection.py --input video.mp4 --output out.mp4 [--no-color] [--no-smooth]
  python lane_detection.py --input image.jpg --output out.jpg
"""
import argparse
import json
import os

import cv2
import numpy as np

# ---- tunable parameters (try changing these) ----
CANNY_LOW, CANNY_HIGH = 50, 150
HOUGH_THRESHOLD = 20        # min votes
HOUGH_MIN_LEN = 20          # px
HOUGH_MAX_GAP = 100         # px
MIN_ABS_SLOPE = 0.4         # reject near-horizontal segments
SMOOTH_ALPHA = 0.2          # EMA weight of the new frame (lower = smoother)
DEPARTURE_THRESHOLD = 0.20  # warn when |offset| > 20% of the lane width
LANE_WIDTH_M = 3.7          # standard US lane width, for metres conversion
MAX_MISSING_FRAMES = 5      # drop a lane line after this many frames without a detection
DEFAULT_LANE_W_FRAC = 0.40  # bird's-eye lane width / image width, used before any two-line fit

# ROI and warp source trapezoid as fractions of (width, height)
ROI = [(0.10, 0.95), (0.45, 0.62), (0.55, 0.62), (0.95, 0.95)]
Y_TOP, Y_BOTTOM = 0.64, 0.95  # rows where the fitted lines are drawn / warped


def color_mask(frame):
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(hls, (0, 190, 0), (180, 255, 255))
    yellow = cv2.inRange(hls, (15, 80, 90), (40, 220, 255))
    return cv2.bitwise_or(white, yellow)


def edges(frame, use_color=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    e = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    if use_color:
        mask = cv2.dilate(color_mask(frame), np.ones((5, 5), np.uint8))
        e = cv2.bitwise_and(e, mask)
    return e


def roi_mask(img):
    h, w = img.shape[:2]
    poly = np.array([[(int(x * w), int(y * h)) for x, y in ROI]], np.int32)
    m = np.zeros_like(img)
    cv2.fillPoly(m, poly, 255)
    return cv2.bitwise_and(img, m)


def fit_lanes(segments, w, h):
    """Return (left, right) lines as (slope, intercept) or None."""
    sides = {"left": [], "right": []}
    if segments is not None:
        for x1, y1, x2, y2 in segments.reshape(-1, 4):
            if x1 == x2:
                continue
            m = (y2 - y1) / (x2 - x1)
            if abs(m) < MIN_ABS_SLOPE:
                continue
            b = y1 - m * x1
            length = np.hypot(x2 - x1, y2 - y1)
            # image y points down: left lane line has negative slope, right lane line positive.
            # Only the slope sign is used (not the screen half), so a line that drifts past the
            # image centre during a lane departure is still assigned to the correct side.
            sides["left" if m < 0 else "right"].append((m, b, length))
    out = []
    for k in ("left", "right"):
        if not sides[k]:
            out.append(None)
            continue
        a = np.array(sides[k])
        out.append((np.average(a[:, 0], weights=a[:, 2]), np.average(a[:, 1], weights=a[:, 2])))
    return out


def x_at(line, y):
    m, b = line
    return (y - b) / m


class LaneTracker:
    def __init__(self, smooth=True):
        self.smooth = smooth
        self.state = [None, None]
        self.missing = [0, 0]
        self.lane_w = None  # bird's-eye lane width (px), remembered for the one-line fallback

    def update(self, lines):
        for i, ln in enumerate(lines):
            if ln is None:
                # keep the last estimate for a few frames (dashed lines have gaps), then drop it
                self.missing[i] += 1
                if self.missing[i] > MAX_MISSING_FRAMES or not self.smooth:
                    self.state[i] = None
                continue
            self.missing[i] = 0
            if self.state[i] is None or not self.smooth:
                self.state[i] = ln
            else:
                old = np.array(self.state[i])
                self.state[i] = tuple((1 - SMOOTH_ALPHA) * old + SMOOTH_ALPHA * np.array(ln))
        return self.state


def birdseye_matrix(w, h):
    src = np.float32([(x * w, y * h) for x, y in ROI])
    dst = np.float32([(0.25 * w, h), (0.25 * w, 0), (0.75 * w, 0), (0.75 * w, h)])
    return cv2.getPerspectiveTransform(src, dst)


def to_birdseye_x(x, y, M):
    return cv2.perspectiveTransform(np.float32([[[x, y]]]), M)[0, 0, 0]


def lane_geometry(left, right, w, h, M, known_lane_w=None):
    """Lane width and car offset measured in the bird's-eye view (pixels -> fraction of lane width).
    If only one line is visible, the other is placed one remembered lane width away."""
    yb = Y_BOTTOM * h
    car = to_birdseye_x(w / 2, yb, M)
    xl = to_birdseye_x(x_at(left, yb), yb, M) if left is not None else None
    xr = to_birdseye_x(x_at(right, yb), yb, M) if right is not None else None
    if xl is not None and xr is not None:
        lane_w = xr - xl
    else:
        lane_w = known_lane_w or DEFAULT_LANE_W_FRAC * w
        xl = xl if xl is not None else xr - lane_w
        xr = xr if xr is not None else xl + lane_w
    wp = (xl, xr, car)
    centre = (wp[0] + wp[1]) / 2
    offset_frac = (wp[2] - centre) / lane_w if lane_w > 1 else 0.0  # + = car right of centre
    return lane_w, offset_frac


def draw(frame, left, right, offset_frac, warn, birdseye):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    yt, yb = Y_TOP * h, Y_BOTTOM * h
    if left is not None and right is not None:
        poly = np.array([[x_at(left, yb), yb], [x_at(left, yt), yt],
                         [x_at(right, yt), yt], [x_at(right, yb), yb]], np.int32)
        cv2.fillPoly(overlay, [poly], (0, 0, 255) if warn else (0, 180, 0))
    for ln in (left, right):
        if ln is not None:
            cv2.line(overlay, (int(x_at(ln, yb)), int(yb)), (int(x_at(ln, yt)), int(yt)), (255, 0, 0), 8)
    out = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    if offset_frac is not None:
        txt = f"offset {offset_frac * LANE_WIDTH_M:+.2f} m ({offset_frac * 100:+.0f}% of lane)"
        cv2.putText(out, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    if warn:
        cv2.putText(out, "LANE DEPARTURE WARNING", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    # picture-in-picture: bird's-eye edge view
    if birdseye is not None:
        pip = cv2.resize(cv2.cvtColor(birdseye, cv2.COLOR_GRAY2BGR), (w // 4, h // 4))
        out[10:10 + h // 4, w - w // 4 - 10:w - 10] = pip
        cv2.putText(out, "bird's-eye", (w - w // 4, h // 4 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out


def process_frame(frame, tracker, M, use_color=True):
    h, w = frame.shape[:2]
    e = roi_mask(edges(frame, use_color))
    segs = cv2.HoughLinesP(e, 1, np.pi / 180, HOUGH_THRESHOLD, minLineLength=HOUGH_MIN_LEN, maxLineGap=HOUGH_MAX_GAP)
    raw = fit_lanes(segs, w, h)
    left, right = tracker.update(raw)
    offset, lane_w, warn = None, None, False
    if left is not None or right is not None:
        lane_w, offset = lane_geometry(left, right, w, h, M, tracker.lane_w)
        if left is not None and right is not None and lane_w > 0:
            tracker.lane_w = lane_w if tracker.lane_w is None else 0.9 * tracker.lane_w + 0.1 * lane_w
        else:
            lane_w = None  # not measured this frame
        warn = abs(offset) > DEPARTURE_THRESHOLD
    bird = cv2.warpPerspective(e, M, (w, h))
    vis = draw(frame, left, right, offset, warn, bird)
    return vis, dict(left_found=raw[0] is not None, right_found=raw[1] is not None,
                     left=left, right=right, offset=offset, lane_w=lane_w, warn=warn, edges=e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--stats", help="write per-frame stats JSON here")
    ap.add_argument("--no-color", action="store_true", help="ablation: Canny only, no colour mask")
    ap.add_argument("--no-smooth", action="store_true", help="ablation: no temporal smoothing")
    a = ap.parse_args()
    tracker = LaneTracker(smooth=not a.no_smooth)

    if a.input.lower().endswith((".jpg", ".png")):
        img = cv2.imread(a.input)
        vis, info = process_frame(img, tracker, birdseye_matrix(img.shape[1], img.shape[0]), not a.no_color)
        cv2.imwrite(a.output, vis)
        print(f"left={info['left_found']} right={info['right_found']} offset={info['offset']}")
        return

    cap = cv2.VideoCapture(a.input)
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
    M = birdseye_matrix(w, h)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    writer = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rows = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis, info = process_frame(frame, tracker, M, not a.no_color)
        writer.write(vis)
        rows.append({k: (list(map(float, v)) if isinstance(v, tuple) else v)
                     for k, v in info.items() if k != "edges"})
    writer.release()
    if a.stats:
        with open(a.stats, "w") as f:
            json.dump(dict(fps=fps, width=w, height=h, frames=rows), f, default=float)
    print(f"{len(rows)} frames -> {a.output}")


if __name__ == "__main__":
    main()
