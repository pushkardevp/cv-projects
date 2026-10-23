# Report: Road Lane Detection and Lane Departure Warning

## 1. Problem
Lane-keeping assistance needs to know, in every camera frame, where the lane lines are and how far the car is from
the lane centre. Lane paint can be white or yellow, solid or dashed, and appears under shadows and on asphalt of
different colour. The system must be stable from frame to frame and must keep working when the car actually crosses
a line, which is exactly when a warning matters.

## 2. Data
Udacity CarND-LaneLines-P1 sample data: 3 highway videos (1153 frames in total, 960×540 and 1280×720) and 6 still
images. No ground-truth labels. Details in [DATASET.md](DATASET.md).

## 3. Method
Classical, interpretable pipeline (no training): HLS colour mask for white/yellow paint → grayscale + Gaussian blur →
Canny → trapezoid ROI → probabilistic Hough transform → slope-based left/right split and length-weighted line fit →
exponential smoothing over time → perspective warp to a bird's-eye view, where lane width and the car's lateral
offset are measured → warning if |offset| > 20% of lane width (0.74 m). Figure: `samples/pipeline_stages.png`.

Preprocessing choices and why:
- **HLS instead of RGB:** lightness and hue separate paint from asphalt better than RGB under changing light.
- **Gaussian blur before Canny:** suppresses asphalt texture, which would otherwise produce many short edges.
- **ROI:** removes edges from the sky, trees, other cars and the hood.
- **Bird's-eye warp:** in the camera view, lane width shrinks with distance. After the warp it is constant, so the
  offset can be measured in metres.

## 4. Evaluation and results
**(a) Real videos, label-free metrics:** both lines detected in 100% / 100% / 98.4% of frames. Jitter is
0.7 / 0.7 / 1.7 px per frame. The bird's-eye lane width varies by 0.6–2.0% (CV). There are 0 warnings, which is correct,
since the car stays in lane.

**(b) Ablations on `challenge.mp4`:** without the colour mask, jitter is 2.4× higher, lane-width variation is 3.7× higher, and
5 false warnings appear. Without smoothing, jitter is 4.4× higher.

**(c) Departure warning on simulated drift:** each of 53 frames is warped with a road-plane homography to simulate
25 lateral shifts between −1.2 m and +1.2 m (1325 tests). Offset error is 0.018 m (mean) and 0.035 m (90th percentile).
Warning accuracy is 98.2%, precision 98.7%, recall 96.9%.

## 5. Challenges and how they were handled
- **The first version failed during departures.** Lines were assigned to left/right by screen half, so when the car drifted,
  the line it was crossing moved to the "wrong" half and was discarded. The drift test showed warning recall of only 8%.
  Assigning lines by slope sign, and keeping a remembered lane width to place a missing line, raised recall to 96.9%.
- **Shadows and the bright bridge in `challenge.mp4`** create strong non-paint edges. The colour mask removes most of them.
- **Dashed lines** disappear for a few frames. The tracker keeps the last estimate for up to 5 frames.
- **OpenCV API change:** `HoughLinesP` output shape differs between versions. The code reshapes it to (N, 4).

## 6. Limitations
- **Straight-line model:** curves are approximated by straight lines near the car. A polynomial fit on the bird's-eye view
  (sliding windows) would handle sharp curves better.
- **No ground-truth labels:** metrics (a) and (b) measure stability, not absolute accuracy. In (c), the "true" offset
  is the pipeline's own estimate on the unshifted frame plus the known shift. The test therefore checks the response to drift,
  not the absolute calibration.
- **Fixed geometry:** the ROI and warp trapezoid assume a fixed camera mounting and a flat road. The metre conversion assumes a 3.7 m lane.
- **Simulated departures:** the homography is exact only for the flat road plane. Cars and barriers are distorted. Real
  departure footage was not available.
- **Conditions:** only daytime highway with clear markings. Night, rain, worn paint and dense traffic are untested and
  would likely break the colour thresholds.

## 7. Conclusion
A classical pipeline is fast, needs no training data and is easy to explain. On the Udacity videos it is reliable and
stable, and the simulated-drift test gives 98% warning accuracy. Evaluating the departure case explicitly was the
most valuable step: it exposed a design flaw that the plain videos could not reveal.
