# Report: Multi-Object Tracking with YOLOv8 + ByteTrack / BoT-SORT

## 1. Problem
Track every pedestrian in a video with a consistent identity, through occlusions, crowds and camera motion.

## 2. Data
MOT17 training sequences 02, 04 and 11 (2,550 frames, 220 identities, static and moving cameras). See [DATASET.md](DATASET.md).

## 3. Method
Tracking-by-detection. A COCO-pretrained YOLOv8 (person class, 1280 px, conf ≥ 0.1) detects people in each frame. Kalman-filter
motion prediction, IoU cost and Hungarian matching associate the detections over time. ByteTrack adds a second association round
with low-confidence detections. BoT-SORT adds camera-motion compensation. No model was trained. Preprocessing is the
detector's letterbox resize. Post-processing includes an output-confidence threshold sweep.

## 4. Results (overall, 3 sequences)
| configuration | MOTA | IDF1 | ID sw. | FPS |
|---|---|---|---|---|
| YOLOv8n + ByteTrack | 43.8% | 50.6% | 300 | 40 |
| YOLOv8s + ByteTrack | 40.9% | 51.9% | 299 | 28 |
| YOLOv8s + BoT-SORT | 40.5% | 52.7% | 271 | 21 |
| YOLOv8s + ByteTrack, conf ≥ 0.4 | 42.8% | 52.8% | 282 | 28 |

False negatives dominate (31–35k FN vs 7.5–13.5k FP over 75,574 GT boxes). MOT17-02 (small, occluded people) is hardest.

## 5. Challenges
- **Occlusion in crowds:** people disappear behind others. ByteTrack's low-score matching and the 30-frame track buffer bridge
  short gaps, but long occlusions still cause ID switches.
- **Small, distant pedestrians:** we used a higher input resolution (1280 instead of 640) to detect them. Recall on MOT17-02 is still below 40%.
- **Moving camera (MOT17-11):** camera-motion compensation (BoT-SORT) did not improve MOTA here (more FPs), with similar IDF1.
- **Evaluation details:** the MOT17 protocol ignores distractor classes. Without that filter, correct detections of static
  or seated people would count as false positives. The `motmetrics` IoU helper is incompatible with NumPy 2, so we implemented IoU ourselves.
- **Ultralytics API:** passing a list of frames runs them as one batch (GPU out of memory). Frames must be fed one by one with `persist=True`.

## 6. Limitations
- **Simplified protocol:** close to, but not identical to, the official TrackEval/HOTA scoring. Numbers are comparable between our
  configurations but not directly to the leaderboard.
- **Limited scope:** only 3 sequences and one run each. The detector was never fine-tuned on MOT17 or pedestrian data.
- **Speed figures:** FPS was measured on one laptop with other processes running.
- **No appearance re-ID:** BoT-SORT's ReID is disabled by default, so long-term re-identification after leaving the frame is not handled.

## 7. Conclusion
Off-the-shelf components give a working multi-object tracker at about 40% MOTA and 50% IDF1 in real time without any training.
Error analysis shows detection recall is the bottleneck. The most effective improvement would be fine-tuning the detector on
pedestrian data. Tuning the confidence threshold per detector gives a free 2-point MOTA gain for YOLOv8s.
