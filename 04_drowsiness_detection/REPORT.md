# Report: Driver Drowsiness Detection (Facial Landmarks, EAR, PERCLOS)

## 1. Problem
Detect driver drowsiness from a driver-facing camera and raise a timely alert, using interpretable rules instead of a trained classifier.

## 2. Data
DDD: 41,793 face crops from 54 recording sessions (28 drowsy, 26 alert), session-level labels. See [DATASET.md](DATASET.md).

## 3. Method
MediaPipe Face Landmarker gives 478 landmarks per face. From them we compute the Eye Aspect Ratio and Mouth Aspect Ratio and apply
threshold rules per frame, plus temporal rules (consecutive closed frames, PERCLOS). Thresholds are tuned for balanced accuracy on 40% of
the people and tested on the other 60%, over 5 random person-wise splits. The live system adds a 3-second per-driver calibration.

## 4. Results (unseen people)
| detector | accuracy | F1 | ROC-AUC |
|---|---|---|---|
| EAR threshold (frame) | 62.3 ± 2.7% | 60.5 ± 4.0% | 0.70 |
| MediaPipe blink score (frame) | 66.0 ± 1.8% | 64.5 ± 5.4% | 0.76 |
| PERCLOS, 30-frame windows | 66.4 ± 6.9% | 69.2 ± 11.5% | 0.75 |

Face detection rate 99.94%. MAR is uninformative on DDD (AUC 0.42).

## 5. Challenges
- **Variation between people:** median open-eye EAR ranges from 0.21 to 0.42. Some people's normal eyes look "closed" to a global
  threshold (the red "alert → drowsy" examples). This motivated the `--calibrate` option for the live demo.
- **Weak labels:** session-level labels mark open-eye frames as drowsy, so perfect frame accuracy is impossible by design.
- **Data leakage risk:** the same person appears in thousands of near-identical frames. Person-wise splitting was essential.
  A random split would have given optimistic numbers.
- **Library changes:** MediaPipe 1.x removed the old `mp.solutions.face_mesh` API, so we used the Tasks API (Face Landmarker
  with a downloaded `.task` model). An Ultralytics/MediaPipe dependency installed a headless OpenCV that broke `cv2.imshow` for
  the webcam demo, and it had to be removed.

## 6. Limitations
- Frontal, daytime, mostly well-lit faces. Night driving (IR cameras), sunglasses, strong head turns and glasses glare are not evaluated.
- The live webcam demo was not quantitatively evaluated (no labelled webcam data). Its thresholds depend on calibration.
- Head nodding and gaze, which are useful drowsiness cues, are not used.
- Thresholds tuned on DDD (227×227 crops) may not transfer to another camera and distance.

## 7. Conclusion
A landmark-based, training-free detector is fast, explainable and deployable on a CPU. Rigorous person-wise evaluation shows the
real difficulty: single-frame eye openness is a weak signal across different people (about 62%), while temporal aggregation (PERCLOS) and
learned blink scores do better (about 66%, F1 up to 69%). Per-driver calibration and temporal logic are the key design elements of the live system.
