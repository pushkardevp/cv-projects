# Dataset: Driver Drowsiness Dataset (DDD)

- **Kaggle:** https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd
  (face regions extracted from the videos of the Real-Life Drowsiness Dataset, RLDD).
- **Content:** 41,793 RGB PNG face crops, 227×227.

| class | images | recording sessions (file prefixes) |
|---|---|---|
| Drowsy | 22,348 | 28 (A … Y, ZA, ZB, ZC) |
| Non Drowsy | 19,445 | 26 (a … y, za, zb, zc; no f, t) |

- **Structure:** the file number is the frame index within a session, so consecutive numbers are consecutive video frames.
  This is what lets us test the temporal (PERCLOS) logic.
- **Important:** prefixes are **different people** (e.g. `A` and `a` are not the same person). We therefore split by prefix
  (person-wise). A random image split would leak near-duplicate frames of the same person into the test set.
- **Labels are per session:** every frame of a "Drowsy" session is labelled drowsy, including frames with open eyes.
- **Download:** `bash ../download_data.sh ddd` (about 2.6 GB zip, needs a Kaggle API token), giving
  `../data/ddd/Driver Drowsiness Dataset (DDD)/{Drowsy,Non Drowsy}/`.
- **Model file:** `models/face_landmarker.task` from
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task (Apache 2.0).
