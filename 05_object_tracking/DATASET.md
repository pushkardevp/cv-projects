# Dataset: MOT17 (MOTChallenge)

- **Source:** https://motchallenge.net/data/MOT17/ (Milan et al., "MOT16: A Benchmark for Multi-Object Tracking", arXiv:1603.00831).
  Licence: CC BY-NC-SA 3.0, non-commercial use with attribution. Sample frames and videos in this repository are derived from it.
- **Content:** 7 training + 7 test sequences of pedestrians (static and moving cameras). Ground truth is public only for the
  training sequences. Each sequence comes with 3 public detection sets (DPM, FRCNN, SDP). The images and GT are identical, so we use
  the `-FRCNN` copy for images and GT and ignore the public detections (we run our own detector).
- **Used here:**

| sequence | camera | frames | resolution | fps | GT identities |
|---|---|---|---|---|---|
| MOT17-02 | static, street level | 600 | 1920×1080 | 30 | 62 |
| MOT17-04 | static, elevated, night street | 1050 | 1920×1080 | 30 | 83 |
| MOT17-11 | moving (hand-held), mall | 900 | 1920×1080 | 30 | 75 |

- **GT format** (`gt/gt.txt`): `frame, id, x, y, w, h, consider_flag, class, visibility`. Class 1 = pedestrian. Classes 2, 7, 8, 12
  (person on vehicle, static person, distractor, reflection) are ignored during evaluation.
- **Download:** `bash ../download_data.sh mot17` (MOT17.zip, about 5.5 GB; only `train/` is extracted, about 2.5 GB).
