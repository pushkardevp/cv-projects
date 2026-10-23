# Dataset: Pothole Detection (Kaggle)

- **Kaggle:** https://www.kaggle.com/datasets/andrewmvd/pothole-detection (by Larxel / andrewmvd)
- **Content:** 665 PNG road images, mixed sources and resolutions (most 400×400, others 400–533 × 300), with
  **1,739 pothole bounding boxes** in Pascal VOC XML (`annotations/*.xml`). One class: `pothole`.
- **Conversion:** `src/prepare_data.py` → YOLO format (`labels/*.txt`: `0 cx cy w h`, normalised) and `data.yaml`.
- **Split (seed 42, by image):**

| split | images | boxes |
|---|---|---|
| train | 465 | 1,212 |
| val | 100 | 251 |
| test | 100 | 276 |

- **Download + convert:** `bash ../download_data.sh pothole` (needs a Kaggle API token), giving `../data/pothole_yolo/`.
- **Caveats:** images come from different cameras and viewpoints (close-ups and street scenes). Small, distant potholes are
  not always annotated.
