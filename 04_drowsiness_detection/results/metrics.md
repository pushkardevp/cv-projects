Face found in 99.94% of 41793 images. Person-wise splits, 5 seeds, mean ± std (%).

| detector | accuracy | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| EAR < t (frame) | 62.3 ± 2.7 | 69.4 ± 7.2 | 54.7 ± 7.8 | 60.5 ± 4.0 | 69.7 ± 3.3 |
| MediaPipe blink score > t (frame) | 66.0 ± 1.8 | 73.8 ± 7.3 | 59.7 ± 13.0 | 64.5 ± 5.4 | 75.9 ± 2.7 |
| EAR < t or MAR > t (frame) | 62.4 ± 2.5 | 69.3 ± 7.0 | 55.0 ± 8.2 | 60.6 ± 4.2 | – |
| EAR / person's p90 EAR < t (frame) | 61.8 ± 0.7 | 76.8 ± 8.8 | 42.5 ± 5.9 | 54.0 ± 3.3 | 65.3 ± 2.4 |
| PERCLOS over 30 frames (temporal) | 66.4 ± 6.9 | 66.3 ± 5.4 | 76.6 ± 21.2 | 69.2 ± 11.5 | 74.7 ± 5.7 |
