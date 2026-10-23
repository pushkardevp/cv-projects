# Dataset: Udacity lane-line sample videos and images

- **Source:** Udacity Self-Driving Car Engineer Nanodegree, project 1 "Finding Lane Lines"
  https://github.com/udacity/CarND-LaneLines-P1 (folders `test_videos/` and `test_images/`). MIT licence.
- **Content:** dash-camera footage from a car on Californian highways, daytime.

| file | type | resolution | frames | notes |
|---|---|---|---|---|
| solidWhiteRight.mp4 | video | 960×540 | 221 | solid white right line, dashed left line |
| solidYellowLeft.mp4 | video | 960×540 | 681 | solid yellow left line |
| challenge.mp4 | video | 1280×720 | 251 | curve, tree shadows, concrete bridge (light road surface), car hood visible |
| 6 × test_images/*.jpg | image | 960×540 | – | still frames, includes a lane change (`whiteCarLaneSwitch.jpg`) |

- **Labels:** none. The evaluation therefore uses label-free metrics plus a simulated-drift test with known ground truth
  (see README, "Results").
- **Download:** `bash ../download_data.sh lane` (from this folder) puts the files in `../data/lane/`.
- The data are not stored in this repository. The annotated outputs are in `results/` and `samples/`.
