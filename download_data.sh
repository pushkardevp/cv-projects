#!/usr/bin/env bash
# Download the datasets into ./data (git-ignored).
# Usage: bash download_data.sh [lane|tomato|pothole|ddd|mot17|all]
# Kaggle datasets need a Kaggle API token: https://www.kaggle.com/settings -> API -> Create New Token,
# then either save it to ~/.kaggle/access_token (new tokens) or ~/.kaggle/kaggle.json (legacy), and `pip install kaggle`.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
what="${1:-all}"

lane() {
  mkdir -p data/lane && cd data/lane
  base=https://raw.githubusercontent.com/udacity/CarND-LaneLines-P1/master
  for f in solidWhiteRight solidYellowLeft challenge; do curl -sSLO "$base/test_videos/$f.mp4"; done
  for f in solidWhiteCurve solidWhiteRight solidYellowCurve solidYellowCurve2 solidYellowLeft whiteCarLaneSwitch; do
    curl -sSLO "$base/test_images/$f.jpg"; done
  cd ../..
}
tomato()  { kaggle datasets download -d kaustubhb999/tomatoleaf -p data/tomato --unzip; }
pothole() { kaggle datasets download -d andrewmvd/pothole-detection -p data/pothole --unzip
            python 03_pothole_detection/src/prepare_data.py --src data/pothole --dst data/pothole_yolo; }
ddd()     { kaggle datasets download -d ismailnasri20/driver-drowsiness-dataset-ddd -p data/ddd --unzip; }
mot17()   { mkdir -p data/mot17 && curl -L -o data/mot17/MOT17.zip https://motchallenge.net/data/MOT17.zip
            (cd data/mot17 && unzip -q MOT17.zip -x "*/test/*"); }

case "$what" in
  all) lane; tomato; pothole; ddd; mot17 ;;
  *) "$what" ;;
esac
echo "done: $what"
