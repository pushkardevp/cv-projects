# Report: Road Pothole Detection with YOLOv8

## 1. Problem
Detect and localise potholes in road images with bounding boxes, as a building block for automatic road-condition surveys.

## 2. Data
Kaggle pothole dataset: 665 images, 1,739 boxes, one class. 70/15/15 split by image (465/100/100). See [DATASET.md](DATASET.md).

## 3. Method
VOC XML → YOLO conversion. COCO-pretrained YOLOv8n and YOLOv8s fine-tuned for 50 epochs at 640 px with Ultralytics'
default augmentation (mosaic, HSV, flip, scale). Model selection uses validation mAP. A single final evaluation is run on the test split.

## 4. Results
| model | precision | recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| YOLOv8n | 0.858 | 0.702 | 0.810 | 0.505 |
| YOLOv8s | 0.845 | 0.713 | 0.796 | 0.505 |

Recall by size (YOLOv8n, conf 0.25): small 69%, medium 79%, large 89%. 0.67 false positives per image.

## 5. Challenges
- **Small dataset:** transfer learning from COCO and strong augmentation make 465 training images usable. The larger model gives
  no benefit at this scale.
- **Small, distant potholes:** at 640 px they cover only a few pixels and are missed most often.
- **Annotation inconsistency:** several "false positives" look like real, unlabelled potholes. This lowers measured precision
  and adds noise to the evaluation.
- **Mixed image sources:** close-ups and wide street views, with very different pothole scales.

## 6. Limitations
- A test set of 100 images gives coarse estimates (±0.02 mAP is not meaningful). There was one split and one run per model.
- No temporal or video evaluation. In real deployment the same pothole is seen across many frames, which needs tracking to count it once.
- Boxes only: no segmentation, depth or severity, which a road authority would need for prioritising repairs.
- Weather, night and wet roads (water-filled potholes look like puddles) are under-represented.

## 7. Conclusion
A 3-million-parameter YOLOv8n, fine-tuned in about 4 minutes, reaches 0.81 mAP@0.5 on held-out images and runs in about 10 ms per
image. The main weakness is small potholes. Higher input resolution, more data and consistent labelling are the next steps.
