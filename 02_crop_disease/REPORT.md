# Report: Tomato Leaf Disease Classification (Transfer Learning + Grad-CAM)

## 1. Problem
Identify tomato leaf diseases from a single photo (10 classes including healthy) and make the decision explainable.

## 2. Data
PlantVillage tomato subset: 11,000 images, 10 balanced classes. 9,000 train / 1,000 validation / 1,000 held-out test.
Details in [DATASET.md](DATASET.md).

## 3. Method
ResNet-18 pretrained on ImageNet with a new 10-class head. Two transfer strategies are compared: linear probe (frozen
backbone) and full fine-tuning. A third variant adds blur/noise augmentation. Preprocessing: resize/crop to 224, ImageNet
normalisation, and flip/crop/colour augmentation for training. Explainability: Grad-CAM on the last convolutional block.

## 4. Results
| model | test accuracy | macro-F1 |
|---|---|---|
| linear probe | 79.9% | 79.5% |
| fine-tune | 98.2% | 98.2% |
| fine-tune + blur/noise aug | **98.4%** | **98.4%** |

- Per-class F1 is 0.955–1.000. Errors occur among visually similar spot diseases (Target Spot, Septoria, Bacterial spot, Spider mites).
- **Robustness:** the plain fine-tuned model drops to 19% (blur) and 22% (noise). With augmentation: 97.1% and 97.4%. JPEG quality 10:
  75% → 82%. Lighting changes cost 2–6 points for both models.
- **Grad-CAM:** 69% of activation falls on the leaf (which covers 53% of the image). Most activation is on the leaf in 94% of images.

## 5. Challenges
- **Over-reliance on fine texture,** found by the robustness test and fixed with augmentation.
- **Background shortcuts** are a known PlantVillage problem. We measured the leaf-vs-background share of Grad-CAM instead of
  only showing pretty heat-maps. The leaf mask is a simple saturation threshold and fails on yellowed leaves, so 341 of the
  1,000 images with a clean mask were used.
- **Library API change:** `pytorch-grad-cam` now requires an explicit `targets` argument.

## 6. Limitations
- **Lab images only.** Results do not transfer directly to field photos with several leaves, soil and natural light. A field test
  set (e.g. PlantDoc) would be needed to measure this. We expect a large drop.
- **One run per configuration,** no confidence intervals. With 1,000 test images, ±1 point is within noise, so fine-tune and
  fine-tune + aug are equal on clean data.
- **Possible duplicates:** the same physical leaf may appear in train and test, which can inflate accuracy.
- **Approximate measurements:** the robustness degradations are synthetic and the Grad-CAM leaf mask is approximate.

## 7. Conclusion
Fine-tuning a small pretrained CNN is enough for 98% accuracy on PlantVillage tomato leaves in under 2 minutes of GPU time.
The most useful findings came from going beyond accuracy: Grad-CAM indicates the model attends to lesions, and the robustness test
exposed (and augmentation fixed) a fragility to blur and noise that a plain accuracy number hides.
