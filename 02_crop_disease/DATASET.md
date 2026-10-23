# Dataset: PlantVillage tomato leaves

- **Kaggle:** https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf (licence CC0-1.0), a tomato-only subset of
  **PlantVillage** (Hughes & Salathé, 2015, https://arxiv.org/abs/1511.08060; original data https://github.com/spMohanty/PlantVillage-Dataset).
- **Classes (10):** Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites (two-spotted),
  Target Spot, Tomato Yellow Leaf Curl Virus, Tomato mosaic virus, healthy.
- **Size:** `train/` 10 × 1000 images, `val/` 10 × 100 images, 256×256 RGB JPEG. Classes are balanced.
- **How we use it:** `train/` is split 90/10 (seed 0) into training and validation (model selection). The provided `val/`
  folder is our **test set** and is never used for training or selection.
- **Download:** `bash ../download_data.sh tomato` (needs a Kaggle API token), giving `../data/tomato/tomato/{train,val}/<class>/*.jpg`.
- **Known bias:** lab conditions (single detached leaf, uniform background, controlled light). Images of the same leaf
  may appear several times from different angles, so train/test are not guaranteed leaf-disjoint.
