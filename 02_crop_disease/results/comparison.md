| model | trainable params | epochs | train time (s) | test accuracy | test macro-F1 |
|---|---|---|---|---|---|
| ResNet-18 linear probe (frozen backbone) | 5,130 | 5 | 50 | 79.9% | 79.5% |
| ResNet-18 full fine-tune | 11,181,642 | 5 | 81 | 98.2% | 98.2% |
| ResNet-18 full fine-tune + blur/noise augmentation | 11,181,642 | 5 | 105 | 98.4% | 98.4% |
