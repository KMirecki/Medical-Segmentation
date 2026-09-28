# Medical Segmentation Benchmark

PyTorch project designed to automatically train, evaluate and compare medical image segmentation models using grid search.

## Grid Search Space:
- **Architectures**: U-Net, U-Net++, DeepLab v3, DeepLab v3+, SegFormer
- **Encoders (Backbones)**: ResNet-34, EfficientNet-B0, MobileNet v2
- **Optimizers**: SGD, Adam, AdamW

## Evaluation and Metrics:
- IoU (Jaccard Index)
- F1 (Dice Coefficient)
- Precision
- Recall
- Side-by-side visual comparisons (Input Image | Ground Truth | Prediction)
