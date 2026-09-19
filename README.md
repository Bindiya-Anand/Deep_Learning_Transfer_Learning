# Transfer Learning
Transfer Learning with ResNet50

### Evaluating Freezing and Unfreezing Strategies for Image Classification

This project investigates **transfer learning using ResNet50 pretrained on ImageNet** for fine-grained flower classification on the **Oxford Flowers102** dataset.

The study focuses on how different **freezing, partial unfreezing, full fine-tuning, and gradual unfreezing strategies** affect model performance when working with a small training dataset.

## Overview

| Component | Details |
|---|---|
| Task | Image Classification |
| Dataset | Oxford Flowers102 |
| Classes | 102 flower categories |
| Training Images | 1,020 |
| Model | ResNet50 |
| Pretrained On | ImageNet |
| Transfer Learning | Feature extraction & fine-tuning |
| Strategies Evaluated | 6 |
| Best Test Accuracy | 47.28% |

The dataset contains only **10 training images per class**, making it a useful setting for studying the effects of transfer learning and catastrophic forgetting. :contentReference[oaicite:1]{index=1}

## Problem

Training a deep neural network from scratch requires large amounts of labelled data and computational resources. Transfer learning addresses this by reusing representations learned from a large source dataset such as ImageNet.

However, when the target dataset is small, **how much of the pretrained network should be frozen or fine-tuned becomes an important consideration**.

This project investigates this problem by comparing different layer-freezing strategies using the same ResNet50 backbone and classification head.

## Objectives

1. Investigate the effectiveness of **transfer learning with ResNet50** on a small, fine-grained image classification dataset.
2. Compare different **freezing and unfreezing strategies** for adapting a pretrained model to the target task.
3. Analyse the effects of **partial fine-tuning, full unfreezing, and gradual unfreezing** on model performance and catastrophic forgetting.

## Dataset

The experiments use the **Oxford Flowers102** dataset:

- **102** flower categories
- **1,020** training images
- **1,020** validation images
- **6,149** test images
- Approximately **10 training images per class**

The limited training data makes the dataset particularly suitable for examining transfer learning under data scarcity. :contentReference[oaicite:2]{index=2}

## Model Architecture

The project uses **ResNet50 pretrained on ImageNet**.

The original ImageNet classification head was replaced with:

```text
ResNet50 Backbone
        ↓
GlobalAveragePooling2D
        ↓
Dense (256, ReLU)
        ↓
Dropout (0.2)
        ↓
Dense (102, Softmax)
```
## Experiments
Six transfer-learning strategies were evaluated:
| Experiment | Strategy | Test Accuracy |
|---|---|---:|
| A | Total Freeze / Evaluation Only | 0.33% |
| B | Feature Extraction | 26.51% |
| C | Partial Unfreezing (conv5 + head) | 34.25% |
| D | Fine-Tuning (conv4 + conv5 + head) | 47.28% |
| E | Full Unfreezing | 23.74% |
| F | Gradual Unfreezing (ULMFiT) | 34.41% |

All experiments used the same ResNet50 backbone and custom classification head, while differing in which layers were trainable.

## Results & Analysis
### Test Accuracy by Freezing Strategy
The results show a substantial improvement from feature extraction to selective fine-tuning. Experiment D achieved the highest test accuracy of **47.28%** among the evaluated strategies.

### Catastrophic Forgetting
Full unfreezing performed substantially worse, reaching **23.74% test accuracy**. The validation accuracy also collapsed during training, illustrating the risk of updating the entire pretrained network when only a small number of target-domain samples are available.

### Gradual Unfreezing
The gradual-unfreezing experiment progressively released deeper layers using different learning rates. This approach avoided the severe collapse observed with full unfreezing and achieved **34.41% test accuracy**.

### Visual Analysis
The project includes visualisations for:

- Test accuracy across freezing strategies
- Validation accuracy learning curves
- Gradual-unfreezing phase analysis
- Training vs. validation accuracy
- Overfitting analysis

These visualisations are included in the ```results/``` directory.

## Key Findings
- Pretrained ImageNet representations transferred effectively to the flower-classification task.
- Selective fine-tuning of deeper layers substantially improved performance compared with training only the classification head.
- Fine-tuning conv4_x and conv5_x produced the highest test accuracy of 47.28%.
- Full unfreezing resulted in catastrophic forgetting on the small training dataset.
- Gradual unfreezing reduced the instability associated with full unfreezing.
- The experiments showed a large train-validation gap, highlighting the limitations imposed by the small training set.

## Technologies
- Python
- TensorFlow / Keras
- ResNet50
- ImageNet
- Oxford Flowers102
- NumPy
- Matplotlib

## Repository Structure
```
Transfer-Learning/
│
├── transfer_learning_clean.ipynb
│
├── results/
│   ├── accuracy_comparison.png
│   ├── learning_curves.png
│   ├── gradual_unfreezing.png
│   └── train_vs_validation.png
│
└── README.md
```

## Academic Context
**Deep Learning Mini Project**

MSc Computer Science - Artificial Intelligence & Machine Learning

South Asian University, New Delhi

2026
