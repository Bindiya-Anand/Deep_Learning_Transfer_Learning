# Transfer_Learning
Transfer Learning with ResNet50

### Evaluating Freezing and Unfreezing Strategies for Image Classification

This project investigates **transfer learning using ResNet50 pretrained on ImageNet** for fine-grained flower classification on the **Oxford Flowers102** dataset.

The study focuses on how different **freezing, partial unfreezing, full fine-tuning, and gradual unfreezing strategies** affect model performance when working with a small training dataset.

## Project at a Glance

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
