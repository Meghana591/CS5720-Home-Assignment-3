# CS5720 Neural Network and Deep Learning

## Home Assignment 3

**University:** University of Central Missouri
**Course:** CS5720 Neural Network and Deep Learning
**Semester:** Fall 2026
**Student Name:** Meghana Haritha Nalla

---

## Assignment Overview

This assignment covers convolution, CNN architectures, representation learning, transfer learning, and fine-tuning.

The programming portion contains two questions:

1. Implement 2-D convolution from scratch using NumPy.
2. Compare transfer learning using a frozen feature extractor with fine-tuning using a pretrained ResNet18 model.

---

# Part II — Programming Questions

## Question 1 — Implement Convolution from Scratch

### Objective

The purpose of this program is to implement 2-D convolution/filtering without using a built-in convolution function.

The input is a 5 × 5 matrix and the filter is a 3 × 3 matrix.

### Input Matrix

```text
1 1 1 0 0
0 1 1 1 0
0 0 1 1 1
0 0 1 1 0
0 1 1 0 0
```

### Filter

```text
1 0 1
0 1 0
1 0 1
```

The program uses:

- Stride = 1
- Padding = 0
- NumPy arrays
- Manual filter sliding
- Dot product at each valid location

### Output

```text
[[4 3 4]
 [2 4 3]
 [2 3 4]]
```

### Output Shape

```text
(3, 3)
```

### Stride Explanation

When the stride changes from 1 to 2, the filter moves two positions at a time instead of one. Therefore, fewer positions are processed and the output feature map becomes smaller.

---

# Question 2 — Transfer Learning: Freeze vs. Fine-Tune

## Objective

The purpose of this experiment is to compare two transfer-learning approaches:

1. Frozen Feature Extraction
2. Fine-Tuning

A pretrained **ResNet18** model is used for image classification.

The CIFAR-10 dataset is used for the experiment.

---

## Experiment A — Feature Extraction

For the first experiment, the pretrained ResNet18 convolutional/base layers are frozen.

Only the final classification layer is trained.

Freezing the base layers means that their pretrained weights are not changed during training.

### Steps

1. Load pretrained ResNet18.
2. Freeze the convolutional/base layers.
3. Replace the final classification layer.
4. Train only the new classifier.
5. Record training time and accuracy.
6. Plot the training loss.

---

## Experiment B — Fine-Tuning

For the second experiment, the pretrained ResNet18 model is used again.

The last convolutional block is unfrozen along with the final classification layer.

These layers are trained on the new dataset.

### Steps

1. Load pretrained ResNet18.
2. Freeze the initial layers.
3. Unfreeze the last convolutional block.
4. Train the last block and final classifier.
5. Record training time and accuracy.
6. Plot the training loss.

---

## Comparison

| Method                   | Trainable Parameters |      Training Time |           Accuracy |
| ------------------------ | -------------------: | -----------------: | -----------------: |
| Frozen Feature Extractor |   **[enter result]** | **[enter result]** | **[enter result]** |
| Fine-Tuned Network       |   **[enter result]** | **[enter result]** | **[enter result]** |

The actual values above should be taken from the program output after running `Q2_Transfer_Learning.py`.

---

## Training Loss

The program generates the following training-loss graph:

```text
training_loss_comparison.png
```

The graph compares the training loss of the frozen feature extractor and the fine-tuned network over the same number of epochs.

---

## Discussion

In this experiment, a pretrained ResNet18 model was used for image classification. In the feature-extraction experiment, the convolutional layers were frozen and only the final classification layer was trained. In the fine-tuning experiment, the last convolutional block and the final classification layer were trained. The frozen feature extractor has fewer trainable parameters, so it generally requires less training time. Fine-tuning has more trainable parameters, which allows the model to adapt its learned features to the new dataset. Because more layers are updated, fine-tuning can require more training time but may provide better accuracy. The training-loss graph helps compare how the two approaches learn during training.

---

# Files

The repository contains the following files:

```text
CS5720-Home-Assignment-3/
│
├── Q1_Convolution.py
├── Q2_Transfer_Learning.py
├── training_loss_comparison.png
└── README.md
```

---

# How to Run

## Question 1

Open a terminal in the assignment folder and run:

```bash
python Q1_Convolution.py
```

## Question 2

Make sure PyTorch and torchvision are installed.

Run:

```bash
python Q2_Transfer_Learning.py
```

The program will train both models, display the results, and create:

```text
training_loss_comparison.png
```

---

# Technologies Used

- Python
- NumPy
- PyTorch
- Torchvision
- Matplotlib
- ResNet18
- CIFAR-10

---

# Conclusion

This assignment demonstrates how convolution works from scratch and how transfer learning can be used with a pretrained CNN. The comparison between feature extraction and fine-tuning shows how changing the number of trainable layers can affect training time and model performance.
