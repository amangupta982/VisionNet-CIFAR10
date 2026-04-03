# Image Classification using CNN (CIFAR-10)

**23CSE610 — Artificial Neural Networks and Deep Learning**  
Jain University | Department of CSE – Data Science

---

## Overview

A Convolutional Neural Network (CNN) trained to classify images into 10 categories from the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

**Classes:** airplane · automobile · bird · cat · deer · dog · frog · horse · ship · truck

**Expected Accuracy:** ~78–82% on the test set.

---

## Project Structure

```
ANNDL-Project/
│
├── train.py                    # Train the CNN and save the model
├── predict.py                  # Load model and classify images
├── requirements.txt            # Python dependencies
├── setup_env.sh                # One-time environment setup
│
├── cifar10_cnn_model.keras     # Saved model (generated after training)
│
├── plots/                      # Generated during training
│   ├── sample_images.png
│   ├── training_history.png
│   └── confusion_matrix.png
│
└── predictions/                # Generated during prediction
    └── random_predictions.png
```

---

## Setup (One Time)

```bash
# Clone the repo
git clone https://github.com/<your-username>/ANNDL-Project.git
cd ANNDL-Project

# Create isolated environment and install dependencies
bash setup_env.sh

# Activate the environment
source p5env/bin/activate
```

---

## Usage

### Step 1 — Train the model
```bash
python3 train.py
```
- Trains for up to 30 epochs with early stopping
- **Auto-saves the best model** to `cifar10_cnn_model.keras` whenever validation accuracy improves
- Saves plots to `plots/`

### Step 2 — Run predictions

```bash
# Classify 12 random test images (default)
python3 predict.py

# Classify more images
python3 predict.py --count 20

# Classify your own image
python3 predict.py --image /path/to/your/cat.jpg

# Show only images of a specific class
python3 predict.py --class dog
```

Each prediction shows:
- The image
- Predicted class + confidence %
- Full probability bar chart for all 10 classes
- Green border = correct, Red border = incorrect

---

## Model Architecture

| Block | Layers | Output Shape |
|-------|--------|-------------|
| Input | — | 32 × 32 × 3 |
| Conv Block 1 | Conv2D(32) + BN + Conv2D(32) + MaxPool + Dropout | 16 × 16 × 32 |
| Conv Block 2 | Conv2D(64) + BN + Conv2D(64) + MaxPool + Dropout | 8 × 8 × 64 |
| Conv Block 3 | Conv2D(128) + BN + Conv2D(128) + MaxPool + Dropout | 4 × 4 × 128 |
| FC | Flatten + Dense(512) + BN + Dropout(0.5) | 512 |
| Output | Dense(10) + Softmax | 10 |

**Total parameters:** ~1.34M

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~80% |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Epochs (max) | 30 (early stopping) |

---

## Requirements

- Python 3.10
- TensorFlow 2.15.0
- NumPy 1.26.4
- See `requirements.txt` for full list
