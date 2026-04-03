"""
============================================================
  23CSE610 - Artificial Neural Networks and Deep Learning
  Project 5: Image Classification using CNN
  FILE: train.py
  
  Run: python3 train.py
  Output: saves model to 'cifar10_cnn_model.keras'
============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH  = "cifar10_cnn_model.keras"
PLOTS_DIR   = "plots"
BATCH_SIZE  = 64
EPOCHS      = 30

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1: Load Dataset
# ─────────────────────────────────────────────
print("\n[1/7] Loading CIFAR-10 dataset...")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(f"  Training samples : {X_train.shape[0]}")
print(f"  Testing  samples : {X_test.shape[0]}")
print(f"  Image shape      : {X_train.shape[1:]}")
print(f"  Number of classes: {len(CLASS_NAMES)}")


# ─────────────────────────────────────────────
# STEP 2: Preprocess
# ─────────────────────────────────────────────
print("\n[2/7] Preprocessing data...")
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32')  / 255.0
y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test,  10)
print(f"  Pixel range : [0.00, 1.00]")
print(f"  Labels shape: {y_train_cat.shape} (one-hot encoded)")


# ─────────────────────────────────────────────
# STEP 3: Visualize Samples
# ─────────────────────────────────────────────
print("\n[3/7] Saving sample images plot...")
plt.figure(figsize=(14, 3))
plt.suptitle("Sample Images from CIFAR-10 Dataset", fontsize=13, y=1.02)
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(X_train[i])
    plt.title(CLASS_NAMES[y_train[i][0]], fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/sample_images.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {PLOTS_DIR}/sample_images.png")


# ─────────────────────────────────────────────
# STEP 4: Data Augmentation
# ─────────────────────────────────────────────
print("\n[4/7] Configuring data augmentation...")
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(X_train)


# ─────────────────────────────────────────────
# STEP 5: Build CNN
# ─────────────────────────────────────────────
print("\n[5/7] Building CNN model...")

model = Sequential([
    # Conv Block 1
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)),
    BatchNormalization(),
    Conv2D(32, (3,3), padding='same', activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Conv Block 2
    Conv2D(64, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Conv Block 3
    Conv2D(128, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3,3), padding='same', activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Fully Connected
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# ─────────────────────────────────────────────
# STEP 6: Train — saves best model automatically
# ─────────────────────────────────────────────
print(f"\n[6/7] Training for up to {EPOCHS} epochs...")
print(f"      Best model will be auto-saved to '{MODEL_PATH}'")

callbacks = [
    # Saves the best model every time val_accuracy improves
    ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test_cat),
    callbacks=callbacks,
    verbose=1
)

print(f"\n  Model saved → {MODEL_PATH}")


# ─────────────────────────────────────────────
# STEP 7: Evaluate + Save All Plots
# ─────────────────────────────────────────────
print("\n[7/7] Evaluating and saving plots...")

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true = y_test.flatten()

print(f"\n{'='*45}")
print(f"  Test Accuracy : {test_acc * 100:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}")
print(f"{'='*45}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Plot 1 — Training History
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Training History", fontsize=13)
axes[0].plot(history.history['accuracy'],     label='Train', color='steelblue')
axes[0].plot(history.history['val_accuracy'], label='Val',   color='coral', linestyle='--')
axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(history.history['loss'],     label='Train', color='steelblue')
axes[1].plot(history.history['val_loss'], label='Val',   color='coral', linestyle='--')
axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/training_history.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {PLOTS_DIR}/training_history.png")

# Plot 2 — Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix — CIFAR-10", fontsize=13)
plt.ylabel("True Label"); plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {PLOTS_DIR}/confusion_matrix.png")

# Per-class accuracy
print("\nPer-Class Accuracy:")
print("-" * 32)
for i, cls in enumerate(CLASS_NAMES):
    mask    = y_true == i
    cls_acc = accuracy_score(y_true[mask], y_pred[mask])
    bar     = '█' * int(cls_acc * 20)
    print(f"  {cls:12s}: {bar:<20} {cls_acc*100:.1f}%")

print(f"\n{'='*45}")
print(f"  Training complete. Final accuracy: {test_acc*100:.2f}%")
print(f"  Model saved at: {MODEL_PATH}")
print(f"  Plots saved in: {PLOTS_DIR}/")
print(f"{'='*45}")
print("\n  Run predict.py to classify your own images!")
