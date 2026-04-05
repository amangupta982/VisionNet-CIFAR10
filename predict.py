"""
============================================================
  FILE: predict.py

  Usage:
    python3 predict.py                     → classify 12 random test images
    python3 predict.py --image cat.jpg     → classify your own image
    python3 predict.py --count 20          → classify 20 random test images
    python3 predict.py --class cat         → show test images for one class
============================================================
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os
import sys

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH  = "cifar10_cnn_model.keras"
OUTPUT_DIR  = "predictions"
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

CLASS_COLORS = {
    'airplane':    '#4A90D9',
    'automobile':  '#E67E22',
    'bird':        '#27AE60',
    'cat':         '#8E44AD',
    'deer':        '#16A085',
    'dog':         '#D35400',
    'frog':        '#2ECC71',
    'horse':       '#F39C12',
    'ship':        '#2980B9',
    'truck':       '#C0392B',
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Model file '{MODEL_PATH}' not found.")
        print("        Run 'python3 train.py' first to train and save the model.")
        sys.exit(1)
    print(f"Loading model from '{MODEL_PATH}'...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.\n")
    return model


def predict_single(model, img_array):
    """img_array: float32 (32,32,3) in [0,1]. Returns (class_idx, confidence, all_probs)."""
    inp   = np.expand_dims(img_array, axis=0)
    probs = model.predict(inp, verbose=0)[0]
    idx   = np.argmax(probs)
    return idx, probs[idx], probs


def preprocess_external_image(path):
    """Load any user-provided image, resize to 32x32, normalize."""
    img   = load_img(path, target_size=(32, 32))
    arr   = img_to_array(img).astype('float32') / 255.0
    return arr


def confidence_bar(ax, probs, predicted_idx, true_idx=None):
    """Draw a horizontal bar chart of class probabilities."""
    colors = []
    for i, cls in enumerate(CLASS_NAMES):
        if i == predicted_idx:
            colors.append('#2ECC71')          # green = predicted
        elif true_idx is not None and i == true_idx and i != predicted_idx:
            colors.append('#E74C3C')          # red = wrong true label
        else:
            colors.append('#BDC3C7')

    bars = ax.barh(CLASS_NAMES, probs * 100, color=colors, edgecolor='none', height=0.6)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=7)
    ax.spines[['top','right','left']].set_visible(False)
    ax.axvline(x=50, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    for bar, prob in zip(bars, probs):
        if prob * 100 > 3:
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f'{prob*100:.1f}%', va='center', ha='left', fontsize=7)


# ─────────────────────────────────────────────
# MODE 1: Predict on your own image
# ─────────────────────────────────────────────
def predict_user_image(model, image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: {image_path}")
        sys.exit(1)

    print(f"Classifying: {image_path}")
    img_arr = preprocess_external_image(image_path)
    pred_idx, confidence, probs = predict_single(model, img_arr)
    pred_class = CLASS_NAMES[pred_idx]

    fig = plt.figure(figsize=(10, 4))
    fig.suptitle("CIFAR-10 Image Classifier — Prediction", fontsize=13, fontweight='bold', y=1.01)

    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1.2, 2], wspace=0.4)

    # Original image
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img_arr)
    ax1.axis('off')
    ax1.set_title("Your Image", fontsize=9, color='gray')

    # Big prediction box
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis('off')
    color = CLASS_COLORS.get(pred_class, '#3498DB')
    ax2.add_patch(mpatches.FancyBboxPatch((0.05, 0.2), 0.9, 0.6,
        boxstyle="round,pad=0.05", facecolor=color, alpha=0.15, edgecolor=color, linewidth=2))
    ax2.text(0.5, 0.72, "Prediction", ha='center', va='center',
             fontsize=9, color='gray')
    ax2.text(0.5, 0.55, pred_class.upper(), ha='center', va='center',
             fontsize=18, fontweight='bold', color=color)
    ax2.text(0.5, 0.35, f"{confidence*100:.1f}% confident",
             ha='center', va='center', fontsize=10, color='#555')

    # Confidence bar chart
    ax3 = fig.add_subplot(gs[2])
    confidence_bar(ax3, probs, pred_idx)
    ax3.set_title("Class Probabilities", fontsize=9, color='gray')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"prediction_{os.path.basename(image_path)}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n{'='*40}")
    print(f"  Prediction : {pred_class.upper()}")
    print(f"  Confidence : {confidence*100:.2f}%")
    print(f"  Saved      : {out_path}")
    print(f"{'='*40}")


# ─────────────────────────────────────────────
# MODE 2: Random test-set predictions (grid)
# ─────────────────────────────────────────────
def predict_random_samples(model, count=12, filter_class=None):
    print("Loading CIFAR-10 test set...")
    (_, _), (X_test, y_test) = cifar10.load_data()
    X_test = X_test.astype('float32') / 255.0
    y_true = y_test.flatten()

    if filter_class:
        cls_idx = CLASS_NAMES.index(filter_class)
        indices = np.where(y_true == cls_idx)[0]
        label   = f"— showing '{filter_class}' images"
    else:
        indices = np.arange(len(X_test))
        label   = ""

    chosen = np.random.choice(indices, min(count, len(indices)), replace=False)

    cols = 4
    rows = (len(chosen) + cols - 1) // cols
    fig  = plt.figure(figsize=(cols * 3.5, rows * 3.8))
    fig.suptitle(f"CNN Predictions on CIFAR-10 Test Set {label}",
                 fontsize=13, fontweight='bold', y=1.01)

    correct = 0
    for pos, idx in enumerate(chosen):
        img   = X_test[idx]
        true  = y_true[idx]
        p_idx, conf, probs = predict_single(model, img)
        is_correct = (p_idx == true)
        if is_correct:
            correct += 1

        gs  = GridSpec(rows, cols, figure=fig, hspace=0.6, wspace=0.4)
        r, c = divmod(pos, cols)
        inner = gs[r, c].subgridspec(2, 1, height_ratios=[1, 1.4], hspace=0.1)

        # Image
        ax_img = fig.add_subplot(inner[0])
        ax_img.imshow(img)
        ax_img.axis('off')
        border_color = '#2ECC71' if is_correct else '#E74C3C'
        for spine in ax_img.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
            spine.set_visible(True)

        status = "✓" if is_correct else "✗"
        title  = f"{status} Pred: {CLASS_NAMES[p_idx]}  ({conf*100:.0f}%)\n   True: {CLASS_NAMES[true]}"
        ax_img.set_title(title, fontsize=7.5,
                         color=border_color, pad=3)

        # Confidence bars
        ax_bar = fig.add_subplot(inner[1])
        confidence_bar(ax_bar, probs, p_idx, true)

    accuracy = correct / len(chosen) * 100
    fig.text(0.5, -0.01,
             f"Accuracy on this sample: {correct}/{len(chosen)} = {accuracy:.1f}%   |   "
             f"Green border = correct   Red border = incorrect",
             ha='center', fontsize=9, color='gray')

    out_path = os.path.join(OUTPUT_DIR, "random_predictions.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n{'='*40}")
    print(f"  Sample accuracy : {correct}/{len(chosen)} = {accuracy:.1f}%")
    print(f"  Saved           : {out_path}")
    print(f"{'='*40}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 CNN Predictor",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python3 predict.py                        # 12 random test images
  python3 predict.py --count 20             # 20 random test images
  python3 predict.py --image mycat.jpg      # classify your own image
  python3 predict.py --class dog            # show only dog test images
        """
    )
    parser.add_argument('--image', type=str, default=None,
                        help='Path to your own image file (jpg/png)')
    parser.add_argument('--count', type=int, default=12,
                        help='Number of random test images to classify (default: 12)')
    parser.add_argument('--class', dest='filter_class', type=str, default=None,
                        choices=CLASS_NAMES,
                        help='Filter test images by class name')
    args = parser.parse_args()

    model = load_trained_model()

    if args.image:
        predict_user_image(model, args.image)
    else:
        predict_random_samples(model, count=args.count, filter_class=args.filter_class)


if __name__ == '__main__':
    main()
