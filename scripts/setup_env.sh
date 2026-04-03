#!/bin/bash
set -e
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="p5env"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  CIFAR-10 CNN — Environment Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ! command -v python3.10 &>/dev/null; then
    echo "[ERROR] python3.10 not found. Install: brew install python@3.10"
    exit 1
fi

echo "[1/4] Creating virtual environment: $ENV_NAME"
python3.10 -m venv "$PROJECT_DIR/$ENV_NAME"

echo "[2/4] Activating..."
source "$PROJECT_DIR/$ENV_NAME/bin/activate"

echo "[3/4] Upgrading pip..."
pip install --upgrade pip --quiet

echo "[4/4] Installing dependencies..."
pip install -r "$PROJECT_DIR/requirements.txt"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done! Commands:"
echo ""
echo "  Activate :  source $ENV_NAME/bin/activate"
echo "  Train    :  python3 train.py"
echo "  Predict  :  python3 predict.py"
echo "  Tests    :  pytest tests/ -v"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
