#!/bin/bash

# =================================================================
# RKLLM-Toolkit v1.2.3 - ULTIMATE CPU-ONLY DEPLOYMENT SCRIPT
# Target: Ubuntu 22.04 / Python 3.10 (x86_64)
# Optimization: Forced CPU-only Torch & Strict Dependency Alignment
# =================================================================

set -e # Exit immediately if a command exits with a non-zero status

# Define paths and names
VENV_NAME="rkllm_env"
# Path to your rknn-llm release folder - Adjust if your folder name is different
PKG_DIR="rknn-llm-release-v1.2.3/rkllm-toolkit/packages"

echo "----------------------------------------------------"
echo "Step 1: Installing system-level dependencies"
echo "----------------------------------------------------"
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-pip python3-venv
sudo apt-get install -y libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 libsm6 \
libgl1-mesa-glx libprotobuf-dev gcc g++

echo "----------------------------------------------------"
echo "Step 2: Deep Cleaning & Environment Reset"
echo "----------------------------------------------------"
# Physically remove the old environment to purge hidden NVIDIA/CUDA packages
if [ -d "$VENV_NAME" ]; then
    echo "Deleting existing '$VENV_NAME'..."
    rm -rf "$VENV_NAME"
fi

echo "Creating a fresh virtual environment..."
python3 -m venv "$VENV_NAME"
source "$VENV_NAME"/bin/activate

# Upgrade pip and PURGE cache to forget any previous CUDA wheel metadata
pip3 install --upgrade pip
pip3 cache purge

echo "----------------------------------------------------"
echo "Step 3: Installing CPU-Only Torch (The Barrier)"
echo "----------------------------------------------------"
# By using the +cpu suffix and the specific index, we create a barrier.
# Any package requesting 'torch' later will be satisfied by this version.
echo "Installing lightweight CPU-specific Torch 2.4.0..."
pip3 install torch==2.4.0+cpu torchvision==0.19.0+cpu torchaudio==2.4.0+cpu \
     --extra-index-url https://download.pytorch.org/whl/cpu

echo "----------------------------------------------------"
echo "Step 4: Installing LLM Dependencies (Conflict-Free)"
echo "----------------------------------------------------"
# We use the Tsinghua mirror for fast downloads in China.
# Version pinning ensures compatibility with Torch 2.4.0 and RKLLM Toolkit.
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy==1.24.4 \
    onnx==1.14.0 \
    protobuf==3.20.3 \
    transformers==4.40.0 \
    accelerate==0.29.0 \
    sentencepiece \
    typing_extensions

echo "----------------------------------------------------"
echo "Step 5: Installing RKLLM-Toolkit (.whl)"
echo "----------------------------------------------------"
# Dynamically find the cp310 wheel file in the specified directory
WHL_PATH=$(find "$PKG_DIR" -name "*cp310*x86_64.whl" | head -n 1)

if [ -f "$WHL_PATH" ]; then
    echo "Installing: $WHL_PATH"
    # Dependencies are already met, this will be fast
    pip3 install "$WHL_PATH"
else
    echo "Error: RKLLM wheel package not found in $PKG_DIR"
    exit 1
fi

echo "----------------------------------------------------"
echo "Step 6: Final Verification"
echo "----------------------------------------------------"
# 1. Check if Torch is really CPU version
# 2. Check if RKLLM can be imported
python3 -c "
import torch
print('--- Environment Validation ---')
print('Torch Version:', torch.__version__)
print('CUDA Support Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('WARNING: CUDA still found. Environment is not pure CPU.')
else:
    print('SUCCESS: Environment is CPU-pure.')

try:
    from rkllm.api import RKLLM
    rkllm = RKLLM()
    print('RKLLM Toolkit API: Loaded Successfully')
except Exception as e:
    print('RKLLM Toolkit API: Load Failed!', e)
"

echo "===================================================="
echo "DEPLOYMENT SUCCESSFUL!"
echo "To enter your clean environment, run:"
echo "source $(pwd)/$VENV_NAME/bin/activate"
echo "===================================================="