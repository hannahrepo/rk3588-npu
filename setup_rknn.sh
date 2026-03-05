#!/bin/bash

# =================================================================
# RKNN-Toolkit2 v2.3.0 Environment Deployment Script
# Target: Ubuntu 22.04 / Python 3.10 (x86_64)
# Fix: Resolved 'pkg_resources' missing and ONNX version conflicts
# =================================================================

set -e # Exit immediately if a command exits with a non-zero status

# Define relative paths based on your directory structure
PKG_DIR="rknn-toolkit2-2.3.0/rknn-toolkit2/packages/x86_64"

echo "----------------------------------------------------"
echo "Step 1: Installing system-level dependencies (apt-get)"
echo "----------------------------------------------------"
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-pip python3-venv
sudo apt-get install -y libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 libsm6 \
libgl1-mesa-glx libprotobuf-dev gcc

echo "----------------------------------------------------"
echo "Step 2: Managing Python Virtual Environment (venv)"
echo "----------------------------------------------------"

if [ ! -d "rknn_env" ]; then
    echo "Creating a new virtual environment: rknn_env..."
    python3 -m venv rknn_env
else
    echo "Virtual environment 'rknn_env' already exists."
fi

# Activate the virtual environment
source rknn_env/bin/activate
echo "Virtual environment 'rknn_env' is now active."

echo "----------------------------------------------------"
echo "Step 3: Installing Python dependencies"
echo "----------------------------------------------------"
# 1. Upgrade pip and install fundamental build tools
# We install setuptools twice to ensure it exists after requirement resolution
pip3 install --upgrade pip
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple setuptools==65.5.0 wheel

# Locate the official requirements file
REQ_FILE=$(find "$PKG_DIR" -name "requirements_cp310*.txt" | head -n 1)

if [ -f "$REQ_FILE" ]; then
    echo "Found official requirements: $REQ_FILE"
    
    # 2. Lock critical versions to avoid conflicts during -r install
    echo "Locking ONNX and Protobuf for RKNN compatibility..."
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple onnx==1.16.0 protobuf==3.20.3
    
    # 3. Install remaining requirements from official file
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r "$REQ_FILE"
else
    echo "Error: Could not find requirements_cp310*.txt in $PKG_DIR"
    exit 1
fi

# 4. CRITICAL FIX: Re-verify setuptools is installed
# Some dependency resolutions in Step 3 might uninstall it
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple setuptools

echo "----------------------------------------------------"
echo "Step 4: Installing RKNN-Toolkit2 core package (.whl)"
echo "----------------------------------------------------"
WHL_PATH=$(find "$PKG_DIR" -name "*cp310*x86_64.whl" | head -n 1)

if [ -f "$WHL_PATH" ]; then
    echo "Installing RKNN-Toolkit2 wheel: $WHL_PATH"
    pip3 install "$WHL_PATH"
else
    echo "Error: Could not find the cp310 .whl package in $PKG_DIR"
    exit 1
fi

echo "----------------------------------------------------"
echo "Step 5: Verifying Installation"
echo "----------------------------------------------------"
# The 'pkg_resources' error usually happens here. 
# Re-installing setuptools in Step 3.4 prevents this.
python3 -c "from rknn.api import RKNN; rknn=RKNN(); print('Success: RKNN-Toolkit2 environment is ready!')" || { 
    echo "Verification failed! Trying one last fix for setuptools..."; 
    pip3 install setuptools;
    python3 -c "from rknn.api import RKNN; print('Success after fix: RKNN-Toolkit2 is ready!')";
}

echo "===================================================="
echo "Deployment Completed Successfully!"
echo "source $(pwd)/rknn_env/bin/activate"
echo "===================================================="