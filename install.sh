#!/bin/bash

# Define the name for your virtual environment
ENV_NAME="fed_learning_env"

echo "--- Setting up Python virtual environment ---"
# Create a virtual environment
python3 -m venv $ENV_NAME

# Activate the virtual environment
echo "Activating virtual environment: $ENV_NAME"
source $ENV_NAME/bin/activate

echo "--- Installing necessary Python libraries ---"

# Install NumPy
echo "Installing NumPy..."
pip install numpy matplotlib ipython scikit-learn scipy pandas wandb 

# Install PyTorch and TorchVision
# IMPORTANT: Choose the correct command below based on your system.
#
# Option 1: For CPU-only (no GPU)
echo "Installing PyTorch (CPU version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Option 2: For CUDA 11.8 (NVIDIA GPU) - Common for many systems
# Uncomment the line below if you have a CUDA-enabled GPU and CUDA 11.8 installed
# echo "Installing PyTorch (CUDA 11.8 version)..."
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Option 3: For CUDA 12.1 (NVIDIA GPU) - Newer systems
# Uncomment the line below if you have a CUDA-enabled GPU and CUDA 12.1 installed
# echo "Installing PyTorch (CUDA 12.1 version)..."
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# You might need to adjust the CUDA version based on your specific NVIDIA driver and CUDA Toolkit installation.
# Visit https://pytorch.org/get-started/locally/ for the most up-to-date and specific installation commands.

echo "--- Verifying installations ---"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import numpy as np; print(f'NumPy version: {np.__version__}')"

echo "--- Installation script finished ---"
echo "To activate the environment in the future, run: source $ENV_NAME/bin/activate"
echo "To deactivate the environment, run: deactivate"