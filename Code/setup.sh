#!/bin/bash

echo "Setting up your environment for Quantum Convolutional Neural Network..."

# Update and install basic system packages (for PIL and other Python dependencies)
sudo apt update
sudo apt install -y python3 python3-pip python3-venv build-essential libjpeg-dev zlib1g-dev

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install numpy matplotlib Pillow pennylane tensorflow

# Optional: install GPU support (only if your system supports it)
# pip install tensorflow-gpu

echo "All dependencies installed successfully."
echo "To activate the environment in future sessions, run:"
echo "source venv/bin/activate"
