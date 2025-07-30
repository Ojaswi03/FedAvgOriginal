#!/bin/bash

echo "--- Setting up Python virtual environment for Federated Learning ---"
source fed_learning_env/bin/activate

echo "--- Starting Federated Learning Training ---"
python3 test.py # Assuming test.py is the main script for training

echo "--- Training completed ---"

# echo "Plotting results..."
# python3 testPlot.py
# echo "--- Plotting completed ---"

# echo "Environemnt dectivating..."
# deactivate
# echo "--- Environment deactivated ---"

