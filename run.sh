#!/bin/bash

echo "--- Setting up Python virtual environment for Federated Learning ---"
source fed_learning_env/bin/activate

echo "--- Starting Federated Learning Training ---"
python3 test.py > output.txt 2>&1 &
echo "--- Training completed ---"

# echo "Plotting results..."
# python3 testPlot.py
# echo "--- Plotting completed ---"

# echo "Environemnt dectivating..."
# deactivate
# echo "--- Environment deactivated ---"

