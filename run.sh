#!/bin/bash

echo "--- Setting up Python virtual environment for Federated Learning ---"
ENV_NAME="fed_learning_env"
source $ENV_NAME/bin/activate

echo "--- Starting Federated Learning Training ---"
python3 fed_avg.py
echo "--- Training completed ---"

echo "Plotting results..."
python3 fed_avg_plots.py
echo "--- Plotting completed ---"

# echo "Environemnt dectivating..."
# deactivate
# echo "--- Environment deactivated ---"

