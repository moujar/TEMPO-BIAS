#!/bin/bash
set -e

# Configuration
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"
CONFIG_FILE="run_config.yaml"

# 1. Check/Create Virtual Environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# 2. Activate Virtual Environment
source "$VENV_DIR/bin/activate"

# 3. Install Dependencies
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies from $REQUIREMENTS_FILE..."
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "Warning: $REQUIREMENTS_FILE not found."
fi

# 4. Run Pipeline
echo "Running TEMPO-BIAS pipeline..."
if [ -f "$CONFIG_FILE" ]; then
    # Ensure current directory is in PYTHONPATH
    export PYTHONPATH=$PYTHONPATH:.
    python -m tempo_bias.main --config "$CONFIG_FILE"
else
    echo "Error: Configuration file $CONFIG_FILE not found!"
    exit 1
fi
