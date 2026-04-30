#!/bin/bash
set -e

echo "=== Dynamic Pricing AI ==="

# Train initial model if none exists
if [ ! -f /app/models/q_table.npy ]; then
    echo "No model found — running initial training (2000 episodes)..."
    cd /app && python src/train.py --episodes 2000
    echo "Initial training complete."
fi

echo "Starting Flask app on port 5000..."
cd /app && python api/app.py
