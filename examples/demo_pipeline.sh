#!/bin/bash
# Complete Preprocessing Pipeline Demo
# This script demonstrates the entire preprocessing workflow

set -e  # Exit on error

echo "=================================================="
echo "Log Preprocessing Pipeline Demo"
echo "=================================================="
echo ""

# Step 1: Generate toy log data
echo "Step 1: Generating toy log data..."
python examples/toy_log_generator.py \
    --num-events 1000 \
    --output-dir data \
    --seed 42

echo ""
echo "Step 2: Preprocessing logs to numerical features..."
python examples/preprocess_logs.py \
    --input data/toy_logs.csv \
    --output data/demo_features.npz \
    --save-preprocessor data/demo_preprocessor.json \
    --sequence-length 10 \
    --stride 5

echo ""
echo "Step 3: Loading and inspecting features..."
python examples/load_features.py \
    --features data/demo_features.npz \
    --preprocessor data/demo_preprocessor.json \
    | head -n 50

echo ""
echo "Step 4: Training a simple classifier..."
python examples/train_classifier.py \
    --input data/toy_logs.csv \
    --test-size 0.3

echo ""
echo "=================================================="
echo "Demo Complete!"
echo "=================================================="
echo ""
echo "Generated files:"
echo "  - data/toy_logs.csv (raw logs)"
echo "  - data/demo_features.npz (preprocessed features)"
echo "  - data/demo_preprocessor.json (preprocessor state)"
echo ""
echo "You can now use these features for:"
echo "  - Training deep learning models"
echo "  - Anomaly detection"
echo "  - Similarity search"
echo "  - Clustering analysis"
echo ""
