#!/bin/bash
# Example script showing how to use the bbox step comparator

# This example compares steps 0, 40, and 180 for the first 5 samples

echo "Running bbox step comparison example..."
echo "Comparing steps 0, 40, and 180 for the first 5 samples"
echo ""

python scripts/bbox_step_comparator.py \
    --rollout-dir logs/rollout/validation/mvtec \
    --test-data data/mvtec/test/test.jsonl \
    --steps 0 40 180 \
    --output-dir logs/rollout/comparison_example \
    --max-samples 5

echo ""
echo "Done! Check logs/rollout/comparison_example/ for results"

