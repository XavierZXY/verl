#!/bin/bash
# Convenience script to compare bbox predictions across different training steps

# Default values
ROLLOUT_DIR="logs/rollout/validation/mvtec"
TEST_DATA="data/mvtec/test/test.jsonl"
OUTPUT_DIR="logs/rollout/comparison"
MVTEC_ROOT="data/mvtec"

# Parse arguments
STEPS=""
MAX_SAMPLES=""
SAMPLE_INDICES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --rollout-dir)
            ROLLOUT_DIR="$2"
            shift 2
            ;;
        --test-data)
            TEST_DATA="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --mvtec-root)
            MVTEC_ROOT="$2"
            shift 2
            ;;
        --steps)
            shift
            while [[ $# -gt 0 ]] && [[ ! $1 =~ ^-- ]]; do
                STEPS="$STEPS $1"
                shift
            done
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --sample-indices)
            shift
            while [[ $# -gt 0 ]] && [[ ! $1 =~ ^-- ]]; do
                SAMPLE_INDICES="$SAMPLE_INDICES $1"
                shift
            done
            ;;
        --help)
            echo "Usage: $0 --steps STEP1 STEP2 ... [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --rollout-dir DIR        Directory containing rollout JSONL files (default: logs/rollout/validation/mvtec)"
            echo "  --test-data FILE         Path to test.jsonl file (default: data/mvtec/test/test.jsonl)"
            echo "  --output-dir DIR         Output directory for comparison images (default: logs/rollout/comparison)"
            echo "  --mvtec-root DIR         Root directory of MVTec dataset (default: data/mvtec)"
            echo "  --steps STEP1 STEP2 ...  Training steps to compare (required)"
            echo "  --max-samples N          Maximum number of samples to compare (optional)"
            echo "  --sample-indices I1 I2.. Specific sample indices to compare (optional)"
            echo ""
            echo "Example:"
            echo "  $0 --steps 0 40 180 --max-samples 10"
            echo "  $0 --steps 0 40 180 --sample-indices 0 5 10 15"
            echo "  $0 --steps 0 40 180 --test-data data/mvtec/test/test.jsonl"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if steps are provided
if [ -z "$STEPS" ]; then
    echo "Error: --steps argument is required"
    echo "Use --help for usage information"
    exit 1
fi

# Build python command
CMD="python scripts/bbox_step_comparator.py --rollout-dir $ROLLOUT_DIR --test-data $TEST_DATA --output-dir $OUTPUT_DIR --mvtec-root $MVTEC_ROOT --steps$STEPS"

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi

if [ -n "$SAMPLE_INDICES" ]; then
    CMD="$CMD --sample-indices$SAMPLE_INDICES"
fi

echo "Running: $CMD"
eval $CMD

