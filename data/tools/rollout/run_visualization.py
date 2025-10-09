#!/usr/bin/env python3
"""
Example script to run bbox visualization on the provided data.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from bbox_visualizer import BboxVisualizer


def main():
    """Run bbox visualization on the provided data."""

    # Define paths
    test_data_path = "/home/zxy/codes/working/RL/verl/data/mvtec/test/test.jsonl"
    rollout_data_path = "/home/zxy/codes/working/RL/verl/logs/rollout/validation/mvtec/40.jsonl"
    output_dir = "/home/zxy/codes/working/RL/verl/logs/rollout/validation/mvtec/img/40"
    mvtec_root = (
        "/data1/huggingface/hub/datasets--XimiaoZhang--MVTec-2K/snapshots/d52ff40b834d44cfcbea1fafc204666fc0da5b18"
    )

    # Check if input files exist
    if not os.path.exists(test_data_path):
        print(f"Error: Test data file not found: {test_data_path}")
        return

    if not os.path.exists(rollout_data_path):
        print(f"Error: Rollout data file not found: {rollout_data_path}")
        return

    print("Starting bbox visualization...")
    print(f"Test data: {test_data_path}")
    print(f"Rollout data: {rollout_data_path}")
    print(f"Output directory: {output_dir}")
    print(f"MVTec root: {mvtec_root}")
    print()

    # Create visualizer
    visualizer = BboxVisualizer(mvtec_data_root=mvtec_root)

    # Run visualization (limit to first 10 images for testing)
    try:
        visualizer.visualize_rollout_results(
            test_jsonl_path=test_data_path,
            rollout_jsonl_path=rollout_data_path,
            output_dir=output_dir,
            # max_images=10,  # Limit for testing
        )
        print("Visualization completed successfully!")

    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
