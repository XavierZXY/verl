#!/usr/bin/env python3
"""
Convert MVTec dataset to JSONL format.
"""

import json
import os
from pathlib import Path
from typing import Any


def process_mvtec_test_dataset(dataset_root: str, output_file: str):
    """
    Process MVTec test dataset and generate JSONL file.

    Args:
        dataset_root: Root directory of MVTec dataset
        output_file: Output JSONL file path
    """
    dataset_path = Path(dataset_root)

    # Get all class directories (skip files and special directories)
    skip_items = {".", "..", "download.sh", "extract.sh", "text"}
    class_dirs = []
    for item in dataset_path.iterdir():
        if item.is_dir() and item.name not in skip_items and not item.name.endswith(".tar.xz"):
            class_dirs.append(item)

    class_dirs = sorted(class_dirs)

    results = []

    for class_dir in class_dirs:
        clsname = class_dir.name
        test_dir = class_dir / "test"
        ground_truth_dir = class_dir / "ground_truth"

        if not test_dir.exists():
            print(f"Warning: {test_dir} does not exist, skipping...")
            continue

        print(f"Processing class: {clsname}")

        # Iterate through all subdirectories in test
        label_dirs = sorted(test_dir.iterdir())
        for label_dir in label_dirs:
            if not label_dir.is_dir():
                continue

            label_name = label_dir.name
            is_good = label_name == "good"
            label = 0 if is_good else 1

            # Process all images in this label directory
            image_files = sorted(label_dir.glob("*.png"))

            for image_file in image_files:
                image_name = image_file.name
                # Relative path from dataset root
                filename = f"{clsname}/test/{label_name}/{image_name}"

                entry: dict[str, Any] = {
                    "filename": filename,
                    "clsname": clsname,
                    "label": label,
                    "label_name": label_name,
                }

                # If not good, check for mask
                if not is_good:
                    mask_filename = image_name.replace(".png", "_mask.png")
                    mask_path = ground_truth_dir / label_name / mask_filename

                    if mask_path.exists():
                        # Add mask path
                        entry["mask"] = f"{clsname}/ground_truth/{label_name}/{mask_filename}"
                    else:
                        print(f"Warning: Mask not found for {filename}")

                results.append(entry)

    # Write to JSONL file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    print(f"\nProcessed {len(results)} test images")
    print(f"Output written to: {output_file}")


def process_mvtec_train_dataset(dataset_root: str, output_file: str):
    """
    Process MVTec train dataset and generate JSONL file.

    Args:
        dataset_root: Root directory of MVTec dataset
        output_file: Output JSONL file path
    """
    dataset_path = Path(dataset_root)

    # Get all class directories (skip files and special directories)
    skip_items = {".", "..", "download.sh", "extract.sh", "text"}
    class_dirs = []
    for item in dataset_path.iterdir():
        if item.is_dir() and item.name not in skip_items and not item.name.endswith(".tar.xz"):
            class_dirs.append(item)

    class_dirs = sorted(class_dirs)

    results = []

    for class_dir in class_dirs:
        clsname = class_dir.name
        train_dir = class_dir / "train"

        if not train_dir.exists():
            print(f"Warning: {train_dir} does not exist, skipping...")
            continue

        print(f"Processing train class: {clsname}")

        # Iterate through all subdirectories in train (usually just 'good')
        label_dirs = sorted(train_dir.iterdir())
        for label_dir in label_dirs:
            if not label_dir.is_dir():
                continue

            label_name = label_dir.name
            # Train data is typically all 'good' samples
            label = 0

            # Process all images in this label directory
            image_files = sorted(label_dir.glob("*.png"))

            for image_file in image_files:
                image_name = image_file.name
                # Relative path from dataset root
                filename = f"{clsname}/train/{label_name}/{image_name}"

                entry: dict[str, Any] = {
                    "filename": filename,
                    "clsname": clsname,
                    "label": label,
                    "label_name": label_name,
                }

                results.append(entry)

    # Write to JSONL file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    print(f"\nProcessed {len(results)} train images")
    print(f"Output written to: {output_file}")


def main():
    dataset_root = "/data1/dataset/MVTec"

    # 获取当前文件的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")

    # 处理 test 数据
    test_output_file = os.path.join(output_dir, "mvtec_test.jsonl")
    print("=" * 50)
    print("Processing TEST dataset...")
    print("=" * 50)
    process_mvtec_test_dataset(dataset_root, test_output_file)

    # 处理 train 数据
    train_output_file = os.path.join(output_dir, "mvtec_train.jsonl")
    print("\n" + "=" * 50)
    print("Processing TRAIN dataset...")
    print("=" * 50)
    process_mvtec_train_dataset(dataset_root, train_output_file)

    print("\n" + "=" * 50)
    print("All done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
