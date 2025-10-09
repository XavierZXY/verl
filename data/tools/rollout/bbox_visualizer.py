#!/usr/bin/env python3
"""
Bbox Visualization Tool for Rollout Results

This tool visualizes bounding boxes from rollout results on original images.
It handles coordinate transformation from 1024x1024 resized images back to original image dimensions.
"""

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np


class BboxVisualizer:
    """
    A tool to visualize bounding boxes from rollout results on original images.

    Handles coordinate transformation from 1024x1024 resized images to original dimensions.
    """

    def __init__(
        self,
        mvtec_data_root: str = "/home/zxy/codes/working/RL/verl/data/mvtec",
    ):
        """
        Initialize the bbox visualizer.

        Args:
            mvtec_data_root: Root directory of the MVTec dataset
        """
        self.mvtec_data_root = Path(mvtec_data_root)
        self.target_size = (1024, 1024)  # Size used during rollout

    def load_test_data(self, test_jsonl_path: str) -> dict[str, dict]:
        """
        Load test data from JSONL file.

        Args:
            test_jsonl_path: Path to the test.jsonl file

        Returns:
            Dictionary mapping filename to test data
        """
        test_data = {}
        with open(test_jsonl_path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                test_data[data["filename"]] = data
        return test_data

    def load_rollout_results(self, rollout_jsonl_path: str) -> list[dict]:
        """
        Load rollout results from JSONL file.

        Args:
            rollout_jsonl_path: Path to the rollout results JSONL file

        Returns:
            List of rollout result dictionaries
        """
        rollout_results = []
        with open(rollout_jsonl_path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                rollout_results.append(data)
        return rollout_results

    def extract_bbox_from_output(self, output: str) -> list[list[int]]:
        """
        Extract bounding box coordinates from model output.

        Args:
            output: Model output string containing bbox information

        Returns:
            List of bounding boxes in [x_min, y_min, x_max, y_max] format
        """
        bboxes = []

        # Find location tags in the output
        location_pattern = r"<location>(.*?)</location>"
        location_match = re.search(location_pattern, output, re.DOTALL)

        if location_match:
            location_content = location_match.group(1).strip()

            # Skip empty location content
            if location_content == "[]" or not location_content:
                return bboxes

            try:
                # Parse JSON content
                location_data = json.loads(location_content)

                # Extract bbox2d coordinates
                for item in location_data:
                    if isinstance(item, dict) and "bbox2d" in item:
                        bbox = item["bbox2d"]
                        if len(bbox) == 4:
                            bboxes.append(bbox)

            except json.JSONDecodeError:
                # Try to extract coordinates using regex if JSON parsing fails
                bbox_pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
                matches = re.findall(bbox_pattern, location_content)
                for match in matches:
                    bbox = [int(x) for x in match]
                    bboxes.append(bbox)

        return bboxes

    def transform_coordinates(self, bbox: list[int], original_size: tuple[int, int]) -> list[int]:
        """
        Transform bbox coordinates from 1024x1024 to original image size.

        Args:
            bbox: Bounding box in [x_min, y_min, x_max, y_max] format (1024x1024 coordinates)
            original_size: Original image size (width, height)

        Returns:
            Transformed bounding box coordinates
        """
        orig_width, orig_height = original_size
        target_width, target_height = self.target_size

        # Calculate scaling factors
        scale_x = orig_width / target_width
        scale_y = orig_height / target_height

        # Transform coordinates
        x_min, y_min, x_max, y_max = bbox

        transformed_bbox = [
            int(x_min * scale_x),
            int(y_min * scale_y),
            int(x_max * scale_x),
            int(y_max * scale_y),
        ]

        return transformed_bbox

    def draw_bboxes_on_image(
        self,
        image_path: str,
        predicted_bboxes: list[list[int]],
        ground_truth_bboxes: list[list[int]] = None,
        output_path: str = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes on image.

        Args:
            image_path: Path to the original image
            predicted_bboxes: List of predicted bounding boxes
            ground_truth_bboxes: List of ground truth bounding boxes (optional)
            output_path: Path to save the output image (optional)

        Returns:
            Image array with drawn bounding boxes
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Get original image size
        height, width = image.shape[:2]
        original_size = (width, height)

        # Transform predicted bboxes to original coordinates
        transformed_predicted = []
        for bbox in predicted_bboxes:
            transformed_bbox = self.transform_coordinates(bbox, original_size)
            transformed_predicted.append(transformed_bbox)

        # Draw predicted bboxes in red
        for bbox in transformed_predicted:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red
            cv2.putText(
                image,
                "Predicted",
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        # Draw ground truth bboxes in green (if provided)
        if ground_truth_bboxes:
            for bbox in ground_truth_bboxes:
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green
                cv2.putText(
                    image,
                    "Ground Truth",
                    (x_min, y_min - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        # Save output image if path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, image)

        return image

    def visualize_rollout_results(
        self,
        test_jsonl_path: str,
        rollout_jsonl_path: str,
        output_dir: str,
        max_images: int = None,
    ):
        """
        Visualize rollout results by drawing bboxes on original images.

        Args:
            test_jsonl_path: Path to the test.jsonl file
            rollout_jsonl_path: Path to the rollout results JSONL file
            output_dir: Directory to save visualization results
            max_images: Maximum number of images to process (None for all)
        """
        # Load data
        test_data = self.load_test_data(test_jsonl_path)
        rollout_results = self.load_rollout_results(rollout_jsonl_path)

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build a mapping from test data index to rollout results
        rollout_map = {}
        for i, result in enumerate(rollout_results):
            rollout_map[i] = result

        # Process each test data entry that has bboxes
        processed_count = 0
        test_entries = list(test_data.values())

        for i, test_entry in enumerate(test_entries):
            if max_images and processed_count >= max_images:
                break

            # Get ground truth bboxes
            ground_truth_bboxes = test_entry.get("bboxes", [])

            # Skip if no ground truth bboxes
            if not ground_truth_bboxes:
                continue

            filename = test_entry["filename"]

            # Extract predicted bboxes from rollout if available
            predicted_bboxes = []
            if i in rollout_map:
                predicted_bboxes = self.extract_bbox_from_output(rollout_map[i]["output"])

            # Construct image path
            image_path = self.mvtec_data_root / filename

            if image_path.exists():
                # Create output filename
                output_filename = f"{i:03d}_{Path(filename).stem}_visualization.jpg"
                output_path = output_dir / output_filename

                try:
                    # Draw bboxes and save
                    self.draw_bboxes_on_image(
                        str(image_path),
                        predicted_bboxes,
                        ground_truth_bboxes,
                        str(output_path),
                    )

                    status = "with predictions" if predicted_bboxes else "ground truth only"
                    print(f"Processed {processed_count + 1}: {filename} ({status})")
                    print(f"  Predicted bboxes: {len(predicted_bboxes)}")
                    print(f"  Ground truth bboxes: {len(ground_truth_bboxes)}")
                    print(f"  Saved to: {output_path}")
                    print()

                    processed_count += 1

                except Exception as e:
                    print(f"Error processing {filename}: {e}")
            else:
                print(f"Image not found: {image_path}")

        print(f"Visualization complete! Processed {processed_count} images.")
        print(f"Results saved to: {output_dir}")


def main():
    """Main function to run the bbox visualizer."""
    parser = argparse.ArgumentParser(description="Visualize bounding boxes from rollout results")
    parser.add_argument("--test-data", required=True, help="Path to test.jsonl file")
    parser.add_argument(
        "--rollout-results",
        required=True,
        help="Path to rollout results JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for visualization results",
    )
    parser.add_argument(
        "--mvtec-root",
        default="/home/zxy/codes/working/RL/verl/data/mvtec",
        help="Root directory of MVTec dataset",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process",
    )

    args = parser.parse_args()

    # Create visualizer
    visualizer = BboxVisualizer(mvtec_data_root=args.mvtec_root)

    # Run visualization
    visualizer.visualize_rollout_results(
        test_jsonl_path=args.test_data,
        rollout_jsonl_path=args.rollout_results,
        output_dir=args.output_dir,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
