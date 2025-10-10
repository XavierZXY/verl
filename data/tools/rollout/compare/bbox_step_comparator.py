#!/usr/bin/env python3
"""
Bbox Step Comparator Tool

This tool compares bounding box predictions across different training steps.
It generates side-by-side visualizations showing how bbox predictions evolve during training.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class BboxStepComparator:
    """
    A tool to compare bounding box predictions across different training steps.
    """

    def __init__(
        self,
        mvtec_data_root: str = "/home/zxy/codes/working/RL/verl/data/mvtec",
        target_size: tuple[int, int] = (1024, 1024),
    ):
        """
        Initialize the bbox step comparator.

        Args:
            mvtec_data_root: Root directory of the MVTec dataset
            target_size: Size used during rollout (default: 1024x1024)
        """
        self.mvtec_data_root = Path(mvtec_data_root)
        self.target_size = target_size

    def load_rollout_results(self, rollout_jsonl_path: str) -> list[dict]:
        """Load rollout results from JSONL file."""
        rollout_results = []
        with open(rollout_jsonl_path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                rollout_results.append(data)
        return rollout_results

    def extract_bbox_from_output(self, output: str) -> list[list[int]]:
        """Extract bounding box coordinates from model output."""
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

    def extract_answer_from_output(self, output: str) -> str:
        """Extract answer (yes/no) from model output."""
        answer_pattern = r"<answer>(.*?)</answer>"
        answer_match = re.search(answer_pattern, output, re.DOTALL | re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip().lower()
        return "unknown"

    def transform_coordinates(self, bbox: list[int], original_size: tuple[int, int]) -> list[int]:
        """Transform bbox coordinates from target size to original image size."""
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

    def draw_single_step_bbox(
        self,
        image: np.ndarray,
        predicted_bboxes: list[list[int]],
        ground_truth_bboxes: list[list[int]],
        step: int,
        score: float,
        answer: str,
        original_size: tuple[int, int],
    ) -> np.ndarray:
        """
        Draw bounding boxes for a single step.

        Args:
            image: Original image
            predicted_bboxes: List of predicted bounding boxes (in 1024x1024)
            ground_truth_bboxes: List of ground truth bounding boxes (in original size)
            step: Training step number
            score: Model score
            answer: Model answer (yes/no)
            original_size: Original image size (width, height)

        Returns:
            Image with drawn bounding boxes
        """
        img_copy = image.copy()

        # Draw ground truth bboxes in green
        for bbox in ground_truth_bboxes:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Transform and draw predicted bboxes in red
        for bbox in predicted_bboxes:
            transformed_bbox = self.transform_coordinates(bbox, original_size)
            x_min, y_min, x_max, y_max = transformed_bbox
            cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        # Add text info at the top
        info_text = f"Step {step} | Score: {score:.3f} | Answer: {answer}"
        cv2.rectangle(img_copy, (0, 0), (img_copy.shape[1], 40), (255, 255, 255), -1)
        cv2.putText(
            img_copy,
            info_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

        # Add legend at bottom
        legend_y = img_copy.shape[0] - 60
        cv2.rectangle(
            img_copy,
            (0, legend_y),
            (img_copy.shape[1], img_copy.shape[0]),
            (255, 255, 255),
            -1,
        )
        cv2.rectangle(img_copy, (10, legend_y + 10), (30, legend_y + 30), (0, 255, 0), 2)
        cv2.putText(
            img_copy,
            "Ground Truth",
            (40, legend_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        cv2.rectangle(img_copy, (200, legend_y + 10), (220, legend_y + 30), (0, 0, 255), 2)
        cv2.putText(
            img_copy,
            "Predicted",
            (230, legend_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

        return img_copy

    def load_test_data(self, test_jsonl_path: str) -> list[dict]:
        """Load test data from JSONL file."""
        test_data = []
        with open(test_jsonl_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    test_data.append(data)
        return test_data

    def compare_steps(
        self,
        rollout_paths: dict[int, str],
        test_data_path: str,
        output_dir: str,
        max_samples: Optional[int] = None,
        sample_indices: Optional[list[int]] = None,
    ):
        """
        Compare bbox predictions across multiple training steps.

        Args:
            rollout_paths: Dictionary mapping step number to rollout JSONL path
            test_data_path: Path to test.jsonl file (required)
            output_dir: Directory to save comparison images
            max_samples: Maximum number of samples to compare (None for all)
            sample_indices: Specific sample indices to compare (None for all)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load test data
        print(f"Loading test data from {test_data_path}")
        test_data = self.load_test_data(test_data_path)
        print(f"Loaded {len(test_data)} test samples")

        # Load rollout results for all steps
        steps_data = {}
        for step, path in rollout_paths.items():
            steps_data[step] = self.load_rollout_results(path)
            print(f"Loaded step {step}: {len(steps_data[step])} samples")

        # Get sorted steps
        sorted_steps = sorted(steps_data.keys())

        # Determine number of samples to process
        num_samples = len(test_data)
        if sample_indices:
            sample_indices = [i for i in sample_indices if 0 <= i < num_samples]
        else:
            sample_indices = list(range(num_samples))
            if max_samples:
                sample_indices = sample_indices[:max_samples]

        print(f"\nComparing {len(sample_indices)} samples across {len(sorted_steps)} steps")

        # Process each sample
        for sample_idx in sample_indices:
            try:
                # Get test data for this sample
                test_sample = test_data[sample_idx]

                # Get image filename from test data
                image_filename = test_sample.get("filename")
                if not image_filename:
                    print(f"Warning: Could not find filename in test data for sample {sample_idx}, skipping")
                    continue

                image_path = self.mvtec_data_root / image_filename
                if not image_path.exists():
                    print(f"Warning: Image not found: {image_path}, skipping")
                    continue

                # Load original image
                original_image = cv2.imread(str(image_path))
                if original_image is None:
                    print(f"Warning: Could not load image: {image_path}, skipping")
                    continue

                height, width = original_image.shape[:2]
                original_size = (width, height)

                # Get ground truth bboxes from test data
                gt_bboxes = test_sample.get("bboxes", [])

                # Create comparison images for each step
                step_images = []
                for step in sorted_steps:
                    step_data = steps_data[step][sample_idx]

                    # Extract bbox predictions
                    predicted_bboxes = self.extract_bbox_from_output(step_data.get("output", ""))
                    score = step_data.get("score", 0.0)
                    answer = self.extract_answer_from_output(step_data.get("output", ""))

                    # Draw bboxes on image
                    step_image = self.draw_single_step_bbox(
                        original_image,
                        predicted_bboxes,
                        gt_bboxes,
                        step,
                        score,
                        answer,
                        original_size,
                    )
                    step_images.append(step_image)

                # Combine images side by side
                combined_image = self._combine_images_horizontal(step_images)

                # Save combined image
                output_filename = f"sample_{sample_idx:03d}_comparison.jpg"
                output_path = output_dir / output_filename
                cv2.imwrite(str(output_path), combined_image)

                print(f"Processed sample {sample_idx}: {image_filename}")
                print(f"  Saved to: {output_path}")

            except Exception as e:
                print(f"Error processing sample {sample_idx}: {e}")
                import traceback

                traceback.print_exc()

        print(f"\nComparison complete! Results saved to: {output_dir}")

    def _combine_images_horizontal(self, images: list[np.ndarray]) -> np.ndarray:
        """Combine multiple images horizontally with equal spacing."""
        if not images:
            return None

        # Get max height
        max_height = max(img.shape[0] for img in images)

        # Resize all images to same height while maintaining aspect ratio
        resized_images = []
        for img in images:
            if img.shape[0] != max_height:
                scale = max_height / img.shape[0]
                new_width = int(img.shape[1] * scale)
                resized_img = cv2.resize(img, (new_width, max_height))
                resized_images.append(resized_img)
            else:
                resized_images.append(img)

        # Add spacing between images
        spacing = 10
        white_bar = np.ones((max_height, spacing, 3), dtype=np.uint8) * 255

        # Combine images with spacing
        combined_parts = []
        for i, img in enumerate(resized_images):
            combined_parts.append(img)
            if i < len(resized_images) - 1:
                combined_parts.append(white_bar)

        combined_image = np.hstack(combined_parts)
        return combined_image


def main():
    """Main function to run the bbox step comparator."""
    parser = argparse.ArgumentParser(description="Compare bounding box predictions across different training steps")
    parser.add_argument(
        "--rollout-dir",
        type=str,
        required=True,
        help="Directory containing rollout JSONL files (e.g., logs/rollout/validation/mvtec/)",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test.jsonl file (e.g., data/mvtec/test/test.jsonl)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        required=True,
        help="Training steps to compare (e.g., 0 40 180)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for comparison images",
    )
    parser.add_argument(
        "--mvtec-root",
        type=str,
        default="/data1/huggingface/hub/datasets--XimiaoZhang--MVTec-2K/snapshots/d52ff40b834d44cfcbea1fafc204666fc0da5b18",
        help="Root directory of MVTec dataset",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to compare",
    )
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific sample indices to compare (e.g., 0 5 10)",
    )

    args = parser.parse_args()

    # Build rollout paths dictionary
    rollout_dir = Path(args.rollout_dir)
    rollout_paths = {}
    for step in args.steps:
        rollout_path = rollout_dir / f"{step}.jsonl"
        if not rollout_path.exists():
            print(f"Warning: Rollout file not found: {rollout_path}")
            continue
        rollout_paths[step] = str(rollout_path)

    if not rollout_paths:
        print("Error: No valid rollout files found")
        return

    # Create comparator
    comparator = BboxStepComparator(mvtec_data_root=args.mvtec_root)

    # Run comparison
    comparator.compare_steps(
        rollout_paths=rollout_paths,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        sample_indices=args.sample_indices,
    )


if __name__ == "__main__":
    main()
