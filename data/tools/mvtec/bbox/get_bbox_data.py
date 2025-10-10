import json
import logging
import os
from typing import Optional

import numpy as np
from datasets import load_dataset
from PIL import Image
from rich.logging import RichHandler

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger("rich")

# Dataset root path - update this to match your actual path
DATA_ROOT = "/data1/huggingface/hub/datasets--XimiaoZhang--MVTec-2K/snapshots/d52ff40b834d44cfcbea1fafc204666fc0da5b18"


def calculate_bboxes_from_mask(
    mask_path: str,
) -> Optional[list[tuple[int, int, int, int]]]:
    """
    Calculate bounding box coordinates from a mask image.
    Handles multiple disconnected regions and returns separate bboxes for each.

    Args:
        mask_path: Path to the mask image file

    Returns:
        List of tuples (x1, y1, x2, y2) coordinates for each region,
        or None if mask is empty or invalid
    """
    try:
        # Construct full path
        full_mask_path = os.path.join(DATA_ROOT, mask_path)

        if not os.path.exists(full_mask_path):
            log.warning(f"Mask file not found: {full_mask_path}")
            return None

        # Load mask image
        mask_image = Image.open(full_mask_path)

        # Convert to numpy array
        mask_array = np.array(mask_image)

        # Handle different image modes
        if len(mask_array.shape) == 3:
            # Convert to grayscale if it's a color image
            mask_array = np.mean(mask_array, axis=2)

        # Find non-zero pixels (foreground)
        foreground_pixels = np.where(mask_array > 0)

        if len(foreground_pixels[0]) == 0:
            log.warning(f"No foreground pixels found in mask: {mask_path}")
            return None

        # Create binary mask for connected components
        binary_mask = (mask_array > 0).astype(np.uint8)

        # Find connected components
        from scipy import ndimage

        labeled_mask, num_features = ndimage.label(binary_mask)

        bboxes = []

        # Calculate bbox for each connected component
        for label_id in range(1, num_features + 1):
            # Get coordinates of pixels belonging to this component
            component_pixels = np.where(labeled_mask == label_id)

            if len(component_pixels[0]) > 0:
                y_coords, x_coords = component_pixels
                x1 = int(np.min(x_coords))
                y1 = int(np.min(y_coords))
                x2 = int(np.max(x_coords))
                y2 = int(np.max(y_coords))

                bboxes.append((x1, y1, x2, y2))

        log.debug(f"Found {len(bboxes)} bboxes for {mask_path}: {bboxes}")
        return bboxes

    except Exception as e:
        log.error(f"Error calculating bboxes for {mask_path}: {e}")
        return None


def calculate_bbox_from_mask(
    mask_path: str,
) -> Optional[tuple[int, int, int, int]]:
    """
    Calculate a single bounding box that encompasses all mask regions.
    This is kept for backward compatibility.

    Args:
        mask_path: Path to the mask image file

    Returns:
        Tuple of (x1, y1, x2, y2) coordinates, or None if mask is empty or invalid
    """
    bboxes = calculate_bboxes_from_mask(mask_path)

    if bboxes is None or len(bboxes) == 0:
        return None

    if len(bboxes) == 1:
        return bboxes[0]

    # Calculate encompassing bbox for multiple regions
    x1 = min(bbox[0] for bbox in bboxes)
    y1 = min(bbox[1] for bbox in bboxes)
    x2 = max(bbox[2] for bbox in bboxes)
    y2 = max(bbox[3] for bbox in bboxes)

    return (x1, y1, x2, y2)


def load_dataset_with_masks(data_path: str = None, debug: bool = False, sample_size: int = 5):
    """
    Load the dataset and filter for samples that have mask information.

    Args:
        data_path: Path to the JSONL dataset file
        debug: If True, use only a sample of the data
        sample_size: Number of samples to use in debug mode

    Returns:
        Dataset with mask information
    """
    if data_path is None:
        data_path = os.path.join(DATA_ROOT, "test_uni.jsonl")

    log.info(f"Loading dataset from: {data_path}")

    try:
        dataset = load_dataset("json", data_files=data_path, split="train")

        # Filter for samples that have mask information
        dataset_with_masks = dataset.filter(lambda x: "mask" in x and x["mask"] is not None)

        if debug:
            dataset_with_masks = dataset_with_masks.select(range(min(sample_size, len(dataset_with_masks))))

        log.info(f"Dataset loaded. Found {len(dataset_with_masks)} samples with mask information.")
        return dataset_with_masks

    except Exception as e:
        log.error(f"Error loading dataset from {data_path}: {e}")
        raise


def enhance_dataset_with_bboxes(dataset, output_path: str = None, use_multiple_bboxes: bool = True):
    """
    Enhance the dataset by adding bounding box coordinates for each mask.

    Args:
        dataset: The input dataset
        output_path: Path to save the enhanced dataset (optional)
        use_multiple_bboxes: If True, store multiple bboxes per region; if False, use single encompassing bbox

    Returns:
        Enhanced dataset with bounding box information
    """
    log.info("Starting to calculate bounding boxes for all masks...")

    enhanced_data = []
    successful_count = 0
    failed_count = 0

    for i, sample in enumerate(dataset):
        log.info(f"Processing sample {i + 1}/{len(dataset)}: {sample.get('filename', 'unknown')}")

        # Create enhanced sample
        enhanced_sample = sample.copy()

        if use_multiple_bboxes:
            # Calculate multiple bboxes for separate regions
            bboxes = calculate_bboxes_from_mask(sample["mask"])

            if bboxes is not None and len(bboxes) > 0:
                enhanced_sample["bboxes"] = bboxes
                enhanced_sample["bbox_count"] = len(bboxes)
                enhanced_sample["bbox_valid"] = True
                # Also provide a single encompassing bbox for convenience
                if len(bboxes) == 1:
                    enhanced_sample["bbox"] = bboxes[0]
                else:
                    x1 = min(bbox[0] for bbox in bboxes)
                    y1 = min(bbox[1] for bbox in bboxes)
                    x2 = max(bbox[2] for bbox in bboxes)
                    y2 = max(bbox[3] for bbox in bboxes)
                    enhanced_sample["bbox"] = (x1, y1, x2, y2)
                successful_count += 1
            else:
                enhanced_sample["bboxes"] = None
                enhanced_sample["bbox"] = None
                enhanced_sample["bbox_count"] = 0
                enhanced_sample["bbox_valid"] = False
                failed_count += 1
                log.warning(f"Failed to calculate bboxes for: {sample.get('filename', 'unknown')}")
        else:
            # Calculate single encompassing bbox (backward compatibility)
            bbox = calculate_bbox_from_mask(sample["mask"])

            if bbox is not None:
                enhanced_sample["bbox"] = bbox
                enhanced_sample["bbox_valid"] = True
                enhanced_sample["bbox_count"] = 1
                enhanced_sample["bboxes"] = [bbox]
                successful_count += 1
            else:
                enhanced_sample["bbox"] = None
                enhanced_sample["bboxes"] = None
                enhanced_sample["bbox_count"] = 0
                enhanced_sample["bbox_valid"] = False
                failed_count += 1
                log.warning(f"Failed to calculate bbox for: {sample.get('filename', 'unknown')}")

        enhanced_data.append(enhanced_sample)

    log.info(f"Bbox calculation completed. Success: {successful_count}, Failed: {failed_count}")

    # Save enhanced dataset if output path is provided
    if output_path:
        log.info(f"Saving enhanced dataset to: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in enhanced_data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        log.info("Enhanced dataset saved successfully.")

    return enhanced_data


def main():
    """
    Main function to demonstrate the bbox calculation process.
    """
    log.info("Starting bbox calculation for MVTec-2K dataset...")

    # Load dataset
    dataset = load_dataset_with_masks(debug=False)

    # Enhance with bounding boxes
    enhanced_data = enhance_dataset_with_bboxes(dataset, "enhanced_dataset_with_bboxes.jsonl")

    # Display some results
    log.info("\nSample results:")
    for i, sample in enumerate(enhanced_data[:3]):
        log.info(f"Sample {i + 1}:")
        log.info(f"  Filename: {sample.get('filename', 'N/A')}")
        log.info(f"  Mask: {sample.get('mask', 'N/A')}")
        log.info(f"  Bbox: {sample.get('bbox', 'N/A')}")
        log.info(f"  Valid: {sample.get('bbox_valid', 'N/A')}")
        log.info("")


if __name__ == "__main__":
    main()
