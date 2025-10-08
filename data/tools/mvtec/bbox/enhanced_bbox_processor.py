import logging
import os
from typing import Any, Optional

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


def calculate_bbox_area(bbox: tuple[int, int, int, int]) -> int:
    """
    Calculate the area of a bounding box.

    Args:
        bbox: Tuple of (x1, y1, x2, y2) coordinates

    Returns:
        Area of the bounding box
    """
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def get_top_n_bboxes_by_area(bboxes: list[tuple[int, int, int, int]], n: int = 3) -> list[tuple[int, int, int, int]]:
    """
    Get the top N bounding boxes by area.

    Args:
        bboxes: List of bounding box tuples (x1, y1, x2, y2)
        n: Number of top bboxes to return (default: 3)

    Returns:
        List of top N bboxes sorted by area (largest first)
    """
    if not bboxes:
        return []

    # Calculate area for each bbox and sort by area (descending)
    bbox_with_areas = [(bbox, calculate_bbox_area(bbox)) for bbox in bboxes]
    bbox_with_areas.sort(key=lambda x: x[1], reverse=True)

    # Return top N bboxes
    top_bboxes = [bbox for bbox, _ in bbox_with_areas[:n]]

    log.debug(f"Selected top {len(top_bboxes)} bboxes from {len(bboxes)} total bboxes")
    return top_bboxes


def scale_bbox_coordinates(
    bbox: tuple[int, int, int, int], original_size: tuple[int, int], new_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """
    Scale bounding box coordinates based on image size change.

    Args:
        bbox: Original bounding box (x1, y1, x2, y2)
        original_size: Original image size (width, height)
        new_size: New image size (width, height)

    Returns:
        Scaled bounding box coordinates
    """
    x1, y1, x2, y2 = bbox
    orig_width, orig_height = original_size
    new_width, new_height = new_size

    # Calculate scaling factors
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height

    # Scale coordinates
    scaled_x1 = int(x1 * scale_x)
    scaled_y1 = int(y1 * scale_y)
    scaled_x2 = int(x2 * scale_x)
    scaled_y2 = int(y2 * scale_y)

    # Ensure coordinates are within bounds
    scaled_x1 = max(0, min(scaled_x1, new_width))
    scaled_y1 = max(0, min(scaled_y1, new_height))
    scaled_x2 = max(0, min(scaled_x2, new_width))
    scaled_y2 = max(0, min(scaled_y2, new_height))

    return (scaled_x1, scaled_y1, scaled_x2, scaled_y2)


def get_image_size_from_path(image_path: str) -> Optional[tuple[int, int]]:
    """
    Get image size from file path.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (width, height) or None if failed
    """
    try:
        with Image.open(image_path) as img:
            return img.size  # PIL returns (width, height)
    except Exception as e:
        log.error(f"Failed to get image size from {image_path}: {e}")
        return None


def calculate_thumbnail_size(original_size: tuple[int, int], max_size: tuple[int, int]) -> tuple[int, int]:
    """
    Calculate the thumbnail size that maintains aspect ratio.
    This mimics PIL's thumbnail behavior.

    Args:
        original_size: Original image size (width, height)
        max_size: Maximum allowed size (width, height)

    Returns:
        Calculated thumbnail size (width, height)
    """
    orig_width, orig_height = original_size
    max_width, max_height = max_size

    # Calculate scaling factor to fit within max_size while maintaining aspect ratio
    scale_x = max_width / orig_width
    scale_y = max_height / orig_height
    scale = min(scale_x, scale_y)

    # Calculate new size
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)

    return (new_width, new_height)


def process_bboxes_with_image_scaling(
    bboxes: list[tuple[int, int, int, int]],
    original_image_path: str,
    max_image_size: Optional[tuple[int, int]] = None,
    max_bboxes: int = 3,
) -> tuple[list[tuple[int, int, int, int]], dict[str, Any]]:
    """
    Process bounding boxes with image scaling consideration.

    Args:
        bboxes: List of original bounding boxes
        original_image_path: Path to the original image
        max_image_size: Maximum image size for scaling (width, height)
        max_bboxes: Maximum number of bboxes to return (default: 3)

    Returns:
        Tuple of (processed_bboxes, metadata)
    """
    if not bboxes:
        return [], {"original_count": 0, "processed_count": 0, "scaled": False}

    # Get original image size
    original_size = get_image_size_from_path(original_image_path)
    if original_size is None:
        log.warning(f"Could not get image size for {original_image_path}, returning original bboxes")
        top_bboxes = get_top_n_bboxes_by_area(bboxes, max_bboxes)
        return top_bboxes, {
            "original_count": len(bboxes),
            "processed_count": len(top_bboxes),
            "scaled": False,
            "error": "Could not get original image size",
        }

    # First, get top N bboxes by area
    top_bboxes = get_top_n_bboxes_by_area(bboxes, max_bboxes)

    # If no scaling is needed, return the top bboxes
    if max_image_size is None:
        return top_bboxes, {
            "original_count": len(bboxes),
            "processed_count": len(top_bboxes),
            "scaled": False,
            "original_size": original_size,
        }

    # Calculate new image size after scaling
    new_size = calculate_thumbnail_size(original_size, max_image_size)

    # If no actual scaling occurs, return original bboxes
    if new_size == original_size:
        return top_bboxes, {
            "original_count": len(bboxes),
            "processed_count": len(top_bboxes),
            "scaled": False,
            "original_size": original_size,
            "new_size": new_size,
        }

    # Scale the bounding boxes
    scaled_bboxes = []
    for bbox in top_bboxes:
        scaled_bbox = scale_bbox_coordinates(bbox, original_size, new_size)
        scaled_bboxes.append(scaled_bbox)

    metadata = {
        "original_count": len(bboxes),
        "processed_count": len(scaled_bboxes),
        "scaled": True,
        "original_size": original_size,
        "new_size": new_size,
        "scale_factors": (new_size[0] / original_size[0], new_size[1] / original_size[1]),
    }

    log.debug(f"Scaled {len(top_bboxes)} bboxes from {original_size} to {new_size}")
    return scaled_bboxes, metadata


def format_bboxes_for_output(bboxes: list[tuple[int, int, int, int]]) -> list[dict[str, list[int]]]:
    """
    Format bounding boxes for output in the expected format.

    Args:
        bboxes: List of bounding box tuples (x1, y1, x2, y2)

    Returns:
        List of formatted bbox dictionaries
    """
    formatted = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        formatted.append({"bbox_2d": [int(x1), int(y1), int(x2), int(y2)]})
    return formatted


def enhance_item_with_processed_bboxes(
    item: dict[str, Any], dataset_root: str, max_image_size: Optional[tuple[int, int]] = None, max_bboxes: int = 3
) -> dict[str, Any]:
    """
    Enhance a dataset item with processed bounding boxes.

    Args:
        item: Original dataset item
        dataset_root: Root directory of the dataset
        max_image_size: Maximum image size for scaling
        max_bboxes: Maximum number of bboxes to keep

    Returns:
        Enhanced item with processed bboxes
    """
    enhanced_item = item.copy()

    # Get original bboxes
    original_bboxes = item.get("bboxes", [])
    if not original_bboxes:
        # No bboxes to process
        enhanced_item["processed_bboxes"] = []
        enhanced_item["bbox_metadata"] = {"original_count": 0, "processed_count": 0, "scaled": False}
        return enhanced_item

    # Get image path
    filename = item.get("filename")
    if not filename:
        log.warning("No filename found in item, cannot process bboxes")
        enhanced_item["processed_bboxes"] = original_bboxes[:max_bboxes]
        enhanced_item["bbox_metadata"] = {"error": "No filename found"}
        return enhanced_item

    image_path = os.path.join(dataset_root, filename)

    # Process bboxes
    processed_bboxes, metadata = process_bboxes_with_image_scaling(
        original_bboxes, image_path, max_image_size, max_bboxes
    )

    # Update item
    enhanced_item["processed_bboxes"] = processed_bboxes
    enhanced_item["bbox_metadata"] = metadata

    # Also update the original bboxes field for backward compatibility
    enhanced_item["bboxes"] = processed_bboxes
    enhanced_item["bbox_count"] = len(processed_bboxes)
    enhanced_item["bbox_valid"] = len(processed_bboxes) > 0

    return enhanced_item


if __name__ == "__main__":
    # Example usage
    log.info("Enhanced bbox processor module loaded")

    # Test with sample data
    sample_bboxes = [
        (100, 100, 200, 200),  # Area: 10000
        (50, 50, 80, 80),  # Area: 900
        (300, 300, 500, 600),  # Area: 60000 (largest)
        (10, 10, 30, 40),  # Area: 600
        (150, 150, 250, 350),  # Area: 20000 (second largest)
    ]

    top_3 = get_top_n_bboxes_by_area(sample_bboxes, 3)
    log.info(f"Top 3 bboxes by area: {top_3}")

    # Test scaling
    scaled = scale_bbox_coordinates((100, 100, 200, 200), (800, 600), (400, 300))
    log.info(f"Scaled bbox: {scaled}")
