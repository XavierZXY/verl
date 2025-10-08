#!/usr/bin/env python3
"""
Demo script showing how to use the enhanced bbox processing functionality.
"""

import logging

from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger("demo")


def demo_bbox_area_selection():
    """Demonstrate bbox selection by area."""
    log.info("=== Demo: Bbox Selection by Area ===")

    # Simulate the get_top_n_bboxes_by_area function
    def calculate_bbox_area(bbox):
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def get_top_n_bboxes_by_area(bboxes, n=3):
        if not bboxes:
            return []

        bbox_with_areas = [(bbox, calculate_bbox_area(bbox)) for bbox in bboxes]
        bbox_with_areas.sort(key=lambda x: x[1], reverse=True)
        return [bbox for bbox, _ in bbox_with_areas[:n]]

    # Example from the dataset
    original_bboxes = [
        [494, 578, 635, 699],  # Area: 17061
        [1536, 804, 1639, 921],  # Area: 12051
        [618, 1490, 737, 1581],  # Area: 10829
        [100, 100, 200, 200],  # Area: 10000
        [50, 50, 80, 80],  # Area: 900
    ]

    log.info(f"Original bboxes: {len(original_bboxes)}")
    for i, bbox in enumerate(original_bboxes):
        area = calculate_bbox_area(bbox)
        log.info(f"  {i + 1}. {bbox} -> Area: {area}")

    top_3 = get_top_n_bboxes_by_area(original_bboxes, 3)

    log.info("\nTop 3 bboxes by area:")
    for i, bbox in enumerate(top_3):
        area = calculate_bbox_area(bbox)
        log.info(f"  {i + 1}. {bbox} -> Area: {area}")


def demo_bbox_scaling():
    """Demonstrate bbox coordinate scaling."""
    log.info("\n=== Demo: Bbox Coordinate Scaling ===")

    def scale_bbox_coordinates(bbox, original_size, new_size):
        x1, y1, x2, y2 = bbox
        orig_width, orig_height = original_size
        new_width, new_height = new_size

        scale_x = new_width / orig_width
        scale_y = new_height / orig_height

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

    # Example: scaling from 2048x1536 to 1024x768 (0.5x scale)
    original_bbox = [494, 578, 635, 699]
    original_size = (2048, 1536)
    new_size = (1024, 768)

    log.info(f"Original bbox: {original_bbox}")
    log.info(f"Original image size: {original_size}")
    log.info(f"New image size: {new_size}")

    scaled_bbox = scale_bbox_coordinates(original_bbox, original_size, new_size)

    log.info(f"Scaled bbox: {list(scaled_bbox)}")

    # Calculate scale factors
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]
    log.info(f"Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")


def demo_format_for_output():
    """Demonstrate bbox formatting for output."""
    log.info("\n=== Demo: Bbox Formatting for Output ===")

    def format_bboxes_for_output(bboxes):
        formatted = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            formatted.append({"bbox_2d": [int(x1), int(y1), int(x2), int(y2)]})
        return formatted

    sample_bboxes = [[494, 578, 635, 699], [1536, 804, 1639, 921], [618, 1490, 737, 1581]]

    log.info(f"Input bboxes: {sample_bboxes}")

    formatted = format_bboxes_for_output(sample_bboxes)

    log.info("Formatted for output:")
    for i, bbox_dict in enumerate(formatted):
        log.info(f"  {i + 1}. {bbox_dict}")


def demo_complete_workflow():
    """Demonstrate the complete workflow."""
    log.info("\n=== Demo: Complete Workflow ===")

    # Simulate a dataset item
    sample_item = {
        "filename": "grid/test/bent/000.png",
        "clsname": "grid",
        "label": 1,
        "label_name": "bent",
        "bboxes": [
            [494, 578, 635, 699],  # Area: 17061
            [1536, 804, 1639, 921],  # Area: 12051
            [618, 1490, 737, 1581],  # Area: 10829
            [100, 100, 200, 200],  # Area: 10000
            [50, 50, 80, 80],  # Area: 900
            [300, 300, 500, 600],  # Area: 60000 (largest)
        ],
    }

    log.info(f"Original item has {len(sample_item['bboxes'])} bboxes")

    # Step 1: Select top 3 by area
    def calculate_bbox_area(bbox):
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def get_top_n_bboxes_by_area(bboxes, n=3):
        if not bboxes:
            return []
        bbox_with_areas = [(bbox, calculate_bbox_area(bbox)) for bbox in bboxes]
        bbox_with_areas.sort(key=lambda x: x[1], reverse=True)
        return [bbox for bbox, _ in bbox_with_areas[:n]]

    top_3_bboxes = get_top_n_bboxes_by_area(sample_item["bboxes"], 3)
    log.info(f"Selected top 3 bboxes: {top_3_bboxes}")

    # Step 2: Scale if needed (simulate image resize from 2048x1536 to 1024x768)
    def scale_bbox_coordinates(bbox, original_size, new_size):
        x1, y1, x2, y2 = bbox
        orig_width, orig_height = original_size
        new_width, new_height = new_size

        scale_x = new_width / orig_width
        scale_y = new_height / orig_height

        return (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))

    original_size = (2048, 1536)
    new_size = (1024, 768)

    scaled_bboxes = []
    for bbox in top_3_bboxes:
        scaled_bbox = scale_bbox_coordinates(bbox, original_size, new_size)
        scaled_bboxes.append(scaled_bbox)

    log.info(f"Scaled bboxes: {scaled_bboxes}")

    # Step 3: Format for output
    def format_bboxes_for_output(bboxes):
        formatted = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            formatted.append({"bbox_2d": [int(x1), int(y1), int(x2), int(y2)]})
        return formatted

    formatted_bboxes = format_bboxes_for_output(scaled_bboxes)
    log.info(f"Final formatted bboxes: {formatted_bboxes}")

    # Show the metadata that would be included
    metadata = {
        "original_count": len(sample_item["bboxes"]),
        "processed_count": len(formatted_bboxes),
        "scaled": True,
        "original_size": original_size,
        "new_size": new_size,
        "scale_factors": (new_size[0] / original_size[0], new_size[1] / original_size[1]),
    }

    log.info(f"Metadata: {metadata}")


def main():
    """Run all demos."""
    log.info("Enhanced Bbox Processing Demo")
    log.info("=" * 50)

    demo_bbox_area_selection()
    demo_bbox_scaling()
    demo_format_for_output()
    demo_complete_workflow()

    log.info("=" * 50)
    log.info("Demo completed! 🎉")

    log.info("\nKey improvements:")
    log.info("✓ Only keeps the 3 largest bboxes by area")
    log.info("✓ Automatically scales bbox coordinates when image is resized")
    log.info("✓ Provides metadata about the processing")
    log.info("✓ Maintains compatibility with existing format")


if __name__ == "__main__":
    main()
