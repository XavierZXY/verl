#!/usr/bin/env python3
"""
Test script for enhanced bbox processing functionality.
"""

import logging
import os
import sys

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bbox.enhanced_bbox_processor import (
    calculate_thumbnail_size,
    enhance_item_with_processed_bboxes,
    format_bboxes_for_output,
    get_top_n_bboxes_by_area,
    scale_bbox_coordinates,
)
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger("test")


def test_bbox_area_selection():
    """Test bbox selection by area."""
    log.info("Testing bbox selection by area...")

    # Test data with different areas
    test_bboxes = [
        (100, 100, 200, 200),  # Area: 10000
        (50, 50, 80, 80),  # Area: 900
        (300, 300, 500, 600),  # Area: 60000 (largest)
        (10, 10, 30, 40),  # Area: 600
        (150, 150, 250, 350),  # Area: 20000 (second largest)
        (400, 400, 450, 500),  # Area: 5000 (third largest)
    ]

    top_3 = get_top_n_bboxes_by_area(test_bboxes, 3)

    expected_order = [
        (300, 300, 500, 600),  # Area: 60000
        (150, 150, 250, 350),  # Area: 20000
        (100, 100, 200, 200),  # Area: 10000
    ]

    assert top_3 == expected_order, f"Expected {expected_order}, got {top_3}"
    log.info("✓ Bbox area selection test passed")


def test_bbox_scaling():
    """Test bbox coordinate scaling."""
    log.info("Testing bbox coordinate scaling...")

    # Test scaling from 800x600 to 400x300 (0.5x scale)
    original_bbox = (100, 100, 200, 200)
    original_size = (800, 600)
    new_size = (400, 300)

    scaled_bbox = scale_bbox_coordinates(original_bbox, original_size, new_size)
    expected_bbox = (50, 50, 100, 100)

    assert scaled_bbox == expected_bbox, f"Expected {expected_bbox}, got {scaled_bbox}"
    log.info("✓ Bbox scaling test passed")


def test_thumbnail_size_calculation():
    """Test thumbnail size calculation."""
    log.info("Testing thumbnail size calculation...")

    # Test case 1: Landscape image
    original_size = (1600, 1200)
    max_size = (800, 600)
    result = calculate_thumbnail_size(original_size, max_size)
    expected = (800, 600)  # Exact fit
    assert result == expected, f"Expected {expected}, got {result}"

    # Test case 2: Portrait image that needs scaling
    original_size = (800, 1200)
    max_size = (400, 400)
    result = calculate_thumbnail_size(original_size, max_size)
    expected = (266, 400)  # Height limited, width scaled proportionally
    assert result == expected, f"Expected {expected}, got {result}"

    log.info("✓ Thumbnail size calculation test passed")


def test_format_bboxes():
    """Test bbox formatting for output."""
    log.info("Testing bbox formatting...")

    test_bboxes = [(100, 150, 200, 250), (300, 350, 400, 450)]

    formatted = format_bboxes_for_output(test_bboxes)
    expected = [{"bbox_2d": [100, 150, 200, 250]}, {"bbox_2d": [300, 350, 400, 450]}]

    assert formatted == expected, f"Expected {expected}, got {formatted}"
    log.info("✓ Bbox formatting test passed")


def test_with_sample_data():
    """Test with sample data from the dataset."""
    log.info("Testing with sample dataset item...")

    # Create a sample item similar to the dataset
    sample_item = {
        "filename": "grid/test/bent/000.png",
        "clsname": "grid",
        "label": 1,
        "label_name": "bent",
        "bboxes": [
            [494, 578, 635, 699],  # Area: 17061
            [1536, 804, 1639, 921],  # Area: 12051
            [618, 1490, 737, 1581],  # Area: 10829
            [100, 100, 150, 150],  # Area: 2500 (smallest)
            [200, 200, 400, 500],  # Area: 60000 (largest)
        ],
    }

    # Test without image scaling (just area selection)
    enhanced_item = enhance_item_with_processed_bboxes(
        sample_item,
        "/fake/dataset/root",  # This won't be used since no scaling
        max_image_size=None,
        max_bboxes=3,
    )

    # Should select top 3 by area
    expected_top_3 = [
        [200, 200, 400, 500],  # Area: 60000
        [494, 578, 635, 699],  # Area: 17061
        [1536, 804, 1639, 921],  # Area: 12051
    ]

    assert enhanced_item["processed_bboxes"] == expected_top_3
    assert enhanced_item["bbox_count"] == 3
    assert enhanced_item["bbox_valid"]

    log.info("✓ Sample data test passed")


def run_all_tests():
    """Run all tests."""
    log.info("Starting enhanced bbox processor tests...")

    try:
        test_bbox_area_selection()
        test_bbox_scaling()
        test_thumbnail_size_calculation()
        test_format_bboxes()
        test_with_sample_data()

        log.info("🎉 All tests passed successfully!")
        return True

    except Exception as e:
        log.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
