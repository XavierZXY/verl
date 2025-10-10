#!/usr/bin/env python3
"""
Image Column Replacement Tool for Parquet Files

This script replaces the 'images' column in a source parquet file with
the 'images' column from a test parquet file, while preserving all other columns.
"""

import logging
import os
import sys

import pandas as pd
from rich.logging import RichHandler

# Configure rich logging with English comments and outputs
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("rich")


def replace_image_column(
    source_parquet_path: str, test_parquet_path: str, output_path: str, image_column_name: str = "images"
) -> bool:
    """
    Replace the image column in source parquet with images from test parquet.

    Args:
        source_parquet_path: Path to the source parquet file (60 samples)
        test_parquet_path: Path to the test parquet file containing replacement images
        output_path: Path where the modified parquet file will be saved
        image_column_name: Name of the image column to replace (default: "images")

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read source parquet file
        log.info(f"Reading source parquet file: {source_parquet_path}")
        source_df = pd.read_parquet(source_parquet_path)
        log.info(f"Source data shape: {source_df.shape}")

        # Read test parquet file
        log.info(f"Reading test parquet file: {test_parquet_path}")
        test_df = pd.read_parquet(test_parquet_path)
        log.info(f"Test data shape: {test_df.shape}")

        # Check if image column exists in both files
        if image_column_name not in source_df.columns:
            log.error(f"Image column '{image_column_name}' not found in source file")
            return False

        if image_column_name not in test_df.columns:
            log.error(f"Image column '{image_column_name}' not found in test file")
            return False

        # Check if we have enough images in test file
        source_count = len(source_df)
        test_count = len(test_df)

        if test_count < source_count:
            log.warning(
                f"Test file has {test_count} images but source needs {source_count}. Will cycle through test images."
            )

        # Create a copy of source dataframe
        result_df = source_df.copy()

        # Replace image column with images from test file
        # If test file has fewer images, cycle through them
        test_images = test_df[image_column_name].tolist()

        new_images = []
        for i in range(source_count):
            # Use modulo to cycle through test images if needed
            test_index = i % test_count
            new_images.append(test_images[test_index])

        result_df[image_column_name] = new_images

        log.info(f"Successfully replaced {source_count} images")

        # Save the result
        log.info(f"Saving result to: {output_path}")
        result_df.to_parquet(output_path, index=False)

        log.info("Image replacement completed successfully!")
        log.info(f"Output file: {output_path}")
        log.info(f"Final data shape: {result_df.shape}")

        return True

    except FileNotFoundError as e:
        log.error(f"File not found: {e}")
        return False
    except Exception as e:
        log.error(f"Error during image replacement: {e}")
        return False


def validate_file_paths(source_path: str, test_path: str) -> bool:
    """
    Validate that input files exist and are readable.

    Args:
        source_path: Path to source parquet file
        test_path: Path to test parquet file

    Returns:
        bool: True if both files are valid, False otherwise
    """
    if not os.path.exists(source_path):
        log.error(f"Source file does not exist: {source_path}")
        return False

    if not os.path.exists(test_path):
        log.error(f"Test file does not exist: {test_path}")
        return False

    if not source_path.endswith(".parquet"):
        log.error(f"Source file is not a parquet file: {source_path}")
        return False

    if not test_path.endswith(".parquet"):
        log.error(f"Test file is not a parquet file: {test_path}")
        return False

    return True


def main() -> None:
    """
    Main function to handle command line arguments and execute image replacement.
    """
    if len(sys.argv) < 4:
        print("Usage: python change_image.py <source_parquet> <test_parquet> <output_parquet> [image_column_name]")
        print("Example: python change_image.py data_60_samples.parquet test.parquet output.parquet images")
        sys.exit(1)

    source_parquet = sys.argv[1]
    test_parquet = sys.argv[2]
    output_parquet = sys.argv[3]
    image_column = sys.argv[4] if len(sys.argv) > 4 else "images"

    # Validate input files
    if not validate_file_paths(source_parquet, test_parquet):
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(output_parquet))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Perform image replacement
    success = replace_image_column(
        source_parquet_path=source_parquet,
        test_parquet_path=test_parquet,
        output_path=output_parquet,
        image_column_name=image_column,
    )

    if success:
        log.info("✅ Image replacement completed successfully!")
    else:
        log.error("❌ Image replacement failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
