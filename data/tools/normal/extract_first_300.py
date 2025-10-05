#!/usr/bin/env python3
"""
Script to extract the first 300 rows from parquet data files.
This tool provides functionality to read parquet files and extract a subset of data.
"""

import argparse
import logging
import os
import sys
from typing import Optional

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]"
)
logger = logging.getLogger(__name__)


def extract_first_300_rows(input_path: str, output_path: Optional[str] = None, num_rows: int = 300) -> pd.DataFrame:
    """
    Extract the first N rows from a parquet file.

    Args:
        input_path (str): Path to the input parquet file
        output_path (Optional[str]): Path to save the extracted data. If None, no file is saved
        num_rows (int): Number of rows to extract (default: 300)

    Returns:
        pd.DataFrame: DataFrame containing the first N rows

    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If the file is not a valid parquet file
        Exception: For other unexpected errors
    """
    try:
        # Validate input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Validate file extension
        if not input_path.lower().endswith(".parquet"):
            logger.warning(f"File {input_path} doesn't have .parquet extension")

        logger.info(f"Reading parquet file: {input_path}")

        # Read the parquet file
        df = pd.read_parquet(input_path)

        # Log basic information about the dataset
        logger.info(f"Original dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")

        # Extract first N rows
        if len(df) < num_rows:
            logger.warning(f"Dataset has only {len(df)} rows, less than requested {num_rows}")
            extracted_df = df.copy()
        else:
            extracted_df = df.head(num_rows).copy()

        logger.info(f"Extracted {len(extracted_df)} rows")

        # Save to output file if specified
        if output_path:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # Save the extracted data
            extracted_df.to_parquet(output_path, index=False)
            logger.info(f"Extracted data saved to: {output_path}")

        return extracted_df

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing parquet file: {e}")
        raise ValueError(f"Invalid parquet file: {input_path}") from e
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


def display_data_info(df: pd.DataFrame) -> None:
    """
    Display basic information about the extracted data.

    Args:
        df (pd.DataFrame): DataFrame to analyze
    """
    print("\n" + "=" * 60)
    print("EXTRACTED DATA INFORMATION")
    print("=" * 60)

    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nBasic statistics:")
    print(df.describe())

    print("=" * 60)


def main():
    """
    Main function to handle command line arguments and execute the extraction.
    """
    parser = argparse.ArgumentParser(
        description="Extract the first 300 rows from a parquet file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_first_300.py data/dataset.parquet
  python extract_first_300.py data/dataset.parquet -o output/first_300.parquet
  python extract_first_300.py data/dataset.parquet -n 500 -o output/first_500.parquet
        """,
    )

    parser.add_argument("input_file", help="Path to the input parquet file")

    parser.add_argument("-o", "--output", help="Path to save the extracted data (optional)")

    parser.add_argument("-n", "--num-rows", type=int, default=300, help="Number of rows to extract (default: 300)")

    parser.add_argument(
        "--show-info", action="store_true", help="Display detailed information about the extracted data"
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Extract the data
        extracted_df = extract_first_300_rows(
            input_path=args.input_file, output_path=args.output, num_rows=args.num_rows
        )

        # Display information if requested
        if args.show_info:
            display_data_info(extracted_df)

        logger.info("Extraction completed successfully!")

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
