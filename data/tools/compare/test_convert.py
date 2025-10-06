#!/usr/bin/env python3
"""
Test script to verify the modified convert_data.py logic
"""

import json
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convert_data import convert


def test_convert_logic():
    """Test the convert function with reference images"""
    print("Testing convert logic with reference images...")

    # Test parameters
    input_jsonl = "/home/zxy/codes/working/RL/verl/data/mvtec/good/train.jsonl"
    dataset_root = "/home/zxy/codes/working/RL/verl/data/mvtec"
    output_path = "/home/zxy/codes/working/RL/verl/data/tools/compare/test_output.jsonl"
    good_jsonl_path = "/home/zxy/codes/working/RL/verl/data/mvtec/good/good.jsonl"

    try:
        # Run convert with a small limit for testing
        df, records = convert(
            input_jsonl=input_jsonl,
            dataset_root=dataset_root,
            output_path=output_path,
            good_jsonl_path=good_jsonl_path,
            output_format="jsonl",
            limit=3,  # Only process 3 records for testing
            compress_images=True,
            image_quality=85,
            max_image_size=(256, 256),
        )

        print(f"✅ Successfully processed {len(records)} records")

        # Check if records have two images each
        for i, record in enumerate(records):
            num_images = len(record.images)
            print(f"Record {i + 1}: {num_images} images")
            if num_images != 2:
                print(f"❌ Expected 2 images, got {num_images}")
                return False
            else:
                print(f"✅ Record {i + 1} has correct number of images (2)")

        # Check if output file was created
        if os.path.exists(output_path):
            print(f"✅ Output file created: {output_path}")

            # Read and display first record to verify structure
            with open(output_path) as f:
                first_line = f.readline()
                first_record = json.loads(first_line)
                print(f"✅ First record structure: {list(first_record.keys())}")

                # Check if prompt contains two <image> tags
                prompt_content = first_record.get("prompt", [{}])[0].get("content", "")
                image_count = prompt_content.count("<image>")
                print(f"✅ Prompt contains {image_count} <image> tags")

                if image_count == 2:
                    print("✅ Prompt structure is correct!")
                else:
                    print(f"❌ Expected 2 <image> tags, found {image_count}")
                    return False
        else:
            print(f"❌ Output file not created: {output_path}")
            return False

        print("🎉 All tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_convert_logic()
    sys.exit(0 if success else 1)
