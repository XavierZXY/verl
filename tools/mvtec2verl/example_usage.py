#!/usr/bin/env python3
"""
Example usage of crop tool data conversion and comparison with standard version.
"""

import json
import logging
import os
from typing import Any, Dict

from rich.logging import RichHandler

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("rich")

# Import conversion functions
from convert_data import convert as standard_convert
from convert_data_crop_tool import convert_for_crop_tool


def create_sample_data(output_path: str = "sample_data.jsonl") -> str:
    """Create a small sample dataset for testing"""
    sample_items = [
        {
            "filename": "defect_001.jpg",
            "clsname": "bottle",
            "label": 1,
            "label_name": "crack",
            "bboxes": [[150, 100, 250, 200]],
        },
        {
            "filename": "good_001.jpg",
            "clsname": "bottle",
            "label": 0,
            "label_name": "good",
            "bboxes": [],
        },
        {
            "filename": "defect_002.jpg",
            "clsname": "metal_nut",
            "label": 1,
            "label_name": "scratch",
            "bboxes": [[80, 80, 120, 120], [200, 150, 240, 190]],
        },
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        for item in sample_items:
            f.write(json.dumps(item) + "\n")

    log.info(f"Created sample data: {output_path}")
    return output_path


def compare_prompts():
    """Compare prompts between standard and crop tool versions"""
    log.info("🔍 Comparing System Prompts")
    log.info("=" * 60)

    from convert_data import SYSTEM_PROMPT as standard_prompt
    from convert_data_crop_tool import (
        CROP_INSPECTION_SYSTEM_PROMPT as crop_prompt,
    )

    log.info(
        "📝 Standard System Prompt Length: {} characters".format(
            len(standard_prompt)
        )
    )
    log.info(
        "🛠️  Crop Tool System Prompt Length: {} characters".format(
            len(crop_prompt)
        )
    )

    # Show key differences
    log.info("\n🔑 Key Differences:")
    log.info("Standard: Single-pass direct analysis")
    log.info("Crop Tool: Multi-round progressive inspection with tools")

    # Show tool instructions presence
    has_tool_instructions = "crop_from_location" in crop_prompt
    log.info(f"Tool Instructions Present: {has_tool_instructions}")

    return {
        "standard_length": len(standard_prompt),
        "crop_tool_length": len(crop_prompt),
        "has_tool_instructions": has_tool_instructions,
    }


def convert_with_both_methods(sample_file: str):
    """Convert the same data using both methods and compare results"""
    log.info("\n🔄 Converting Data with Both Methods")
    log.info("=" * 60)

    # Standard conversion
    log.info("📊 Standard Conversion...")
    standard_df, standard_records = standard_convert(
        input_jsonl=sample_file,
        dataset_root=".",
        output_path="standard_output.parquet",
        limit=3,
    )

    # Crop tool conversion
    log.info("🛠️  Crop Tool Conversion...")
    crop_df, crop_records = convert_for_crop_tool(
        input_jsonl=sample_file,
        dataset_root=".",
        output_path="crop_tool_output.parquet",
        limit=3,
    )

    return {
        "standard": {"df": standard_df, "records": standard_records},
        "crop_tool": {"df": crop_df, "records": crop_records},
    }


def analyze_differences(results: Dict[str, Any]):
    """Analyze differences between standard and crop tool outputs"""
    log.info("\n📈 Analyzing Differences")
    log.info("=" * 60)

    standard_record = results["standard"]["records"][0]
    crop_record = results["crop_tool"]["records"][0]

    # Compare basic fields
    log.info("🏷️  Data Source:")
    log.info(f"  Standard: {standard_record.data_source}")
    log.info(f"  Crop Tool: {crop_record.data_source}")

    log.info("\n🎯 Ability Type:")
    log.info(f"  Standard: {standard_record.ability}")
    log.info(f"  Crop Tool: {crop_record.ability}")

    log.info("\n🔧 Environment:")
    log.info(f"  Standard: {standard_record.env_name}")
    log.info(f"  Crop Tool: {crop_record.env_name}")

    # Compare reward models
    log.info("\n🏆 Reward Model:")
    log.info(f"  Standard Style: {standard_record.reward_model.get('style')}")
    log.info(f"  Crop Tool Style: {crop_record.reward_model.get('style')}")

    # Compare extra info
    standard_extra = standard_record.extra_info
    crop_extra = crop_record.extra_info

    log.info("\n📋 Extra Info Fields:")
    log.info(f"  Standard Fields: {len(standard_extra)} fields")
    log.info(f"  Crop Tool Fields: {len(crop_extra)} fields")

    # Show crop-specific fields
    crop_specific_fields = set(crop_extra.keys()) - set(standard_extra.keys())
    if crop_specific_fields:
        log.info(f"  Crop-Specific Fields: {list(crop_specific_fields)}")

    # Compare prompt lengths
    standard_system = standard_record.prompt[0]["content"]
    crop_system = crop_record.prompt[0]["content"]

    log.info("\n📝 System Prompt Comparison:")
    log.info(f"  Standard Length: {len(standard_system)} chars")
    log.info(f"  Crop Tool Length: {len(crop_system)} chars")
    log.info(
        f"  Increase: {len(crop_system) - len(standard_system)} chars ({((len(crop_system) / len(standard_system) - 1) * 100):.1f}%)"
    )


def demonstrate_expected_responses():
    """Show examples of expected model responses for both versions"""
    log.info("\n💬 Expected Model Response Examples")
    log.info("=" * 60)

    log.info("📊 Standard Model Response:")
    standard_response = """<think>I can see a dark crack-like defect in the upper portion of the bottle.</think>
<location>[{"bbox2d": [150, 100, 250, 200]}]</location>
<type>crack</type>
<answer>yes</answer>"""
    log.info(standard_response)

    log.info("\n🛠️  Crop Tool Model Response:")
    crop_response = """<think>I need to examine this image carefully for defects. I can see a suspicious area in the upper portion that requires closer inspection.</think>

<tool_call>
{"name": "crop_from_location", "arguments": {"location_data": "[{\\"bbox2d\\": [140, 90, 260, 210]}]", "crop_index": 0}}
</tool_call>

Now I can see the cropped region more clearly. There appears to be a crack defect. Let me get an even closer look.

<tool_call>
{"name": "crop_from_location", "arguments": {"location_data": "[{\\"bbox2d\\": [145, 95, 255, 205]}]", "crop_index": 0}}
</tool_call>

After detailed multi-level inspection, I can confirm there is a crack defect present.

<answer>yes</answer>
<location>[{"bbox2d": [150, 100, 250, 200]}]</location>
<type>crack</type>"""
    log.info(crop_response)

    log.info(f"\n📏 Response Length Comparison:")
    log.info(f"  Standard: {len(standard_response)} characters")
    log.info(f"  Crop Tool: {len(crop_response)} characters")
    log.info(f"  Tool Calls: {crop_response.count('tool_call')}")


def main():
    """Main demonstration function"""
    log.info("🚀 Crop Tool Data Conversion Demo")
    log.info("=" * 60)

    # Create sample data
    sample_file = create_sample_data()

    # Compare prompts
    prompt_comparison = compare_prompts()

    # Convert with both methods
    results = convert_with_both_methods(sample_file)

    # Analyze differences
    analyze_differences(results)

    # Show expected responses
    demonstrate_expected_responses()

    # Summary
    log.info("\n✅ Summary")
    log.info("=" * 60)
    log.info("✓ Crop tool version created with enhanced prompts")
    log.info("✓ Progressive inspection methodology integrated")
    log.info("✓ Tool-aware reward system configured")
    log.info("✓ Multi-round conversation support enabled")
    log.info("✓ Specialized environment and ability types set")

    log.info(f"\n📊 Generated Files:")
    log.info(f"  • {sample_file} (sample data)")
    log.info(f"  • standard_output.parquet (standard conversion)")
    log.info(f"  • crop_tool_output.parquet (crop tool conversion)")

    # Cleanup
    cleanup_files = [
        sample_file,
        "standard_output.parquet",
        "crop_tool_output.parquet",
    ]
    for file in cleanup_files:
        if os.path.exists(file):
            os.remove(file)
            log.info(f"  🧹 Cleaned up: {file}")

    log.info("\n🎯 Next Steps:")
    log.info("  1. Use convert_data_crop_tool.py for crop tool training data")
    log.info(
        "  2. Configure training pipeline with crop_inspection_tool environment"
    )
    log.info("  3. Use compute_crop_inspection_score for reward evaluation")
    log.info("  4. Train models with progressive inspection examples")


if __name__ == "__main__":
    main()
