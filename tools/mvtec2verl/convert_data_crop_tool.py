import base64
import json
import logging
import os
from dataclasses import dataclass
from re import I
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rich.logging import RichHandler

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    tomllib = None
    try:
        import tomli as tomllib  # type: ignore
    except Exception:  # pragma: no cover
        tomllib = None


# Configure rich logging (English comments and outputs, no print usage)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("rich")

CROP_INSPECTION_SYSTEM_PROMPT: str = (
    "You are an advanced quality control inspector equipped with crop inspection tools for detailed defect analysis. "
    "Your mission is to thoroughly analyze images to determine if any defects are present using progressive inspection techniques."
    "\n\n**INSPECTION METHODOLOGY:**\n"
    "1. **Initial Assessment**: First examine the full image for potential defect areas\n"
    "2. **Progressive Cropping**: Use the crop tool to focus on suspicious regions for detailed inspection\n"
    "3. **Multi-level Analysis**: You can perform up to 3 levels of cropping to get increasingly detailed views\n"
    "4. **Final Decision**: Make your final determination based on the detailed inspection\n"
    "\n**AVAILABLE TOOLS:**\n"
    "- `crop_from_location`: Crop specific regions based on location data for detailed inspection\n"
    "  - Arguments: `location_data` (JSON string with bbox coordinates), `crop_index` (which bbox to crop)\n"
    '  - Example: `{"name": "crop_from_location", "arguments": {"location_data": "[{\\"bbox2d\\": [100, 150, 200, 250]}]", "crop_index": 0}}`\n'
    "\n**RESPONSE FORMAT:**\n"
    "Your response must follow this structure:\n"
    "1. `<think></think>`: Your reasoning process and inspection strategy\n"
    "2. **Tool Usage** (if needed): Use `<tool_call></tool_call>` tags to crop suspicious areas\n"
    "3. **Final Response** (required): Provide all four tags in this exact order:\n"
    '   - `<answer></answer>`: "yes" if defects found, "no" if no defects\n'
    "   - `<location></location>`: JSON list of defect locations (empty list if no defects)\n"
    '   - `<type></type>`: Defect type ("good" if no defects, otherwise specify type like "crack", "scratch", etc.)\n'
    "\n**LOCATION FORMAT:**\n"
    'For defects, provide coordinates as: `[{"bbox2d": [x_min, y_min, x_max, y_max]}]`\n'
    "Maximum 3 bounding boxes. If uncertain, provide approximate bounding box encompassing the defect area.\n"
    "\n**DEFECT TYPES:**\n"
    'Common types: "crack", "discoloration", "scratch", "hole", "surface", "contamination", "other"\n'
    'Use "unspecified" if type is uncertain, "good" only if no defects found.\n'
    "\n**INSPECTION STRATEGY:**\n"
    "- Start with broad assessment of the entire image\n"
    "- If suspicious areas are found, use crop tool to examine them closely\n"
    "- Use progressive cropping (coarse to fine) for better accuracy\n"
    "- Limit crops to 2-3 levels for efficiency\n"
    "- Make final decision based on detailed inspection results"
)

CROP_INSPECTION_INSTRUCTION_PROMPT: str = (
    "<image>\n"
    "Analyze this image for defects using crop inspection tools for detailed examination. "
    "You may use the crop_from_location tool to examine suspicious areas more closely. "
    "If there is no defect, answer 'no'. If there is a defect, answer 'yes'."
)


@dataclass
class Record:
    data_source: str
    prompt: List[Dict[str, Any]]
    images: List[Dict[str, bytes]]
    ability: str
    env_name: str
    reward_model: Any
    extra_info: Dict[str, Any]


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"Skip malformed JSON at line {line_no}: {e}")
                continue
            items.append(obj)
    return items


def _resolve_image_path(item: Dict[str, Any], root: str) -> Optional[str]:
    # Prefer 'foreground' if present, else fallback to 'filename'
    key = "filename"
    val = item.get(key)
    if isinstance(val, str):
        candidate = os.path.join(root, val) if not os.path.isabs(val) else val
        if os.path.exists(candidate):
            return candidate
    return None


def _read_image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _format_answer_bboxes(item: Dict[str, Any]) -> List[Dict[str, List[int]]]:
    bboxes = item.get("bboxes")
    if not bboxes:
        return []
    formatted: List[Dict[str, List[int]]] = []
    for bbox in bboxes:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        try:
            formatted.append({"bbox2d": [int(x1), int(y1), int(x2), int(y2)]})
        except Exception:
            # Skip invalid numeric conversions
            continue
    return formatted


def _crop_reward_model_value(item: Dict[str, Any]) -> Any:
    """
    Create reward model value specifically for crop inspection tool evaluation.
    Uses crop_inspection_reward instead of standard vl_agent reward.
    """
    bboxes = _format_answer_bboxes(item)
    answer = (
        "Yes. There has been a defect detected."
        if item.get("label") == 1
        else "No. There is no defect detected."
    )

    # Specify crop inspection reward model
    reward_value = {
        "style": "crop_inspection",  # Use crop inspection reward system
        "ground_truth": {"answer": answer, "bboxes": bboxes},
        "use_crop_tool": True,  # Enable crop tool functionality
        "reward_function": "compute_crop_inspection_score",  # Specific reward function
    }
    return reward_value


def convert_for_crop_tool(
    input_jsonl: str,
    dataset_root: str,
    output_path: str,
    output_format: str = "parquet",
    limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[Record]]:
    """
    Convert data specifically for crop inspection tool training.
    Uses crop-specific prompts and reward models.
    """
    rows: List[Record] = []
    items = _read_jsonl(input_jsonl)
    total_items = len(items)
    if limit is not None and limit > 0:
        items = items[:limit]

    write_index = 0
    for idx, item in enumerate(items):
        img_path = _resolve_image_path(item, dataset_root)
        if img_path is None:
            log.warning(
                f"[{idx}] Image path not found for item; skipping. keys={list(item.keys())}"
            )
            continue
        try:
            img_bytes = _read_image_bytes(img_path)
        except FileNotFoundError:
            log.warning(f"[{idx}] Image file missing: {img_path}; skipping.")
            continue
        except Exception as e:
            log.warning(f"[{idx}] Failed to read image {img_path}: {e}; skipping.")
            continue

        # Use crop inspection specific prompts
        prompt = [
            {"role": "system", "content": CROP_INSPECTION_SYSTEM_PROMPT},
            {"role": "user", "content": CROP_INSPECTION_INSTRUCTION_PROMPT},
        ]
        images = [{"bytes": img_bytes}]

        # Use crop-specific reward model
        reward_model = _crop_reward_model_value(item)
        bboxes_formatted = _format_answer_bboxes(item)

        extra_info = {
            "answer": reward_model["ground_truth"],
            "question": CROP_INSPECTION_INSTRUCTION_PROMPT,
            "clsname": item.get("clsname"),
            "label": item.get("label"),
            "type": item.get("label_name"),
            "bboxes": bboxes_formatted,
            "tool_enabled": True,  # Mark as tool-enabled dataset
            "crop_tool_available": True,  # Specific crop tool flag
            "reward_type": "crop_inspection",  # Specify reward type
        }
        # Add sequential index for rows that are actually written
        extra_info["index"] = write_index

        record = Record(
            data_source="vstar_crop",  # Distinguish from regular vstar
            prompt=prompt,
            images=images,
            ability="vl_crop_inspection",  # New ability type
            env_name="visual_toolbox_v2",  # Use crop inspection environment
            reward_model=reward_model,
            extra_info=extra_info,
        )
        rows.append(record)
        write_index += 1

    # Build DataFrame compatible with the example schema
    df = pd.DataFrame(
        [
            {
                "data_source": r.data_source,
                "prompt": r.prompt,
                "images": r.images,
                "ability": r.ability,
                "env_name": r.env_name,
                "reward_model": r.reward_model,
                "extra_info": r.extra_info,
            }
            for r in rows
        ]
    )

    # Persist
    output_format = output_format.lower()
    if output_format == "parquet":
        df.to_parquet(output_path, index=False)
        log.info(
            f"Processed {total_items} items for crop inspection tool (limit={limit if limit else 'none'}), "
            f"wrote parquet with {len(df)} rows to: {output_path}"
        )
    elif output_format == "jsonl":
        # Serialize bytes as base64 for JSONL
        with open(output_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                images_serialized = []
                for img in row_dict.get("images", []):
                    b = img.get("bytes", b"")
                    images_serialized.append(
                        {"bytes": base64.b64encode(b).decode("ascii")}
                    )
                row_dict["images"] = images_serialized
                f.write(json.dumps(row_dict, ensure_ascii=False) + "\n")
        log.info(
            f"Processed {total_items} items for crop inspection tool (limit={limit if limit else 'none'}), "
            f"wrote JSONL with {len(df)} rows to: {output_path}"
        )
    else:
        raise ValueError("output_format must be 'parquet' or 'jsonl'")

    return df, rows


def _load_config_from_toml(toml_path: str) -> Dict[str, Any]:
    if tomllib is None:
        raise RuntimeError(
            "TOML parser not available. Please use Python 3.11+ or install 'tomli'."
        )
    if not os.path.exists(toml_path):
        raise FileNotFoundError(f"Config file not found: {toml_path}")
    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)
    return cfg or {}


def main() -> None:
    # Load configuration from crop_data.toml located next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    toml_path = os.path.join(script_dir, "crop_data.toml")

    # Fallback to data.toml if crop_data.toml doesn't exist
    if not os.path.exists(toml_path):
        toml_path = os.path.join(script_dir, "data.toml")
        log.info("Using fallback data.toml configuration")

    try:
        cfg = _load_config_from_toml(toml_path)
    except Exception as e:
        log.error(f"Failed to load TOML config: {e}")
        return

    # Expected keys in the root of TOML:
    # input, root, output, format, limit
    input_path = cfg.get("input", "data.jsonl")
    dataset_root = cfg.get("root", ".")
    output_path = cfg.get(
        "output", "verl_crop_dataset.parquet"
    )  # Different default name
    output_format = cfg.get("format", "parquet")
    limit = cfg.get("limit", None)

    if not os.path.exists(input_path):
        log.error(f"Input file not found: {input_path}")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    convert_for_crop_tool(
        input_jsonl=input_path,
        dataset_root=dataset_root,
        output_path=output_path,
        output_format=output_format,
        limit=limit,
    )


if __name__ == "__main__":
    main()
