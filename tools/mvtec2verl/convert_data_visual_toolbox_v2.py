import base64
import json
import logging
import os
from dataclasses import dataclass
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

# System prompt for visual_toolbox_v2 with defect detection focus
VISUAL_TOOLBOX_V2_SYSTEM_PROMPT: str = """You are a highly precise and meticulous quality control inspector equipped with advanced visual analysis tools. Your primary mission is to analyze images and determine if any defects are present. Your decision must be grounded in visual evidence.

# Tools
You may call one or more functions to assist with defect detection:
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label. Use this tool to get a closer look at potential defects.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox_2d"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "potential defect area"}}  
</tool_call>

# Response Format
Carefully follow these instructions for your response format:

**If you detect one or more defects:**
Your response MUST be structured with the following four tags in this exact order:
1. `<think></think>`: Provide a step-by-step reasoning process. Describe the visual characteristics of the anomaly (e.g., "I observe a dark, irregular crack on the upper left surface..."). Use tools if needed for closer inspection.
2. `<location></location>`: Provide a JSON list of all detected defect locations. Notice! You should give the location of only the defects, not the full object. Each item in the list must be a JSON object with a "bbox2d" key and coordinates in `[x_min, y_min, x_max, y_max]` format. For example: `[{"bbox2d": [100, 150, 200, 250]}, {"bbox2d": [300, 350, 400, 450]}]`. Do not give more than 3 bounding boxes. If you are uncertain about the exact location, provide an approximate bounding box that best encompasses the defect area.
3. `<type></type>`: Specify the type of defect found (e.g., "crack", "discoloration", "scratch", "hole", "surface" and "other"). If the type is uncertain, use "unspecified".
4. `<answer></answer>`: Conclude with "yes".

**If you detect NO defects:**
Your response MUST be structured with the following four tags:
1. `<think></think>`: Explain why you believe the object is defect-free. Describe the normal and healthy features you observed. Use tools if needed for thorough inspection.
2. `<location></location>`: Provide an empty JSON list.
3. `<type></type>`: "good".
4. `<answer></answer>`: Conclude with "no"."""

# User prompt for defect detection with visual_toolbox_v2
VISUAL_TOOLBOX_V2_USER_PROMPT: str = (
    "<image>\n"
    "Analyze this image for defects. Use the image_zoom_in_tool if you need to examine specific areas more closely. "
    "If there is no defect, answer 'no'. If there is a defect, answer 'yes'. "
    "Format your response strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) "
    "<location>...</location> <type>...</type> <answer>...</answer>"
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


def _visual_toolbox_v2_reward_model_value(item: Dict[str, Any]) -> Any:
    """
    Create reward model value specifically for visual_toolbox_v2 defect detection evaluation.
    Aligns with the visual_toolbox_v2 environment and tools for quality control inspection.
    Uses the specialized visual_toolbox_v2_reward module for enhanced evaluation.
    """
    bboxes = _format_answer_bboxes(item)
    answer = (
        "Yes. There has been a defect detected."
        if item.get("label") == 1
        else "No. There is no defect detected."
    )

    # Specify visual_toolbox_v2 specific reward model
    reward_value = {
        "style": "visual_toolbox_v2",  # Use specialized visual_toolbox_v2 reward system
        "ground_truth": {"answer": answer, "bboxes": bboxes},
        "use_visual_tool": True,  # Enable visual tool functionality
        "reward_function": "compute_visual_toolbox_v2_score",  # Specialized VT2 reward function
        "reward_module": "verl.utils.reward_score.visual_toolbox_v2_reward",  # Reference to new module
        "enhanced_grounding": True,  # Enable enhanced grounding evaluation
        "tool_aware_scoring": True,  # Enable tool-aware scoring
    }
    return reward_value


def convert_for_visual_toolbox_v2(
    input_jsonl: str,
    dataset_root: str,
    output_path: str,
    output_format: str = "parquet",
    limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[Record]]:
    """
    Convert data specifically for visual_toolbox_v2 defect detection training.
    Uses visual_toolbox_v2-specific prompts with defect detection focus and reward models
    that align with the environment implementation for quality control inspection.
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
            log.warning(
                f"[{idx}] Failed to read image {img_path}: {e}; skipping."
            )
            continue

        # Use visual_toolbox_v2 specific prompts for defect detection
        prompt = [
            {"role": "system", "content": VISUAL_TOOLBOX_V2_SYSTEM_PROMPT},
            {"role": "user", "content": VISUAL_TOOLBOX_V2_USER_PROMPT},
        ]
        images = [{"bytes": img_bytes}]

        # Use visual_toolbox_v2-specific reward model
        reward_model = _visual_toolbox_v2_reward_model_value(item)
        bboxes_formatted = _format_answer_bboxes(item)

        extra_info = {
            "answer": reward_model["ground_truth"],
            "question": VISUAL_TOOLBOX_V2_USER_PROMPT,
            "clsname": item.get("clsname"),
            "label": item.get("label"),
            "type": item.get(
                "label_name", "unspecified"
            ),  # Defect type for visual_toolbox_v2
            "defect_type": item.get(
                "label_name", "unspecified"
            ),  # Alternative key for defect type
            "anomaly_type": item.get(
                "label_name", "unspecified"
            ),  # Another key for defect type
            "bboxes": bboxes_formatted,
            "tool_enabled": True,  # Mark as tool-enabled dataset
            "visual_toolbox_v2_available": True,  # Specific visual_toolbox_v2 flag
            "reward_type": "visual_toolbox_v2",  # Specify specialized reward type
            "reward_function": "compute_visual_toolbox_v2_score",  # Reference to the new function
            "reward_module": "verl.utils.reward_score.visual_toolbox_v2_reward",  # Module path
            "supported_tools": [
                "image_zoom_in_tool",
                "image_rotate_tool",
            ],  # Tools available in visual_toolbox_v2
            "enhanced_scoring": True,  # Flag for enhanced scoring capabilities
            "grounding_evaluation": True,  # Enable grounding evaluation
            "tool_usage_tracking": True,  # Enable tool usage tracking
            "industrial_inspection": True,  # Mark as industrial inspection task
        }
        # Add sequential index for rows that are actually written
        extra_info["index"] = write_index

        record = Record(
            data_source="vstar_visual_toolbox_v2",  # Distinguish from other variants
            prompt=prompt,
            images=images,
            ability="vl_agent",  # Standard VL agent ability
            env_name="visual_toolbox_v2",  # Use visual_toolbox_v2 environment
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
            f"Processed {total_items} items for visual_toolbox_v2 defect detection (limit={limit if limit else 'none'}), "
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
            f"Processed {total_items} items for visual_toolbox_v2 defect detection (limit={limit if limit else 'none'}), "
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
    # Load configuration from visual_toolbox_v2_data.toml located next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    toml_path = os.path.join(script_dir, "visual_toolbox_v2_data.toml")

    # Fallback to data.toml if visual_toolbox_v2_data.toml doesn't exist
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
        "output", "verl_visual_toolbox_v2_dataset.parquet"
    )  # Different default name
    output_format = cfg.get("format", "parquet")
    limit = cfg.get("limit", None)

    if not os.path.exists(input_path):
        log.error(f"Input file not found: {input_path}")
        return

    os.makedirs(
        os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True
    )

    convert_for_visual_toolbox_v2(
        input_jsonl=input_path,
        dataset_root=dataset_root,
        output_path=output_path,
        output_format=output_format,
        limit=limit,
    )


if __name__ == "__main__":
    main()
