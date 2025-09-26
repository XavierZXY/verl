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

SYSTEM_PROMPT: str = (
    "You are a highly precise and meticulous quality control inspector. Your primary mission is to analyze images and determine if any defects are present. Your decision must be grounded in visual evidence."
    "Carefully follow these instructions for your response format:"
    "**If you detect one or more defects:**"
    "Your response MUST be structured with the following four tags in this exact order:"
    '1.  `<think></think>`: Provide a step-by-step reasoning process. Describe the visual characteristics of the anomaly (e.g., "I observe a dark, irregular crack on the upper left surface...").'
    '2.  `<location></location>`: Provide a JSON list of all detected defect locations.Notice! You should give the location of the only defects not the full object. Each item in the list must be a JSON object with a "bbox2d" key and coordinates in `[x_min, y_min, x_max, y_max]` format. For example: `[{"bbox2d": [100, 150, 200, 250]}, {"bbox2d": [300, 350, 400, 450]}]`.Do not give more than 3 bounding boxes. If you are uncertain about the exact location, provide an approximate bounding box that best encompasses the defect area.'
    '3.  `<type></type>`: Specify the type of defect found (e.g., "crack", "discoloration", "scratch", "hole", "surface" and "other"). If the type is uncertain, use "unspecified".'
    '4.  `<answer></answer>`: Conclude with "yes".'
    "**If you detect NO defects:**"
    "Your response MUST be structured with the following two tags:"
    "1.  `<think></think>`: Explain why you believe the object is defect-free. Describe the normal and healthy features you observed."
    "2.  `<location></location>`: Provide an empty JSON list."
    '3.  `<type></type>`: "good".'
    '4.  `<answer></answer>`: Conclude with "no".'
)
INSTRUCTION_PROMPT: str = "<image>.\nAnalyze this  image for defects. If there is no defect, answer 'no'. If there is defect, answer 'yes'."


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
            formatted.append({"bbox_2d": [int(x1), int(y1), int(x2), int(y2)]})
        except Exception:
            # Skip invalid numeric conversions
            continue
    return formatted


def _reward_model_value(item: Dict[str, Any]) -> Any:
    # Use original bboxes directly if present; else empty list
    bboxes = _format_answer_bboxes(item)
    # log.info(f"Item label: {item.get('label')}")
    answer = (
        "Yes. There has been a defect detected."
        if item.get("label") == 1
        else "No. There is no defect detected."
    )
    reward_value = {
        "style": "model",
        "ground_truth": {"answer": answer, "bboxes": bboxes},
    }
    return reward_value


def convert(
    input_jsonl: str,
    dataset_root: str,
    output_path: str,
    output_format: str = "parquet",
    limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[Record]]:
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

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": INSTRUCTION_PROMPT},
        ]
        images = [{"bytes": img_bytes}]

        reward_model = _reward_model_value(item)
        bboxes_formatted = _format_answer_bboxes(item)

        extra_info = {
            "answer": reward_model["ground_truth"],
            "question": INSTRUCTION_PROMPT,
            "clsname": item.get("clsname"),
            "label": item.get("label"),
            "type": item.get("label_name"),
            "bboxes": bboxes_formatted,
        }
        # Add sequential index for rows that are actually written
        extra_info["index"] = write_index

        record = Record(
            data_source="vstar",
            prompt=prompt,
            images=images,
            ability="vl_chart",
            env_name="visual_toolbox_v2",
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
            f"Processed {total_items} items (limit={limit if limit else 'none'}), wrote parquet with {len(df)} rows to: {output_path}"
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
            f"Processed {total_items} items (limit={limit if limit else 'none'}), wrote JSONL with {len(df)} rows to: {output_path}"
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
    # Load configuration from data.toml located next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    toml_path = os.path.join(script_dir, "data.toml")
    try:
        cfg = _load_config_from_toml(toml_path)
    except Exception as e:
        log.error(f"Failed to load TOML config: {e}")
        return

    # Expected keys in the root of TOML:
    # input, root, output, format, limit
    input_path = cfg.get("input", "data.jsonl")
    dataset_root = cfg.get("root", ".")
    output_path = cfg.get("output", "verl_dataset.parquet")
    output_format = cfg.get("format", "parquet")
    limit = cfg.get("limit", None)

    if not os.path.exists(input_path):
        log.error(f"Input file not found: {input_path}")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    convert(
        input_jsonl=input_path,
        dataset_root=dataset_root,
        output_path=output_path,
        output_format=output_format,
        limit=limit,
    )


if __name__ == "__main__":
    main()
