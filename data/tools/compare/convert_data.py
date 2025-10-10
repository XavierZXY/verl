import base64
import json
import logging
import os
import random
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

import pandas as pd
from PIL import Image
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
    "You are a highly precise and meticulous quality control inspector. Your primary mission is to "
    "analyze images and determine if any defects are present. "
    "Your decision must be grounded in visual evidence."
    "Carefully follow these instructions for your response format:"
    "**If you detect one or more defects:**"
    "Your response MUST be structured with the following four tags in this exact order:"
    "1.  `<think></think>`: Provide a step-by-step reasoning process. Describe the visual "
    'characteristics of the anomaly (e.g., "I observe a dark, irregular crack on the upper left surface...").'
    "2.  `<location></location>`: Provide a JSON list of all detected defect locations.Notice! You should give the location of the only defects not the full object. Each item in the list must be a "
    'JSON object with a "bbox_2d" key and coordinates in `[x_min, y_min, x_max, y_max]` format. For example: `[{"bbox_2d": [100, 150, 200, 250]}, {"bbox_2d": [300, 350, 400, 450]}]`.Do not give more than 3 bounding boxes. '
    "If you are uncertain about the exact location, provide an approximate bounding box that best encompasses the defect area."
    '3.  `<type></type>`: Specify the type of defect found (e.g., "crack", "discoloration", "scratch", "hole", "surface" and "other"). If the type is uncertain, use "unspecified".'
    '4.  `<answer></answer>`: Conclude with "Yes. There has been a defect detected.".'
    "**If you detect NO defects:**"
    "Your response MUST be structured with the following two tags:"
    "1.  `<think></think>`: Explain why you believe the object is defect-free. Describe the normal and healthy features you observed."
    "2.  `<location></location>`: Provide an empty JSON list."
    '3.  `<type></type>`: "good".'
    '4.  `<answer></answer>`: Conclude with "No. There is no defect detected.".'
)
# Multiple instruction prompt variants for data diversity
INSTRUCTION_PROMPTS: list[str] = [
    "<image>.\nAnalyze this image for defects. If there is no defect, answer 'no'. If there is defect, answer 'yes'.There is a reference image. You can use the reference image to help you analyze the defect. <image>",
    "<image>.\nExamine this image carefully and determine whether any defects are present. Respond with 'no' if defect-free, 'yes' if defects are found.There is a reference image. You can use the reference image to help you analyze the defect. <image>",
    "<image>.\nInspect this image for any quality issues or anomalies. Answer 'no' for normal items, 'yes' if defects are found.There is a reference image. You can use the reference image to help you analyze the defect. <image>",
    "<image>.\nPlease evaluate this image to identify any manufacturing defects. Reply 'no' if the item is good, 'yes' if there are defects.There is a reference image. You can use the reference image to help you analyze the defect. <image>",
    "<image>.\nLook at this image and assess whether there are any flaws or irregularities. Answer 'no' if perfect, 'yes' if imperfect.There is a reference image. You can use the reference image to help you analyze the defect. <image>",
    "<image>.\nReview this image for defect detection. Provide 'no' if the object appears normal, 'yes' if abnormalities are detected.There is a reference image. You can use the reference image to help you analyze the defect. <image>",
    "<image>.\nCheck this image for any signs of damage or defects. Respond 'no' for intact items, 'yes' for damaged items.There is a reference image. You can use the reference image to help you analyze the defect. <image>",
    "<image>.\nAnalyze the quality of the object in this image. Answer 'no' if it meets quality standards, 'yes' if it has defects.There is a reference image. You can use the reference image to help you analyze the defect. <image>",
    "<image>.\nExamine this image to determine if the item has any defects or quality issues. Reply 'no' if acceptable, 'yes' if unacceptable.There is a reference image. You can use the reference image to help you analyze the defect. <image>",
    "<image>.\nInspect this image and identify whether any defects are visible. Answer 'no' for defect-free objects, 'yes' for defective objects.There is a reference image. You can use the reference image to help you analyze the defect. <image>",
]


def get_random_instruction_prompt() -> tuple[str, int]:
    """
    Get a random instruction prompt from the available variants.

    Returns:
        tuple: (selected_prompt, prompt_index)
    """
    prompt_index = random.randint(0, len(INSTRUCTION_PROMPTS) - 1)
    return INSTRUCTION_PROMPTS[prompt_index], prompt_index


@dataclass
class Record:
    data_source: str
    prompt: list[dict[str, Any]]
    images: list[dict[str, bytes]]
    ability: str
    env_name: str
    reward_model: Any
    extra_info: dict[str, Any]


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
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


def _resolve_image_path(item: dict[str, Any], root: str) -> Optional[str]:
    # Prefer 'foreground' if present, else fallback to 'filename'
    key = "filename"
    val = item.get(key)
    if isinstance(val, str):
        candidate = os.path.join(root, val) if not os.path.isabs(val) else val
        if os.path.exists(candidate):
            return candidate
    return None


def _read_image_bytes(
    path: str,
    compress: bool = True,
    quality: int = 85,
    max_size: Optional[tuple[int, int]] = None,
) -> bytes:
    """
    读取图像文件并返回字节数据，支持压缩选项

    Args:
        path: 图像文件路径
        compress: 是否启用压缩 (默认: True)
        quality: JPEG压缩质量 1-100 (默认: 85)
        max_size: 最大尺寸 (width, height)，如果指定则会等比例缩放

    Returns:
        压缩后的图像字节数据
    """
    if not compress:
        # 如果不压缩，直接返回原始字节
        with open(path, "rb") as f:
            return f.read()

    try:
        # 使用PIL打开图像
        with Image.open(path) as img:
            # 转换为RGB模式（确保兼容JPEG格式）
            if img.mode in ("RGBA", "LA", "P"):
                # 对于带透明度的图像，创建白色背景
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # 如果指定了最大尺寸，进行等比例缩放
            if max_size is not None:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)

            # 压缩图像到内存
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            return buffer.getvalue()

    except Exception as e:
        # 如果PIL处理失败，回退到原始读取方式
        log.warning(f"Failed to compress image {path}: {e}, falling back to original")
        with open(path, "rb") as f:
            return f.read()


def _format_answer_bboxes(item: dict[str, Any]) -> list[dict[str, list[int]]]:
    bboxes = item.get("bboxes")
    if not bboxes:
        return []
    formatted: list[dict[str, list[int]]] = []
    for bbox in bboxes:
        if not isinstance(bbox, list | tuple) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        try:
            formatted.append({"bbox_2d": [int(x1), int(y1), int(x2), int(y2)]})
        except Exception:
            # Skip invalid numeric conversions
            continue
    return formatted


def _load_reference_images(good_jsonl_path: str, dataset_root: str) -> dict[str, list[dict[str, Any]]]:
    """
    Load reference images from good.jsonl and organize them by class.

    Args:
        good_jsonl_path: Path to the good.jsonl file containing reference images
        dataset_root: Root directory for resolving image paths

    Returns:
        Dictionary mapping class names to lists of good image items
    """
    reference_images: dict[str, list[dict[str, Any]]] = {}

    if not os.path.exists(good_jsonl_path):
        log.warning(f"Reference image file not found: {good_jsonl_path}")
        return reference_images

    good_items = _read_jsonl(good_jsonl_path)

    for item in good_items:
        clsname = item.get("clsname")
        if clsname and item.get("label") == 0:  # Only use good images (label=0)
            if clsname not in reference_images:
                reference_images[clsname] = []
            reference_images[clsname].append(item)

    log.info(f"Loaded reference images for {len(reference_images)} classes: {list(reference_images.keys())}")
    for cls, items in reference_images.items():
        log.info(f"  {cls}: {len(items)} reference images")

    return reference_images


def _get_random_reference_image(
    reference_images: dict[str, list[dict[str, Any]]],
    clsname: str,
    dataset_root: str,
    compress_images: bool = True,
    image_quality: int = 85,
    max_image_size: Optional[tuple[int, int]] = None,
) -> Optional[bytes]:
    """
    Get a random reference image for the given class.

    Args:
        reference_images: Dictionary of reference images organized by class
        clsname: Class name to get reference image for
        dataset_root: Root directory for resolving image paths
        compress_images: Whether to compress the image
        image_quality: JPEG compression quality
        max_image_size: Maximum image size for resizing

    Returns:
        Image bytes if found, None otherwise
    """
    if clsname not in reference_images or not reference_images[clsname]:
        log.warning(f"No reference images found for class: {clsname}")
        return None

    # Randomly select a reference image
    ref_item = random.choice(reference_images[clsname])
    ref_img_path = _resolve_image_path(ref_item, dataset_root)

    if ref_img_path is None:
        log.warning(f"Reference image path not found for class {clsname}")
        return None

    try:
        ref_img_bytes = _read_image_bytes(
            ref_img_path,
            compress=compress_images,
            quality=image_quality,
            max_size=max_image_size,
        )
        return ref_img_bytes
    except Exception as e:
        log.warning(f"Failed to read reference image {ref_img_path}: {e}")
        return None


def _reward_model_value(item: dict[str, Any]) -> Any:
    # Use original bboxes directly if present; else empty list
    bboxes = _format_answer_bboxes(item)
    # log.info(f"Item label: {item.get('label')}")
    answer = "Yes. There has been a defect detected." if item.get("label") == 1 else "No. There is no defect detected."
    reward_value = {
        "style": "model",
        "ground_truth": {"answer": answer, "bboxes": bboxes},
    }
    return reward_value


def convert(
    input_jsonl: str,
    dataset_root: str,
    output_path: str,
    good_jsonl_path: str,
    output_format: str = "parquet",
    limit: Optional[int] = None,
    compress_images: bool = True,
    image_quality: int = 85,
    max_image_size: Optional[tuple[int, int]] = None,
) -> tuple[pd.DataFrame, list[Record]]:
    rows: list[Record] = []
    items = _read_jsonl(input_jsonl)
    total_items = len(items)
    if limit is not None and limit > 0:
        items = items[:limit]

    # Load reference images from good.jsonl
    reference_images = _load_reference_images(good_jsonl_path, dataset_root)

    write_index = 0
    for idx, item in enumerate(items):
        img_path = _resolve_image_path(item, dataset_root)
        if img_path is None:
            log.warning(f"[{idx}] Image path not found for item; skipping. keys={list(item.keys())}")
            continue
        try:
            img_bytes = _read_image_bytes(
                img_path,
                compress=compress_images,
                quality=image_quality,
                max_size=max_image_size,
            )
        except FileNotFoundError:
            log.warning(f"[{idx}] Image file missing: {img_path}; skipping.")
            continue
        except Exception as e:
            log.warning(f"[{idx}] Failed to read image {img_path}: {e}; skipping.")
            continue

        # Get reference image for the same class
        clsname = item.get("clsname")
        ref_img_bytes = _get_random_reference_image(
            reference_images, clsname, dataset_root, compress_images, image_quality, max_image_size
        )

        # Get random instruction prompt for data diversity
        selected_instruction_prompt, prompt_index = get_random_instruction_prompt()

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": selected_instruction_prompt},
        ]

        # First image is the main image from train.jsonl, second is reference from good.jsonl
        images = [{"bytes": img_bytes}]
        if ref_img_bytes is not None:
            images.append({"bytes": ref_img_bytes})
        else:
            log.warning(f"[{idx}] No reference image found for class {clsname}, using only main image")

        reward_model = _reward_model_value(item)
        bboxes_formatted = _format_answer_bboxes(item)

        extra_info = {
            "answer": reward_model["ground_truth"],
            "question": selected_instruction_prompt,
            "prompt_variant_index": prompt_index,  # Track which prompt variant was used
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
            f"Processed {total_items} items (limit={limit if limit else 'none'}), wrote parquet with "
            f"{len(df)} rows to: {output_path}"
        )
    elif output_format == "jsonl":
        # Serialize bytes as base64 for JSONL
        with open(output_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                images_serialized = []
                for img in row_dict.get("images", []):
                    b = img.get("bytes", b"")
                    images_serialized.append({"bytes": base64.b64encode(b).decode("ascii")})
                row_dict["images"] = images_serialized
                f.write(json.dumps(row_dict, ensure_ascii=False) + "\n")
        log.info(
            f"Processed {total_items} items (limit={limit if limit else 'none'}), wrote JSONL with "
            f"{len(df)} rows to: {output_path}"
        )
    else:
        raise ValueError("output_format must be 'parquet' or 'jsonl'")

    return df, rows


def _load_config_from_toml(toml_path: str) -> dict[str, Any]:
    if tomllib is None:
        raise RuntimeError("TOML parser not available. Please use Python 3.11+ or install 'tomli'.")
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
    # input, root, output, format, limit, compress_images, image_quality, max_image_size, good_jsonl
    input_path = cfg.get("input", "data.jsonl")
    dataset_root = cfg.get("root", ".")
    output_path = cfg.get("output", "verl_dataset.parquet")
    output_format = cfg.get("format", "parquet")
    limit = cfg.get("limit", None)
    good_jsonl_path = cfg.get("good_jsonl", "good.jsonl")

    # 图像压缩相关配置
    compress_images = cfg.get("compress_images", True)
    image_quality = cfg.get("image_quality", 85)
    max_image_size = cfg.get("max_image_size", None)

    # 如果max_image_size是列表，转换为元组
    if isinstance(max_image_size, list) and len(max_image_size) == 2:
        max_image_size = tuple(max_image_size)
    elif max_image_size is not None:
        log.warning(f"Invalid max_image_size format: {max_image_size}, ignoring")
        max_image_size = None

    if not os.path.exists(input_path):
        log.error(f"Input file not found: {input_path}")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    convert(
        input_jsonl=input_path,
        dataset_root=dataset_root,
        output_path=output_path,
        good_jsonl_path=good_jsonl_path,
        output_format=output_format,
        limit=limit,
        compress_images=compress_images,
        image_quality=image_quality,
        max_image_size=max_image_size,
    )


if __name__ == "__main__":
    main()
