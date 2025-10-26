import base64
import json
import logging
import os
import random
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

import cv2
import numpy as np
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


# Configure rich logging
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
    "<image>.\nAnalyze this image for defects. If there is no defect, answer 'no'. If there is defect, answer 'yes'.",
    "<image>.\nExamine this image carefully and determine whether any defects are present. Respond with 'no' if defect-free, 'yes' if defects are found.",
    "<image>.\nInspect this image for any quality issues or anomalies. Answer 'no' for normal items, 'yes' for defective items.",
    "<image>.\nPlease evaluate this image to identify any manufacturing defects. Reply 'no' if the item is good, 'yes' if there are defects.",
    "<image>.\nLook at this image and assess whether there are any flaws or irregularities. Answer 'no' if perfect, 'yes' if imperfect.",
    "<image>.\nReview this image for defect detection. Provide 'no' if the object appears normal, 'yes' if abnormalities are detected.",
    "<image>.\nCheck this image for any signs of damage or defects. Respond 'no' for intact items, 'yes' for damaged items.",
    "<image>.\nAnalyze the quality of the object in this image. Answer 'no' if it meets quality standards, 'yes' if it has defects.",
    "<image>.\nExamine this image to determine if the item has any defects or quality issues. Reply 'no' if acceptable, 'yes' if unacceptable.",
    "<image>.\nInspect this image and identify whether any defects are visible. Answer 'no' for defect-free objects, 'yes' for defective objects.",
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
    """Resolve image path from item"""
    key = "filename"
    val = item.get(key)
    if isinstance(val, str):
        candidate = os.path.join(root, val) if not os.path.isabs(val) else val
        if os.path.exists(candidate):
            return candidate
    return None


def _resolve_mask_path(item: dict[str, Any], root: str) -> Optional[str]:
    """Resolve mask image path from item"""
    key = "mask"
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
    Read image file and return bytes data with compression support

    Args:
        path: Image file path
        compress: Whether to enable compression (default: True)
        quality: JPEG compression quality 1-100 (default: 85)
        max_size: Maximum size (width, height), will scale proportionally if specified

    Returns:
        Compressed image bytes data
    """
    if not compress:
        # If no compression, return original bytes
        with open(path, "rb") as f:
            return f.read()

    try:
        # Open image with PIL
        with Image.open(path) as img:
            # Convert to RGB mode (ensure JPEG format compatibility)
            if img.mode in ("RGBA", "LA", "P"):
                # For images with transparency, create white background
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # If max_size is specified, scale proportionally
            if max_size is not None:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Compress image to memory
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            return buffer.getvalue()

    except Exception as e:
        # If PIL processing fails, fallback to original reading
        log.warning(f"Failed to compress image {path}: {e}, falling back to original")
        with open(path, "rb") as f:
            return f.read()


def _read_mask_image_bytes(
    path: str,
    max_size: Optional[tuple[int, int]] = None,
) -> Optional[bytes]:
    """
    Read mask image and apply same resize as main image

    Args:
        path: Mask image file path
        max_size: Maximum size (width, height), same as main image

    Returns:
        Processed mask image bytes or None if failed
    """
    try:
        with Image.open(path) as img:
            # Keep mask as grayscale or binary
            if img.mode not in ("L", "1"):
                img = img.convert("L")

            # Apply same resize as main image
            if max_size is not None:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Save as PNG to preserve mask quality
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()

    except Exception as e:
        log.warning(f"Failed to read mask image {path}: {e}")
        return None


def _detect_defect_regions(mask_bytes: bytes, min_area: int = 100) -> list[tuple[int, int, int, int]]:
    """
    Detect individual defect regions from mask image using connected components analysis

    Args:
        mask_bytes: Mask image bytes data
        min_area: Minimum area threshold to filter out noise (default: 100 pixels)

    Returns:
        List of bounding boxes [(x_min, y_min, x_max, y_max), ...] for each defect region
    """
    try:
        # Load mask image from bytes
        mask_image = Image.open(BytesIO(mask_bytes))
        mask_array = np.array(mask_image.convert("L"))

        # Binarize: threshold at 127 (anything > 127 is considered defect)
        _, binary_mask = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        defect_regions = []
        # Skip label 0 (background)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter out small noise regions
            if area < min_area:
                continue
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            defect_regions.append((x, y, x + w, y + h))
        
        log.info(f"Detected {len(defect_regions)} defect region(s) from mask")
        return defect_regions

    except Exception as e:
        log.warning(f"Failed to detect defect regions: {e}")
        return []


def _calculate_crop_box(
    defect_bbox: tuple[int, int, int, int],
    img_size: tuple[int, int],
    min_crop_size: int = 768,
    padding_ratio: float = 0.1,
) -> tuple[int, int, int, int]:
    """
    Calculate crop box that includes the defect region and meets minimum size requirement

    Args:
        defect_bbox: Defect bounding box (x_min, y_min, x_max, y_max)
        img_size: Original image size (width, height)
        min_crop_size: Minimum crop size in pixels (default: 768)
        padding_ratio: Additional padding around defect as ratio of defect size (default: 0.1)

    Returns:
        Crop box (x_min, y_min, x_max, y_max) adjusted for boundaries
    """
    x_min, y_min, x_max, y_max = defect_bbox
    img_width, img_height = img_size

    # Calculate defect dimensions
    defect_width = x_max - x_min
    defect_height = y_max - y_min

    # Add padding around defect
    padding_w = int(defect_width * padding_ratio)
    padding_h = int(defect_height * padding_ratio)
    
    x_min = max(0, x_min - padding_w)
    y_min = max(0, y_min - padding_h)
    x_max = min(img_width, x_max + padding_w)
    y_max = min(img_height, y_max + padding_h)

    # Calculate center point
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    # Determine crop dimensions (at least min_crop_size)
    crop_width = max(x_max - x_min, min_crop_size)
    crop_height = max(y_max - y_min, min_crop_size)

    # If original image is smaller than min_crop_size, use original size
    crop_width = min(crop_width, img_width)
    crop_height = min(crop_height, img_height)

    # Calculate new crop box centered on defect
    new_x_min = center_x - crop_width // 2
    new_x_max = center_x + crop_width // 2
    new_y_min = center_y - crop_height // 2
    new_y_max = center_y + crop_height // 2

    # Adjust if crop box exceeds image boundaries
    if new_x_min < 0:
        new_x_max = min(new_x_max - new_x_min, img_width)
        new_x_min = 0
    if new_x_max > img_width:
        new_x_min = max(0, new_x_min - (new_x_max - img_width))
        new_x_max = img_width

    if new_y_min < 0:
        new_y_max = min(new_y_max - new_y_min, img_height)
        new_y_min = 0
    if new_y_max > img_height:
        new_y_min = max(0, new_y_min - (new_y_max - img_height))
        new_y_max = img_height

    return (new_x_min, new_y_min, new_x_max, new_y_max)


def _crop_image_and_mask(
    img_bytes: bytes,
    mask_bytes: bytes,
    crop_box: tuple[int, int, int, int],
    compress: bool = True,
    quality: int = 85,
) -> tuple[bytes, bytes]:
    """
    Crop both image and mask using the same crop box

    Args:
        img_bytes: Original image bytes
        mask_bytes: Mask image bytes
        crop_box: Crop box (x_min, y_min, x_max, y_max)
        compress: Whether to compress the cropped image
        quality: JPEG quality for compression

    Returns:
        Tuple of (cropped_image_bytes, cropped_mask_bytes)
    """
    try:
        # Load images
        img = Image.open(BytesIO(img_bytes))
        mask = Image.open(BytesIO(mask_bytes))

        # Crop both images
        cropped_img = img.crop(crop_box)
        cropped_mask = mask.crop(crop_box)

        # Convert and compress main image
        if cropped_img.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", cropped_img.size, (255, 255, 255))
            if cropped_img.mode == "P":
                cropped_img = cropped_img.convert("RGBA")
            background.paste(cropped_img, mask=cropped_img.split()[-1] if cropped_img.mode in ("RGBA", "LA") else None)
            cropped_img = background
        elif cropped_img.mode != "RGB":
            cropped_img = cropped_img.convert("RGB")

        # Save cropped main image
        img_buffer = BytesIO()
        if compress:
            cropped_img.save(img_buffer, format="JPEG", quality=quality, optimize=True)
        else:
            cropped_img.save(img_buffer, format="PNG")
        img_bytes_out = img_buffer.getvalue()

        # Save cropped mask (always PNG to preserve quality)
        if cropped_mask.mode not in ("L", "1"):
            cropped_mask = cropped_mask.convert("L")
        
        mask_buffer = BytesIO()
        cropped_mask.save(mask_buffer, format="PNG")
        mask_bytes_out = mask_buffer.getvalue()

        return img_bytes_out, mask_bytes_out

    except Exception as e:
        log.error(f"Failed to crop images: {e}")
        raise


def _build_good_image_map(items: list[dict[str, Any]], dataset_root: str) -> dict[str, str]:
    """
    Build a mapping from class name to a good image path.
    For each class, select the first good image found.

    Args:
        items: List of dataset items
        dataset_root: Root directory of dataset

    Returns:
        Dictionary mapping class name to good image path
    """
    good_image_map: dict[str, str] = {}

    for item in items:
        clsname = item.get("clsname")
        label = item.get("label")

        # Skip if not a good sample or class already has a good image
        if label != 0 or not clsname or clsname in good_image_map:
            continue

        # Try to resolve image path
        img_path = _resolve_image_path(item, dataset_root)
        if img_path and os.path.exists(img_path):
            good_image_map[clsname] = img_path
            log.info(f"Selected good image for class '{clsname}': {img_path}")

    return good_image_map


def _reward_model_value(item: dict[str, Any]) -> Any:
    """Create reward model value"""
    answer = "Yes. There has been a defect detected." if item.get("label") == 1 else "No. There is no defect detected."
    reward_value = {
        "style": "model",
        "ground_truth": {"answer": answer},
    }
    return reward_value


def convert(
    input_jsonl: str,
    dataset_root: str,
    output_path: str,
    output_format: str = "parquet",
    limit: Optional[int] = None,
    compress_images: bool = True,
    image_quality: int = 85,
    max_image_size: Optional[tuple[int, int]] = None,
    crop_enabled: bool = True,
    min_crop_size: int = 768,
    min_defect_area: int = 100,
    crop_padding_ratio: float = 0.1,
) -> tuple[pd.DataFrame, list[Record]]:
    rows: list[Record] = []
    items = _read_jsonl(input_jsonl)
    total_items = len(items)
    
    if limit is not None and limit > 0:
        items = items[:limit]

    # Build good image map for all classes
    log.info("Building good image map for all classes...")
    good_image_map = _build_good_image_map(items, dataset_root)
    log.info(f"Found good images for {len(good_image_map)} classes")

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

        # Read mask image if exists (for defective samples)
        mask_bytes = None
        defect_regions = []
        if item.get("label") == 1:  # Defective sample
            mask_path = _resolve_mask_path(item, dataset_root)
            if mask_path:
                mask_bytes = _read_mask_image_bytes(mask_path, max_size=max_image_size)
                if mask_bytes is None:
                    log.warning(f"[{idx}] Failed to read mask image: {mask_path}")
                elif crop_enabled:
                    # Detect defect regions from mask
                    defect_regions = _detect_defect_regions(mask_bytes, min_area=min_defect_area)
                    if not defect_regions:
                        log.warning(f"[{idx}] No defect regions detected in mask, will process as single image")

        # Get good reference image for this class
        clsname = item.get("clsname")
        good_image_bytes = None
        if clsname and clsname in good_image_map:
            good_img_path = good_image_map[clsname]
            try:
                good_image_bytes = _read_image_bytes(
                    good_img_path,
                    compress=compress_images,
                    quality=image_quality,
                    max_size=max_image_size,
                )
            except Exception as e:
                log.warning(f"[{idx}] Failed to read good reference image {good_img_path}: {e}")

        # Determine if we should crop multiple regions
        should_crop_multiple = (
            crop_enabled 
            and item.get("label") == 1 
            and mask_bytes is not None 
            and len(defect_regions) > 0
        )

        # Get image size for crop calculation
        img_size = None
        if should_crop_multiple:
            try:
                with Image.open(BytesIO(img_bytes)) as img:
                    img_size = img.size  # (width, height)
            except Exception as e:
                log.warning(f"[{idx}] Failed to get image size: {e}")
                should_crop_multiple = False

        if should_crop_multiple and img_size:
            # Process each defect region as a separate record
            total_crops = len(defect_regions)
            log.info(f"[{idx}] Cropping {total_crops} defect region(s) from image")
            
            for crop_idx, defect_bbox in enumerate(defect_regions):
                try:
                    # Calculate crop box
                    crop_box = _calculate_crop_box(
                        defect_bbox, 
                        img_size, 
                        min_crop_size=min_crop_size,
                        padding_ratio=crop_padding_ratio,
                    )
                    
                    # Crop both image and mask
                    cropped_img_bytes, cropped_mask_bytes = _crop_image_and_mask(
                        img_bytes,
                        mask_bytes,
                        crop_box,
                        compress=compress_images,
                        quality=image_quality,
                    )
                    
                    # Get random instruction prompt for data diversity
                    selected_instruction_prompt, prompt_index = get_random_instruction_prompt()

                    prompt = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": selected_instruction_prompt},
                    ]
                    images = [{"bytes": cropped_img_bytes}]

                    reward_model = _reward_model_value(item)

                    crop_width = crop_box[2] - crop_box[0]
                    crop_height = crop_box[3] - crop_box[1]

                    extra_info = {
                        "answer": reward_model["ground_truth"],
                        "question": selected_instruction_prompt,
                        "prompt_variant_index": prompt_index,
                        "clsname": item.get("clsname"),
                        "label": item.get("label"),
                        "type": item.get("label_name"),
                        "index": write_index,
                        "is_cropped": True,
                        "crop_index": crop_idx,
                        "total_crops": total_crops,
                        "crop_bbox": list(crop_box),
                        "crop_size": [crop_width, crop_height],
                        "original_image_size": list(img_size),
                        "defect_bbox": list(defect_bbox),
                    }

                    # Add cropped mask image to extra_info
                    extra_info["mask_image"] = cropped_mask_bytes

                    # Add good reference image to extra_info if available
                    if good_image_bytes is not None:
                        extra_info["good_reference_image"] = good_image_bytes

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
                    
                except Exception as e:
                    log.warning(f"[{idx}] Failed to crop defect region {crop_idx}: {e}")
                    continue
        else:
            # Process as single image (no crop or good sample)
            # Get random instruction prompt for data diversity
            selected_instruction_prompt, prompt_index = get_random_instruction_prompt()

            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": selected_instruction_prompt},
            ]
            images = [{"bytes": img_bytes}]

            reward_model = _reward_model_value(item)

            extra_info = {
                "answer": reward_model["ground_truth"],
                "question": selected_instruction_prompt,
                "prompt_variant_index": prompt_index,
                "clsname": item.get("clsname"),
                "label": item.get("label"),
                "type": item.get("label_name"),
                "index": write_index,
                "is_cropped": False,
            }

            # Add mask image to extra_info if available
            if mask_bytes is not None:
                extra_info["mask_image"] = mask_bytes

            # Add good reference image to extra_info if available
            if good_image_bytes is not None:
                extra_info["good_reference_image"] = good_image_bytes

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

        # Log progress every 100 items
        if (write_index % 100) == 0:
            log.info(f"Processed {write_index} items...")

    # Build DataFrame
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
                
                # Serialize mask_image and good_reference_image in extra_info if present
                if "extra_info" in row_dict:
                    extra_info = row_dict["extra_info"]
                    if "mask_image" in extra_info and isinstance(extra_info["mask_image"], bytes):
                        extra_info["mask_image"] = base64.b64encode(extra_info["mask_image"]).decode("ascii")
                    if "good_reference_image" in extra_info and isinstance(extra_info["good_reference_image"], bytes):
                        extra_info["good_reference_image"] = base64.b64encode(extra_info["good_reference_image"]).decode("ascii")
                
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

    # Expected keys in the root of TOML
    input_path = cfg.get("input", "data.jsonl")
    dataset_root = cfg.get("root", ".")
    output_path = cfg.get("output", "verl_dataset.parquet")
    output_format = cfg.get("format", "parquet")
    limit = cfg.get("limit", None)

    # Image compression related config
    compress_images = cfg.get("compress_images", True)
    image_quality = cfg.get("image_quality", 85)
    max_image_size = cfg.get("max_image_size", None)

    # Crop related config
    crop_config = cfg.get("crop", {})
    crop_enabled = crop_config.get("enabled", True)
    min_crop_size = crop_config.get("min_crop_size", 768)
    min_defect_area = crop_config.get("min_defect_area", 100)
    crop_padding_ratio = crop_config.get("padding_ratio", 0.1)

    # Convert max_image_size from list to tuple if needed
    if isinstance(max_image_size, list) and len(max_image_size) == 2:
        max_image_size = tuple(max_image_size)
    elif max_image_size is not None:
        log.warning(f"Invalid max_image_size format: {max_image_size}, ignoring")
        max_image_size = None

    if not os.path.exists(input_path):
        log.error(f"Input file not found: {input_path}")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    log.info("=" * 60)
    log.info("Configuration:")
    log.info(f"  Input: {input_path}")
    log.info(f"  Dataset Root: {dataset_root}")
    log.info(f"  Output: {output_path}")
    log.info(f"  Format: {output_format}")
    log.info(f"  Limit: {limit if limit else 'None'}")
    log.info(f"  Compress Images: {compress_images}")
    log.info(f"  Image Quality: {image_quality}")
    log.info(f"  Max Image Size: {max_image_size}")
    log.info(f"  Crop Enabled: {crop_enabled}")
    log.info(f"  Min Crop Size: {min_crop_size}")
    log.info(f"  Min Defect Area: {min_defect_area}")
    log.info(f"  Crop Padding Ratio: {crop_padding_ratio}")
    log.info("=" * 60)

    convert(
        input_jsonl=input_path,
        dataset_root=dataset_root,
        output_path=output_path,
        output_format=output_format,
        limit=limit,
        compress_images=compress_images,
        image_quality=image_quality,
        max_image_size=max_image_size,
        crop_enabled=crop_enabled,
        min_crop_size=min_crop_size,
        min_defect_area=min_defect_area,
        crop_padding_ratio=crop_padding_ratio,
    )


if __name__ == "__main__":
    main()

