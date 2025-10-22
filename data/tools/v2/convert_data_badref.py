import base64
import json
import logging
import os
import random
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

import pandas as pd
from PIL import Image, ImageDraw
from rich.logging import RichHandler
import numpy as np

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


def _build_defect_reference_map(items: list[dict[str, Any]], dataset_root: str) -> dict[str, tuple[str, str]]:
    """
    Build a mapping from class name to a defect image path and its mask path.
    For each class, select the first defect image with mask found.

    Args:
        items: List of dataset items
        dataset_root: Root directory of dataset

    Returns:
        Dictionary mapping class name to (defect_image_path, mask_path) tuple
    """
    defect_reference_map: dict[str, tuple[str, str]] = {}

    for item in items:
        clsname = item.get("clsname")
        label = item.get("label")

        # Skip if not a defect sample or class already has a reference image
        if label != 1 or not clsname or clsname in defect_reference_map:
            continue

        # Try to resolve image path and mask path
        img_path = _resolve_image_path(item, dataset_root)
        mask_path = _resolve_mask_path(item, dataset_root)
        
        if img_path and mask_path and os.path.exists(img_path) and os.path.exists(mask_path):
            defect_reference_map[clsname] = (img_path, mask_path)
            log.info(f"Selected defect reference image for class '{clsname}': {img_path} with mask: {mask_path}")

    return defect_reference_map


def _create_annotated_reference_image(
    image_path: str,
    mask_path: str,
    compress: bool = True,
    quality: int = 85,
    max_size: Optional[tuple[int, int]] = None,
    box_color: tuple[int, int, int] = (255, 0, 0),  # Red
    box_width: int = 3,
) -> Optional[bytes]:
    """
    Create an annotated reference image by drawing bounding boxes around defect regions.
    
    Args:
        image_path: Path to the original defect image
        mask_path: Path to the mask image
        compress: Whether to compress the output
        quality: JPEG compression quality
        max_size: Maximum size for resizing
        box_color: RGB color for the bounding box (default: red)
        box_width: Width of the bounding box line
        
    Returns:
        Annotated image bytes or None if failed
    """
    try:
        # Open both image and mask
        with Image.open(image_path) as img, Image.open(mask_path) as mask:
            # Convert image to RGB
            if img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")
            
            # Resize if needed
            if max_size is not None:
                original_size = img.size
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                # Resize mask proportionally
                if img.size != original_size:
                    mask = mask.resize(img.size, Image.Resampling.LANCZOS)
            
            # Convert mask to numpy array for processing
            mask_array = np.array(mask.convert("L"))
            
            # Find contours/bounding boxes from mask
            # Use connected components to find defect regions
            binary_mask = (mask_array > 128).astype(np.uint8)
            
            # Find all non-zero regions
            from scipy import ndimage
            labeled_array, num_features = ndimage.label(binary_mask)
            
            # Create drawing context
            draw = ImageDraw.Draw(img)
            
            # Draw bounding box for each defect region
            for label_id in range(1, num_features + 1):
                # Find pixels belonging to this region
                region_mask = (labeled_array == label_id)
                rows, cols = np.where(region_mask)
                
                if len(rows) > 0 and len(cols) > 0:
                    # Calculate bounding box
                    y_min, y_max = rows.min(), rows.max()
                    x_min, x_max = cols.min(), cols.max()
                    
                    # Draw rectangle
                    draw.rectangle(
                        [(x_min, y_min), (x_max, y_max)],
                        outline=box_color,
                        width=box_width
                    )
            
            # Save to bytes
            buffer = BytesIO()
            if compress:
                img.save(buffer, format="JPEG", quality=quality, optimize=True)
            else:
                img.save(buffer, format="PNG")
            
            return buffer.getvalue()
            
    except Exception as e:
        log.warning(f"Failed to create annotated reference image from {image_path} and {mask_path}: {e}")
        return None


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
) -> tuple[pd.DataFrame, list[Record]]:
    rows: list[Record] = []
    items = _read_jsonl(input_jsonl)
    total_items = len(items)
    
    if limit is not None and limit > 0:
        items = items[:limit]

    # Build defect reference image map for all classes
    log.info("Building defect reference image map for all classes...")
    defect_reference_map = _build_defect_reference_map(items, dataset_root)
    log.info(f"Found defect reference images for {len(defect_reference_map)} classes")

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
        if item.get("label") == 1:  # Defective sample
            mask_path = _resolve_mask_path(item, dataset_root)
            if mask_path:
                mask_bytes = _read_mask_image_bytes(mask_path, max_size=max_image_size)
                if mask_bytes is None:
                    log.warning(f"[{idx}] Failed to read mask image: {mask_path}")

        # Get annotated defect reference image for this class
        clsname = item.get("clsname")
        annotated_reference_bytes = None
        if clsname and clsname in defect_reference_map:
            defect_img_path, defect_mask_path = defect_reference_map[clsname]
            try:
                annotated_reference_bytes = _create_annotated_reference_image(
                    defect_img_path,
                    defect_mask_path,
                    compress=compress_images,
                    quality=image_quality,
                    max_size=max_image_size,
                )
                if annotated_reference_bytes is None:
                    log.warning(f"[{idx}] Failed to create annotated reference image for class '{clsname}'")
            except Exception as e:
                log.warning(f"[{idx}] Failed to create annotated reference image {defect_img_path}: {e}")

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
        }

        # Add mask image to extra_info if available
        if mask_bytes is not None:
            extra_info["mask_image"] = mask_bytes

        # Add annotated defect reference image to extra_info if available
        if annotated_reference_bytes is not None:
            extra_info["reference_image"] = annotated_reference_bytes

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
                
                # Serialize mask_image and reference_image in extra_info if present
                if "extra_info" in row_dict:
                    extra_info = row_dict["extra_info"]
                    if "mask_image" in extra_info and isinstance(extra_info["mask_image"], bytes):
                        extra_info["mask_image"] = base64.b64encode(extra_info["mask_image"]).decode("ascii")
                    if "reference_image" in extra_info and isinstance(extra_info["reference_image"], bytes):
                        extra_info["reference_image"] = base64.b64encode(extra_info["reference_image"]).decode("ascii")
                
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

    convert(
        input_jsonl=input_path,
        dataset_root=dataset_root,
        output_path=output_path,
        output_format=output_format,
        limit=limit,
        compress_images=compress_images,
        image_quality=image_quality,
        max_image_size=max_image_size,
    )


if __name__ == "__main__":
    main()

