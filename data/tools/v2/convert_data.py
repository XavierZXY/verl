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

# Category-specific prior knowledge and instruction prompts
CATEGORY_KNOWLEDGE: dict[str, dict[str, Any]] = {
    "bottle": {
        "description": "Glass or plastic bottles used in beverage packaging",
        "common_defects": ["broken/cracks", "contamination", "surface damage"],
        "inspection_focus": "surface integrity, cracks, foreign particles",
        "prompts": [
            "<image>.\nInspect this bottle for any cracks, breaks, or contamination. Check the surface carefully for structural damage. Answer 'no' if intact, 'yes' if defective.",
            "<image>.\nExamine this bottle image for manufacturing defects such as cracks, broken parts, or contamination on the surface. Reply 'no' if normal, 'yes' if defects found.",
            "<image>.\nAnalyze the bottle for structural integrity. Look for cracks, breaks, or any foreign particles. Answer 'no' for good quality, 'yes' for defects.",
        ]
    },
    "cable": {
        "description": "Electrical cables with wires and insulation",
        "common_defects": ["bent wire", "cable swap", "cut insulation", "missing components", "poke damage"],
        "inspection_focus": "wire alignment, insulation integrity, cable configuration",
        "prompts": [
            "<image>.\nCheck this cable for bent wires, damaged insulation, or missing components. Verify proper cable configuration. Answer 'no' if correct, 'yes' if defective.",
            "<image>.\nInspect the cable for wire damage, insulation cuts, or incorrect assembly. Look for bent, missing, or swapped components. Reply 'no' if normal, 'yes' if defects present.",
            "<image>.\nExamine this cable carefully for any wire bending, insulation damage, or missing parts. Check if all components are properly assembled. Answer 'no' for good, 'yes' for defects.",
        ]
    },
    "capsule": {
        "description": "Pharmaceutical capsules or pill containers",
        "common_defects": ["crack", "faulty imprint", "poke damage", "scratch", "squeeze deformation"],
        "inspection_focus": "surface smoothness, imprint quality, structural integrity",
        "prompts": [
            "<image>.\nInspect this capsule for cracks, scratches, faulty imprints, or deformation. Check the surface and printing quality. Answer 'no' if perfect, 'yes' if defective.",
            "<image>.\nExamine the capsule for any cracks, poke marks, scratches, or squeeze damage. Verify the imprint is clear and correct. Reply 'no' if normal, 'yes' if defects found.",
            "<image>.\nAnalyze this capsule for surface defects including cracks, scratches, or deformation. Check if the imprint is properly applied. Answer 'no' for good quality, 'yes' for defects.",
        ]
    },
    "carpet": {
        "description": "Textile carpets or fabric materials",
        "common_defects": ["color variation", "cuts", "holes", "metal contamination", "thread issues"],
        "inspection_focus": "color uniformity, surface integrity, foreign objects",
        "prompts": [
            "<image>.\nInspect this carpet for color variations, cuts, holes, or foreign objects like metal contamination. Check thread quality. Answer 'no' if normal, 'yes' if defective.",
            "<image>.\nExamine the carpet surface for any cuts, holes, discoloration, or thread defects. Look for metal particles or other contamination. Reply 'no' if clean, 'yes' if defects present.",
            "<image>.\nAnalyze this carpet for manufacturing defects such as color inconsistency, surface damage, holes, or thread issues. Answer 'no' for good quality, 'yes' for defects.",
        ]
    },
    "grid": {
        "description": "Metal or plastic grid patterns",
        "common_defects": ["bent", "broken", "glue residue", "metal contamination", "thread damage"],
        "inspection_focus": "grid alignment, structural integrity, surface cleanliness",
        "prompts": [
            "<image>.\nCheck this grid for bent or broken sections, glue residue, or contamination. Verify the grid pattern is uniform and intact. Answer 'no' if correct, 'yes' if defective.",
            "<image>.\nInspect the grid structure for any bending, breaks, or foreign materials. Look for glue marks or metal contamination. Reply 'no' if normal, 'yes' if defects found.",
            "<image>.\nExamine this grid for structural defects, including bent or broken sections, and surface contamination. Answer 'no' for good quality, 'yes' for defects.",
        ]
    },
    "hazelnut": {
        "description": "Hazelnut nuts for quality inspection",
        "common_defects": ["crack", "cut", "hole", "print marks"],
        "inspection_focus": "shell integrity, surface damage, holes",
        "prompts": [
            "<image>.\nInspect this hazelnut for cracks, cuts, holes, or printing defects. Check the shell surface carefully. Answer 'no' if intact, 'yes' if defective.",
            "<image>.\nExamine the hazelnut shell for any cracks, cuts, holes, or surface marks. Look for structural damage. Reply 'no' if normal, 'yes' if defects present.",
            "<image>.\nAnalyze this hazelnut for shell defects including cracks, cuts, or holes. Check for any abnormal marks. Answer 'no' for good quality, 'yes' for defects.",
        ]
    },
    "leather": {
        "description": "Leather material for manufacturing",
        "common_defects": ["color variation", "cuts", "folds", "glue marks", "poke damage"],
        "inspection_focus": "surface smoothness, color uniformity, structural integrity",
        "prompts": [
            "<image>.\nInspect this leather for color variations, cuts, folds, or glue marks. Check for poke damage or surface irregularities. Answer 'no' if smooth, 'yes' if defective.",
            "<image>.\nExamine the leather surface for any cuts, folds, discoloration, glue residue, or poke marks. Look for texture abnormalities. Reply 'no' if normal, 'yes' if defects found.",
            "<image>.\nAnalyze this leather material for defects such as color inconsistency, cuts, folds, or surface damage. Answer 'no' for good quality, 'yes' for defects.",
        ]
    },
    "pill": {
        "description": "Pharmaceutical pills or tablets",
        "common_defects": ["scratches", "surface damage"],
        "inspection_focus": "surface smoothness, coating integrity",
        "prompts": [
            "<image>.\nInspect this pill for scratches or surface damage. Check the coating and overall appearance. Answer 'no' if smooth, 'yes' if defective.",
            "<image>.\nExamine the pill surface for any scratches, chips, or coating defects. Look for irregularities. Reply 'no' if normal, 'yes' if defects present.",
            "<image>.\nAnalyze this pill for surface defects including scratches or damage to the coating. Answer 'no' for good quality, 'yes' for defects.",
        ]
    },
    "screw": {
        "description": "Metal screws or fasteners",
        "common_defects": ["manipulated front", "head scratches", "neck scratches", "thread damage"],
        "inspection_focus": "thread integrity, head condition, surface scratches",
        "prompts": [
            "<image>.\nInspect this screw for scratches on the head or neck, thread damage, or front manipulation. Check all screw components. Answer 'no' if intact, 'yes' if defective.",
            "<image>.\nExamine the screw for any scratches, thread defects, or manipulation marks. Look at the head, neck, and threaded portions. Reply 'no' if normal, 'yes' if defects found.",
            "<image>.\nAnalyze this screw for manufacturing defects including scratches, thread damage, or deformation. Answer 'no' for good quality, 'yes' for defects.",
        ]
    },
    "tile": {
        "description": "Ceramic or floor tiles",
        "common_defects": ["cracks", "glue strips", "gray strokes", "oil stains", "rough surface"],
        "inspection_focus": "surface smoothness, cracks, stains, texture uniformity",
        "prompts": [
            "<image>.\nInspect this tile for cracks, glue residue, stains, or surface roughness. Check for gray strokes or oil marks. Answer 'no' if smooth, 'yes' if defective.",
            "<image>.\nExamine the tile surface for any cracks, glue strips, discoloration, oil stains, or rough patches. Look for texture irregularities. Reply 'no' if normal, 'yes' if defects present.",
            "<image>.\nAnalyze this tile for defects such as cracks, surface contamination, rough areas, or glue marks. Answer 'no' for good quality, 'yes' for defects.",
        ]
    },
    "toothbrush": {
        "description": "Toothbrush products",
        "common_defects": ["defective bristles", "handle damage", "assembly issues"],
        "inspection_focus": "bristle quality, handle integrity, overall assembly",
        "prompts": [
            "<image>.\nInspect this toothbrush for defective bristles, handle damage, or assembly issues. Check the overall product quality. Answer 'no' if correct, 'yes' if defective.",
            "<image>.\nExamine the toothbrush for any defects in bristles, handle, or assembly. Look for manufacturing irregularities. Reply 'no' if normal, 'yes' if defects found.",
            "<image>.\nAnalyze this toothbrush for quality issues including bristle defects, handle damage, or assembly problems. Answer 'no' for good quality, 'yes' for defects.",
        ]
    },
    "transistor": {
        "description": "Electronic transistor components",
        "common_defects": ["bent leads", "cut leads", "damaged case", "misplacement"],
        "inspection_focus": "lead alignment, case integrity, component positioning",
        "prompts": [
            "<image>.\nInspect this transistor for bent or cut leads, damaged case, or misplacement. Check lead alignment and case condition. Answer 'no' if correct, 'yes' if defective.",
            "<image>.\nExamine the transistor for any lead damage, case cracks, or incorrect positioning. Look for bent, cut, or misaligned components. Reply 'no' if normal, 'yes' if defects found.",
            "<image>.\nAnalyze this transistor for manufacturing defects including lead issues, case damage, or placement errors. Answer 'no' for good quality, 'yes' for defects.",
        ]
    },
    "wood": {
        "description": "Wood material or wooden products",
        "common_defects": ["color variation", "combined defects", "holes", "liquid stains", "scratches"],
        "inspection_focus": "surface smoothness, color uniformity, holes, stains",
        "prompts": [
            "<image>.\nInspect this wood for color variations, holes, scratches, or liquid stains. Check the surface texture and uniformity. Answer 'no' if smooth, 'yes' if defective.",
            "<image>.\nExamine the wood surface for any holes, scratches, discoloration, or staining. Look for texture irregularities or combined defects. Reply 'no' if normal, 'yes' if defects present.",
            "<image>.\nAnalyze this wood material for defects such as color inconsistency, holes, scratches, or liquid damage. Answer 'no' for good quality, 'yes' for defects.",
        ]
    },
}

# Generic fallback prompts for unknown categories
GENERIC_INSTRUCTION_PROMPTS: list[str] = [
    "<image>.\nAnalyze this image for defects. If there is no defect, answer 'no'. If there is defect, answer 'yes'.",
    "<image>.\nExamine this image carefully and determine whether any defects are present. Respond with 'no' if defect-free, 'yes' if defects are found.",
    "<image>.\nInspect this image for any quality issues or anomalies. Answer 'no' for normal items, 'yes' for defective items.",
    "<image>.\nPlease evaluate this image to identify any manufacturing defects. Reply 'no' if the item is good, 'yes' if there are defects.",
    "<image>.\nLook at this image and assess whether there are any flaws or irregularities. Answer 'no' if perfect, 'yes' if imperfect.",
]


def get_category_specific_prompt(clsname: Optional[str]) -> tuple[str, int]:
    """
    Get a category-specific instruction prompt based on the class name.

    Args:
        clsname: The class name (e.g., 'bottle', 'cable', etc.)

    Returns:
        tuple: (selected_prompt, prompt_index)
    """
    if clsname and clsname in CATEGORY_KNOWLEDGE:
        prompts = CATEGORY_KNOWLEDGE[clsname]["prompts"]
        prompt_index = random.randint(0, len(prompts) - 1)
        return prompts[prompt_index], prompt_index
    else:
        # Fallback to generic prompts
        prompt_index = random.randint(0, len(GENERIC_INSTRUCTION_PROMPTS) - 1)
        return GENERIC_INSTRUCTION_PROMPTS[prompt_index], prompt_index


def get_random_instruction_prompt() -> tuple[str, int]:
    """
    Get a random instruction prompt from the generic variants.
    Kept for backward compatibility.

    Returns:
        tuple: (selected_prompt, prompt_index)
    """
    prompt_index = random.randint(0, len(GENERIC_INSTRUCTION_PROMPTS) - 1)
    return GENERIC_INSTRUCTION_PROMPTS[prompt_index], prompt_index


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
        if item.get("label") == 1:  # Defective sample
            mask_path = _resolve_mask_path(item, dataset_root)
            if mask_path:
                mask_bytes = _read_mask_image_bytes(mask_path, max_size=max_image_size)
                if mask_bytes is None:
                    log.warning(f"[{idx}] Failed to read mask image: {mask_path}")

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

        # Get category-specific instruction prompt based on class name
        clsname = item.get("clsname")
        selected_instruction_prompt, prompt_index = get_category_specific_prompt(clsname)

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": selected_instruction_prompt},
        ]
        images = [{"bytes": img_bytes}]

        reward_model = _reward_model_value(item)

        # Add category knowledge to extra_info if available
        category_info = {}
        if clsname and clsname in CATEGORY_KNOWLEDGE:
            category_info = {
                "category_description": CATEGORY_KNOWLEDGE[clsname]["description"],
                "common_defects": CATEGORY_KNOWLEDGE[clsname]["common_defects"],
                "inspection_focus": CATEGORY_KNOWLEDGE[clsname]["inspection_focus"],
            }

        extra_info = {
            "answer": reward_model["ground_truth"],
            "question": selected_instruction_prompt,
            "prompt_variant_index": prompt_index,
            "clsname": clsname,
            "label": item.get("label"),
            "type": item.get("label_name"),
            "index": write_index,
            "category_knowledge": category_info,
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

