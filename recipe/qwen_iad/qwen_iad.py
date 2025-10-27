# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import io
import logging
import math
import os
import random
import re

import requests
from openai import OpenAI
from PIL import Image

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask
from recipe.qwen_iad.qwen_iad_improved_prompt import IAD_SYSTEM_PROMPT
logger = logging.getLogger(__name__)

SYSTEM_PROMPT: str = IAD_SYSTEM_PROMPT
openai_api_key = "EMPTY"
openai_api_base = os.environ.get("LLM_AS_A_JUDGE_BASE", "http://10.1.100.71:18901/v1")

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model_name = ""
# if openai_api_base:
#     try:
#         response = requests.get(f"{openai_api_base}/models")
#         response.raise_for_status()
#         models = response.json()
#         if models.get("data"):
#             model_name = models["data"][0]["id"]
#         else:
#             logger.warning("No models found at the specified API base for reward scoring.")
#     except (requests.exceptions.RequestException, KeyError, IndexError) as e:
#         logger.warning(f"Failed to get model from {openai_api_base}: {e}. Reward scoring will be disabled.")


class CustomRLHFDataset(RLHFDataset):
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        row_dict[self.prompt_key] = [
            {
                "role": "system",
                # We don't need tool description, because custom_chat_template will add it.
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": row_dict[self.prompt_key][1]["content"],
            },
        ]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                images = [Image.open(io.BytesIO(image["bytes"])) for image in row_dict_images]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205  # noqa: E501
                multi_modal_data["image"] = images

            model_inputs = self.processor(text=[raw_prompt], images=images, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        
        # Get good reference image from extra_info if available
        good_reference_image = row_dict.get("extra_info", {}).get("good_reference_image")
        
        # Note: mask_image is kept in extra_info for reward calculation only, not passed to zoom tool
        
        tools_kwargs = {
            "image_zoom_in_tool": {
                "create_kwargs": {
                    "image": images[0],
                },
                # "execute_kwargs": {},
                # "calc_reward_kwargs": {},
                # "release_kwargs": {},
            },
            "image_reference_tool": {
                "create_kwargs": {"good_reference_image": good_reference_image},
                # "execute_kwargs": {},
                # "calc_reward_kwargs": {},
                # "release_kwargs": {},
            }
        }
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["agent_name"] = "tool_agent"
        return row_dict


def _validate_bbox(left, top, right, bottom):
    """
    Validate that a bounding box has positive dimensions.
    
    Args:
        left, top, right, bottom: Bounding box coordinates
        
    Returns:
        bool: True if bbox is valid, False otherwise
    """
    try:
        return right > left and bottom > top
    except Exception as e:
        logger.warning(f"Bbox validation error: {e}")
        return False


def _maybe_resize_bbox(bbox_2d, image_width, image_height, min_dimension=28):
    """
    Clamp, validate, and potentially resize a bounding box.
    
    This function ensures the final bounding box is within image bounds and meets the minimum
    dimension requirements. If the initial box is too small, it attempts to expand it
    from its center. It performs a final check to guarantee the output dimensions are valid.
    
    This implements the same logic used in ImageZoomInTool to ensure consistency between
    the bbox used for IoU calculation and the bbox actually used by the tool.
    
    Args:
        bbox_2d: Bounding box as [left, top, right, bottom]
        image_width: Width of the image
        image_height: Height of the image
        min_dimension: Minimum dimension requirement (default: 28, same as ImageZoomInTool)
        
    Returns:
        A valid bounding box as a list of coordinates, or None if validation fails.
    """
    from math import floor, ceil
    
    left, top, right, bottom = bbox_2d
    
    # 1. Clamp the initial bounding box to the image dimensions.
    left = max(0.0, float(left))
    top = max(0.0, float(top))
    right = min(float(image_width), float(right))
    bottom = min(float(image_height), float(bottom))
    
    # 2. If clamped bbox is invalid, return immediately.
    if not _validate_bbox(left, top, right, bottom):
        return None
    
    current_bbox = [left, top, right, bottom]
    height = bottom - top
    width = right - left
    
    # 3. If the box is too small, attempt to resize it.
    if height < min_dimension or width < min_dimension:
        logger.debug(f"Bbox {width}x{height} is smaller than {min_dimension}, attempting resize.")
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        
        min_dim = min(height, width)
        if min_dim == 0:  # Safeguard for zero-area boxes
            return None
        
        # 1. Calculate the target dimensions to make the smallest side min_dimension.
        ratio = min_dimension / min_dim
        target_width = width * ratio
        target_height = height * ratio
        
        # 2. If the target size is larger than the image, scale it down to fit.
        if target_width > image_width:
            scale_down = image_width / target_width
            target_width = image_width
            target_height *= scale_down
        
        if target_height > image_height:
            scale_down = image_height / target_height
            target_height = image_height
            target_width *= scale_down
        
        # 3. Determine the coordinates for the box centered on the original center.
        new_half_width = target_width / 2.0
        new_half_height = target_height / 2.0
        new_left = center_x - new_half_width
        new_top = center_y - new_half_height
        
        # 4. Shift the box if it extends beyond the image boundaries to keep its size.
        if new_left < 0:
            new_left = 0
        if new_top < 0:
            new_top = 0
        if new_left + target_width > image_width:
            new_left = image_width - target_width
        if new_top + target_height > image_height:
            new_top = image_height - target_height
        
        new_right = new_left + target_width
        new_bottom = new_top + target_height
        
        # Use floor and ceil for final integer coordinates.
        current_bbox = [floor(new_left), floor(new_top), ceil(new_right), ceil(new_bottom)]
    
    # 4. Final validation on the resulting bounding box (either original or resized).
    final_left, final_top, final_right, final_bottom = current_bbox
    if not _validate_bbox(final_left, final_top, final_right, final_bottom):
        logger.warning(f"Final bbox is invalid after processing: {current_bbox}")
        return None
    
    final_height = floor(final_bottom) - floor(final_top)
    final_width = floor(final_right) - floor(final_left)
    
    if final_height < min_dimension or final_width < min_dimension:
        logger.warning(
            f"Final bbox size ({final_width}x{final_height}) are still smaller than minimum ({min_dimension}). "
            f"Original bbox: {bbox_2d}, original image size: {image_width}x{image_height}"
        )
        return None
    
    return current_bbox


def _extract_last_zoom_bbox(solution_str):
    """
    Extract the LAST image_zoom_in_tool bbox call from the solution.
    
    Args:
        solution_str: The model's solution string
        
    Returns:
        Last bbox coordinates [x1, y1, x2, y2] or None if no zoom call found
    """
    import json
    
    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    tool_calls = re.findall(tool_call_pattern, solution_str, re.DOTALL)
    
    last_zoom_bbox = None
    for tool_call_content in tool_calls:
        try:
            tool_data = json.loads(tool_call_content.strip())
            
            # Handle both list and dict formats
            if isinstance(tool_data, dict):
                tool_name = tool_data.get("tool_name") or tool_data.get("name")
                if tool_name == "image_zoom_in_tool":
                    args = tool_data.get("parameters", {}) or tool_data.get("arguments", {})
                    bbox = args.get("bbox_2d") or args.get("bbox2d")
                    if bbox and isinstance(bbox, list) and len(bbox) == 4:
                        last_zoom_bbox = [float(v) for v in bbox]
            elif isinstance(tool_data, list):
                for item in tool_data:
                    if isinstance(item, dict):
                        tool_name = item.get("tool_name") or item.get("name")
                        if tool_name == "image_zoom_in_tool":
                            args = item.get("parameters", {}) or item.get("arguments", {})
                            bbox = args.get("bbox_2d") or args.get("bbox2d")
                            if bbox and isinstance(bbox, list) and len(bbox) == 4:
                                last_zoom_bbox = [float(v) for v in bbox]
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    
    return last_zoom_bbox


def _transform_bbox_with_offsets(bbox, zoom_offsets):
    """
    Transform a bbox from a nested coordinate system to the original coordinate system.
    
    When zoom is called multiple times, each zoom crops the previous image, creating nested
    coordinate systems. This function applies cumulative offsets to transform a bbox back
    to the original image's coordinate system.
    
    Args:
        bbox: Bbox in the final zoomed coordinate system [x1, y1, x2, y2]
        zoom_offsets: List of (x_offset, y_offset) tuples from each zoom operation
        
    Returns:
        Bbox in original coordinate system [x1, y1, x2, y2]
    """
    if not zoom_offsets:
        return bbox
    
    # Accumulate all offsets
    cumulative_x = sum(offset[0] for offset in zoom_offsets)
    cumulative_y = sum(offset[1] for offset in zoom_offsets)
    
    # Transform bbox by adding cumulative offsets
    transformed_bbox = [
        bbox[0] + cumulative_x,
        bbox[1] + cumulative_y,
        bbox[2] + cumulative_x,
        bbox[3] + cumulative_y,
    ]
    
    logger.debug(f"Transformed bbox {bbox} with {len(zoom_offsets)} offsets to {transformed_bbox}")
    return transformed_bbox


def _compute_mask_iou(pred_boxes, gt_mask_bytes, max_pred_boxes=3):
    """
    Compute IoU between predicted bboxes and ground truth mask image.
    
    This function is called only when gt_mask_bytes is confirmed to be valid bytes data.
    
    Args:
        pred_boxes: List of predicted bounding boxes in [x1, y1, x2, y2] format
        gt_mask_bytes: Bytes data of ground truth mask image (must not be None)
        max_pred_boxes: Maximum number of predicted boxes to consider
        
    Returns:
        float: IoU score between 0 and 1
    """
    from io import BytesIO
    import numpy as np
    
    try:
        from PIL import Image
    except ImportError:
        logger.error("PIL not available, cannot compute mask IoU")
        return 0.0
    
    # Limit predicted boxes
    if len(pred_boxes) > max_pred_boxes:
        pred_boxes = pred_boxes[:max_pred_boxes]
    
    # Handle empty prediction case
    if len(pred_boxes) == 0:
        # No predictions but there's a ground truth mask = penalty for missing detection
        logger.warning("No predicted boxes but ground truth mask exists")
        return 0.0
    
    try:
        # Load ground truth mask image
        gt_mask_img = Image.open(BytesIO(gt_mask_bytes))
        
        # Convert to grayscale if needed
        if gt_mask_img.mode != 'L':
            gt_mask_img = gt_mask_img.convert('L')
        
        # Convert to numpy array and binarize
        gt_mask = np.array(gt_mask_img)
        gt_mask_binary = (gt_mask > 127).astype(np.uint8)
        
        # Get image dimensions
        height, width = gt_mask_binary.shape
        
        # Apply resize bbox logic to match what zoom tool actually uses
        resized_pred_boxes = []
        for box in pred_boxes:
            resized_box = _maybe_resize_bbox(box, image_width=width, image_height=height)
            if resized_box is not None:
                resized_pred_boxes.append(resized_box)
            else:
                logger.warning(f"Bbox {box} failed resize validation, skipping")
        
        if len(resized_pred_boxes) == 0:
            logger.warning("No valid resized boxes after resize validation")
            return 0.0
        
        # Create prediction mask from resized bboxes
        pred_mask = np.zeros((height, width), dtype=np.uint8)
        
        for box in resized_pred_boxes:
            x1, y1, x2, y2 = box
            # Convert to integer coordinates and clip to image bounds
            x1 = int(max(0, min(width - 1, x1)))
            y1 = int(max(0, min(height - 1, y1)))
            x2 = int(max(0, min(width, x2)))
            y2 = int(max(0, min(height, y2)))
            
            # Fill the bbox region in prediction mask
            if x2 > x1 and y2 > y1:
                pred_mask[y1:y2, x1:x2] = 1
        
        # Compute IoU between prediction mask and ground truth mask
        # Convert to Python int to ensure JSON serialization compatibility
        intersection = np.logical_and(pred_mask, gt_mask_binary).sum()
        union = np.logical_or(pred_mask, gt_mask_binary).sum()
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        iou = float(intersection) / float(union)
        
        # Apply size penalty if prediction area is much larger than ground truth
        pred_area = float(pred_mask.sum())
        gt_area = float(gt_mask_binary.sum())
        
        size_penalty = 1.0
        if gt_area > 0:
            area_ratio = pred_area / gt_area
            if area_ratio > 100.0:  # Prediction area is 20x larger than GT
                import math
                size_penalty = 1.0 / (1.0 + math.log(area_ratio / 3.0))
                logger.debug(f"Size penalty applied: area_ratio={area_ratio:.2f}, penalty={size_penalty:.4f}")
        
        iou_with_penalty = iou * size_penalty
        
        # print intersection and union
        # print(f"[DEBUG IOU]------------ Intersection: {intersection}, Union: {union}, IoU: {iou:.4f}, "
        #       f"Pred_area: {pred_area:.0f}, GT_area: {gt_area:.0f}, Size_penalty: {size_penalty:.4f}, "
        #       f"Final_IoU: {iou_with_penalty:.4f} ------------")
        
        # Save comparison image of pred_mask and gt_mask_binary
        # try:
        #     import time
        #     os.makedirs("logs/compare_iou", exist_ok=True)
            
        #     # Convert masks to binary images (0 or 255)
        #     pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8), mode='L')
        #     gt_img = Image.fromarray((gt_mask_binary * 255).astype(np.uint8), mode='L')
            
        #     # Create a combined image with pred_mask on the left and gt_mask on the right
        #     combined_width = width * 2
        #     combined_height = height
        #     combined_img = Image.new('L', (combined_width, combined_height))
            
        #     # Paste pred_mask on the left and gt_mask on the right
        #     combined_img.paste(pred_img, (0, 0))
        #     combined_img.paste(gt_img, (width, 0))
            
        #     # Generate unique filename with timestamp
        #     timestamp = int(time.time() * 1000000)  # microseconds for uniqueness
        #     filename = f"logs/compare_iou/mask_comparison_{float(intersection) / float(union)}_{timestamp}.png"
        #     combined_img.save(filename)
        #     logger.info(f"Saved mask comparison image to {filename}")
        # except Exception as e:
        #     logger.warning(f"Failed to save mask comparison image: {e}")
        
        
        # logger.debug(f"Mask IoU computed: original_boxes={len(pred_boxes)}, resized_boxes={len(resized_pred_boxes)}, iou={iou:.4f}")
        
        return iou_with_penalty
        
    except Exception as e:
        logger.error(f"Failed to compute mask IoU: {e}")
        return 0.0


def _extract_ground_truth_answer(ground_truth, extra_info):
    """Return the textual ground truth answer if available."""
    answer = None

    if isinstance(ground_truth, dict):
        answer = ground_truth.get("answer")
    elif isinstance(ground_truth, str):
        answer = ground_truth
    elif ground_truth is not None:
        answer = str(ground_truth)

    if answer is None and extra_info is not None:
        answer_info = extra_info.get("answer")
        if isinstance(answer_info, dict):
            answer = answer_info.get("answer")
        elif isinstance(answer_info, str):
            answer = answer_info

    return (answer or "").strip()


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Compute reward score for defect detection task.

    The score consists of three components:
    1. Format reward: Hard reward (1.0 if no errors, 0.0 if any errors)
    2. Answer correctness reward: Hard reward (1.0 if match, 0.0 otherwise)
    3. Tool reward: Combination of tool usage (0.0 or 1.0) and bbox IoU
       - Only valid when tool_call and tool_response counts match

    Args:
        data_source: Source of the data (not used currently)
        solution_str: Model's solution string
        ground_truth: Ground truth answer (can be dict or string)
        extra_info: Additional information including question, bboxes, etc.

    Returns:
        dict: Dictionary containing score and reward components
    """
    import json

    # ============================================================================
    # 1. FORMAT REWARD: Check tag pairing
    # ============================================================================
    is_format_error = False
    tags_to_check = [
        ("<think>", "</think>"),
        ("<tool_call>", "</tool_call>"),
        ("<tool_response>", "</tool_response>"),
        ("<answer>", "</answer>"),
        ("<location>", "</location>"),
        ("<type>", "</type>"),
    ]
    
    for open_tag, close_tag in tags_to_check:
        if solution_str.count(open_tag) != solution_str.count(close_tag):
            is_format_error = True
            break
    
    format_reward = 0.0 if is_format_error else 1.0
    
    # ============================================================================
    # 2. ANSWER REWARD: Check answer correctness
    # ============================================================================
    answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    answer_text = answer_match.group(1).strip() if answer_match else ""
    ground_truth_answer = _extract_ground_truth_answer(ground_truth, extra_info)
    
    acc_reward = 0.0
    if answer_text:
        answer_norm = answer_text.strip().lower()
        gt_norm = ground_truth_answer.strip().lower()
        
        if "yes" in gt_norm:
            acc_reward = 1.0 if "yes" in answer_norm else 0.0
        elif "no" in gt_norm:
            acc_reward = 1.0 if "no" in answer_norm else 0.0
        else:
            acc_reward = 1.0 if answer_norm == gt_norm else -1.0
    
    # ============================================================================
    # 3. TOOL REWARD: Only valid when tool_call and tool_response counts match
    # ============================================================================
    count_tool_call = solution_str.count("<tool_call>")
    count_tool_response = solution_str.count("<tool_response>")
    has_valid_tool_usage = (count_tool_call == count_tool_response) and (count_tool_call > 0)
    
    # Initialize all tool-related rewards
    bbox_iou = 0.0
    bbox_iou_transformed = 0.0
    tools_used = set()
    zoom_call_count = 0
    tool_diversity_bonus = 0.0
    zoom_count_reward = 0.0
    tool_reward = 0.0
    tool_valid_reward = 0.0
    
    if not has_valid_tool_usage:
        # Tool call/response counts don't match or no tool usage
        logger.debug(f"Invalid tool usage: tool_call={count_tool_call}, tool_response={count_tool_response}")
    else:
        # Valid tool usage: extract and analyze tool calls
        tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", solution_str, re.DOTALL)
        
        # Parse tool calls to extract tool names and count zoom calls
        for tool_call_content in tool_calls:
            try:
                tool_data = json.loads(tool_call_content.strip())
                tool_items = tool_data if isinstance(tool_data, list) else [tool_data]
                
                for item in tool_items:
                    if isinstance(item, dict):
                        tool_name = item.get("tool_name") or item.get("name")
                        if tool_name:
                            tools_used.add(tool_name)
                            if tool_name == "image_zoom_in_tool":
                                zoom_call_count += 1
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        
        # Extract last zoom bbox and compute IoU
        last_zoom_bbox = _extract_last_zoom_bbox(solution_str)
        if last_zoom_bbox:
            zoom_offsets = extra_info.get("zoom_offsets", []) if extra_info else []
            
            # Transform bbox to original coordinates if offsets exist
            if zoom_offsets:
                pred_bbox = _transform_bbox_with_offsets(last_zoom_bbox, zoom_offsets)
                logger.debug(f"Transformed bbox {last_zoom_bbox} -> {pred_bbox}")
            else:
                pred_bbox = last_zoom_bbox
                logger.debug(f"Using bbox as-is: {pred_bbox}")
            
            # Compute IoU with ground truth mask if available
            gt_mask_bytes = extra_info.get("mask_image") if extra_info else None
            if gt_mask_bytes and isinstance(gt_mask_bytes, bytes):
                bbox_iou = _compute_mask_iou([pred_bbox], gt_mask_bytes, max_pred_boxes=3) + 0.001
                logger.debug(f"Mask-based IoU: {bbox_iou:.4f}")
            else:
                bbox_iou = 0.001
                logger.debug("No mask available, using base IoU")
        
        # Tool diversity bonus: reward using both zoom and reference tools
        if "image_zoom_in_tool" in tools_used and "image_reference_tool" in tools_used:
            tool_diversity_bonus = 0.1
            logger.debug("Tool diversity bonus applied")
        
        # Zoom count reward: encourage exactly 2 zoom calls
        if zoom_call_count == 1:
            zoom_count_reward = 0.1
        elif zoom_call_count == 2:
            zoom_count_reward = 0.3
        elif zoom_call_count > 2:
            zoom_count_reward = -0.4
        
        logger.debug(f"Zoom calls: {zoom_call_count}, reward: {zoom_count_reward:.2f}")
        
        # Apply cube root transformation to IoU
        bbox_iou_transformed = math.pow(bbox_iou, 1/3)
        logger.debug(f"IoU: {bbox_iou:.4f} -> {bbox_iou_transformed:.4f} (cube root)")
        
        # Calculate tool rewards
        tool_reward = 2 * bbox_iou_transformed
        
        if acc_reward == 1.0 and zoom_count_reward > 0:
            tool_valid_reward = 0.5
        elif acc_reward == 0.0 and zoom_count_reward > 0:
            tool_valid_reward = -0.4
    
    # ============================================================================
    # 4. FINAL SCORE
    # ============================================================================
    final_score = 0.3 * format_reward + acc_reward + tool_valid_reward + tool_reward
    
    logger.debug(
        f"Score: format={format_reward:.2f}, acc={acc_reward:.2f}, "
        f"tool={tool_reward:.2f}, tool_valid={tool_valid_reward:.2f}, "
        f"final={final_score:.2f} | Tools: {tools_used}"
    )
    
    return {
        "score": float(final_score),
        "format_reward": float(format_reward),
        "acc_reward": float(acc_reward),
        "tool_reward": float(tool_reward),
        "bbox_iou": float(bbox_iou_transformed),
        "tool_diversity_bonus": float(tool_diversity_bonus),
        "zoom_count_reward": float(zoom_count_reward),
        "tool_valid_reward": float(tool_valid_reward),
    }


if __name__ == "__main__":
    import time

    # Test case 1: Well-formatted response with valid tool calls (matching user's format)
    test_case_1 = """<think>Upon inspection, I notice a small protrusion near the edge of the component's metal terminal, which is positioned away from the body of the circuit block. This irregularity could be a manufacturing defect such as a poorly formed lead or partial material. A zoom-in at this location will provide closer details.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [588, 657, 613, 667], "label": "metal terminal"}}
</tool_call>user
<tool_response>
Zoomed in on the image to the region [588, 657, 613, 667] with label metal terminal.
</tool_response>
assistant
<think>I observe a thin, protruding disc located near the metal terminal of the component. This protrusion could indicate a manufacturing flaw or an area that has been accidentally touched or damaged. To ensure the accuracy of the observation, I need to provide a closer look at this region.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [588, 657, 613, 667], "label": "metal terminal"}}
</tool_call>user
<tool_response>
Zoomed in on the image to the region [588, 657, 613, 667] with label metal terminal.
</tool_response>
assistant
<think>By zooming in,I observed that the protrusion was likely a shadow cast from the component housing rather than an actual defect in the terminal. Considering the typical manufacturing techniques and the orientation of the light within the image, the shadowing can be attributed to normal lighting conditions rather than a defect in the terminal itself.</think>
<loc>[{"bbox_2d": [598, 680, 272, 692]},{"bbox_2d": [598, 776, 272, 788]}]</loc>
<type>metal terminal</type>
<answer>no</answer>"""

    ground_truth_1 = {"answer": "no", "bboxes": [{"bbox_2d": [590, 670, 280, 700]}, {"bbox_2d": [590, 770, 280, 795]}]}
    extra_info_1 = {
        "question": "Does this image contain any defects?",
        "bboxes": [{"bbox_2d": [590, 670, 280, 700]}, {"bbox_2d": [590, 770, 280, 795]}],
    }

    print("=== Test Case 1: Well-formatted with valid tool calls ===")
    time_start = time.time()
    score = compute_score("defect_detection", test_case_1, ground_truth_1, extra_info_1)
    print(f"Score: {score}")
    time_end = time.time()
    print(f"Time: {time_end - time_start}")

    # Test case 2: Mismatched tool call tags
    test_case_2 = """<think>
I need to examine the image.
</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [100, 150, 200, 250]}}
<loc>[]</loc>
<type>good</type>
<answer>no</answer>"""

    ground_truth_2 = {"answer": "no", "bboxes": []}
    extra_info_2 = {
        "question": "Does this image contain any defects?",
        "bboxes": [],
    }

    print("\n=== Test Case 2: Mismatched tool call tags (missing </tool_call>) ===")
    time_start = time.time()
    score2 = compute_score("defect_detection", test_case_2, ground_truth_2, extra_info_2)
    print(f"Score: {score2}")
    time_end = time.time()
    print(f"Time: {time_end - time_start}")

    # Test case 3: No tool calls (valid case)
    test_case_3 = """<think>
After examining the image, I can see the surface is clean and smooth with no visible defects.
</think>
<loc>[]</loc>
<type>good</type>
<answer>no</answer>"""

    print("\n=== Test Case 3: No tool calls (valid case) ===")
    time_start = time.time()
    score3 = compute_score("defect_detection", test_case_3, ground_truth_2, extra_info_2)
    print(f"Score: {score3}")
    time_end = time.time()
    print(f"Time: {time_end - time_start}")
    
    # Test case 4: Format error - missing tags
    test_case_4 = """<think>
Let me check for defects.
</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [100, 150, 200, 250]}}
</tool_call>
<answer>yes</answer>"""

    print("\n=== Test Case 4: Format error (missing <loc> and <type> tags) ===")
    time_start = time.time()
    score4 = compute_score("defect_detection", test_case_4, ground_truth_1, extra_info_1)
    print(f"Score: {score4}")
    time_end = time.time()
    print(f"Time: {time_end - time_start}")
    
    # Test case 5: Using both tools (zoom + reference) - should get diversity bonus
    test_case_5 = """<think>I see some texture variations. Let me check the reference image first.</think>
<tool_call>
[{"tool_name": "image_reference_tool", "parameters": {"reason": "to compare surface texture patterns"}}]
</tool_call>
<tool_response>
Reference image retrieved successfully.
</tool_response>
<think>After comparing with the reference, I notice a potential defect. Let me zoom in to confirm.</think>
<tool_call>
[{"tool_name": "image_zoom_in_tool", "parameters": {"bbox_2d": [590, 670, 280, 700]}}]
</tool_call>
<tool_response>
Zoomed in on the region [590, 670, 280, 700].
</tool_response>
<think>After using both tools, I can confirm there is a defect in this region.</think>
<location>[{"bbox2d": [590, 670, 280, 700]}]</location>
<type>surface</type>
<answer>yes</answer>"""
    
    ground_truth_5 = {"answer": "yes", "bboxes": [{"bbox_2d": [590, 670, 280, 700]}]}
    extra_info_5 = {
        "question": "Does this image contain any defects?",
        "bboxes": [{"bbox_2d": [590, 670, 280, 700]}],
    }
    
    print("\n=== Test Case 5: Using both tools (should get diversity bonus) ===")
    time_start = time.time()
    score5 = compute_score("defect_detection", test_case_5, ground_truth_5, extra_info_5)
    print(f"Score: {score5}")
    print(f"Tool diversity bonus: {score5.get('tool_diversity_bonus', 0.0)}")
    print(f"Tools used: {score5.get('tools_used', [])}")
    time_end = time.time()
    print(f"Time: {time_end - time_start}")
