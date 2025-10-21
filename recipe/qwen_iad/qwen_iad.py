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

logger = logging.getLogger(__name__)

SYSTEM_PROMPT: str = (
    "You are a highly precise and meticulous quality control inspector. Your mission is to analyze images for defects based on visual evidence.\n"
    "Your response must follow one of the three formats below, depending on your assessment.\n"
    "---\n"
    "### Format 1: Use When You Need to Use a Tool\n"
    "If you cannot confidently identify or locate a defect and require a closer view or you don't find the defect, you can try to use the tool to get more information,"
    " your response **MUST ONLY** contain these two tags:\n"
    "1.  **`<think></think>`**: Explain why you need to use a tool. For example, describe what you see and why it's ambiguous "
    '(e.g., "I see a potential anomaly in the lower-right corner, but the resolution is too low to confirm if it\'s a crack or a shadow. '
    'I will use the zoom tool to inspect it.").\n'
    "2.  **`<tool_call></tool_call>`**: Provide the tool call needed to get more information.\n"
    "** Example of a Tool Request Turn:**\n"
    "<think>I observe a faint, dark spot on the main body of the component. It is unclear if this is a surface hole or a smudge. "
    "I need to zoom in to determine its nature and precise boundaries.</think>\n"
    "<tool_call>\n"
    '[{"tool_name": "zoom_in", "parameters": {"bbox": [250, 300, 300, 350]}}]\n'
    "</tool_call>\n"
    "---\n"
    "### Format 2: Use When You Have Found a Defect\n"
    "Use this format to provide your final conclusion when you have confidently identified one or more defects "
    "(either initially or after using a tool). Your response MUST NOT contain `<tool_call>`.\n"
    "1.  **`<think></think>`**: Provide your final step-by-step reasoning. If you previously used a tool, explain how the tool's output "
    'helped you make the final decision (e.g., "After zooming in, the spot is clearly a small, pitted hole in the surface.").\n'
    "2.  **`<location></location>`**: Provide a JSON list of all detected defect locations (defect only, not the whole object).\n"
    '    - Must use "bbox2d" key with [x_min, y_min, x_max, y_max] coordinates.\n'
    "    - Maximum of 3 bounding boxes. You should always remember this rule.\n"
    '    - Example: [{"bbox2d": [100, 150, 200, 250]}]\n'
    "3.  **`<type></type>`**: Specify the defect type from the list: \"crack\", \"discoloration\", \"scratch\", \"hole\", \"surface\", \"other\". "
    'Use "unspecified" if uncertain.\n'
    '4.  **`<answer></answer>`**: Conclude with "yes".\n'
    "** Example of a Final \"Defect Found\" Turn (after a tool was used):**\n"
    "<think>The zoomed-in view from the tool confirms that the dark spot is a well-defined circular hole, not a smudge. "
    "I can now confidently mark its location and type.</think>\n"
    '[{"bbox2d": [265, 310, 280, 325]}]\n'
    "<type>hole</type>\n"
    "<answer>yes</answer>\n"
    "---\n"
    "### Format 3: Use When You Have Found NO Defects\n"
    "Use this format when you are confident the item is free of defects.\n"
    "1.  **`<think></think>`**: Explain why you believe the object is defect-free. Describe the normal and healthy features you observed.\n"
    "2.  **`<location></location>`**: Provide an empty JSON list: [].\n"
    '3.  **`<type></type>`**: Use the value "good".\n'
    '4.  **`<answer></answer>`**: Conclude with "no".\n'
    "** Example of a \"No Defect\" Turn:**\n"
    "<think>I have thoroughly scanned the entire surface. The finish is uniform, and there are no signs of cracks, scratches, "
    "or any other anomalies. The object meets quality standards.</think>\n"
    "<location>[]</location>\n"
    "<type>good</type>\n"
    "<answer>no</answer>\n"
)
openai_api_key = "EMPTY"
openai_api_base = os.environ.get("LLM_AS_A_JUDGE_BASE", "http://10.1.100.71:18901/v1")

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model_name = ""
if openai_api_base:
    try:
        response = requests.get(f"{openai_api_base}/models")
        response.raise_for_status()
        models = response.json()
        if models.get("data"):
            model_name = models["data"][0]["id"]
        else:
            logger.warning("No models found at the specified API base for reward scoring.")
    except (requests.exceptions.RequestException, KeyError, IndexError) as e:
        logger.warning(f"Failed to get model from {openai_api_base}: {e}. Reward scoring will be disabled.")


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
        tools_kwargs = {
            "image_zoom_in_tool": {
                "create_kwargs": {"image": images[0]},
                # "execute_kwargs": {},
                # "calc_reward_kwargs": {},
                # "release_kwargs": {},
            }
        }
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["agent_name"] = "tool_agent"
        return row_dict


def extract_answer(text):
    """Extract content from <answer></answer> tags."""
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_location(text):
    """Extract content from <location></location> tags."""
    pattern = r"<location>(.*?)</location>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_type(text):
    """Extract content from <type></type> tags."""
    pattern = r"<type>(.*?)</type>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _parse_predicted_bboxes(location_text):
    """Parse predicted bboxes from location JSON string into list of [x1,y1,x2,y2]."""
    if not location_text:
        return []

    import json

    # Try strict JSON first
    try:
        loc = json.loads(location_text)
    except Exception:
        # Try Python-literal style (single quotes, None, etc.)
        try:
            import ast

            loc = ast.literal_eval(location_text)
        except Exception:
            loc = None

    boxes = []
    if isinstance(loc, list):
        for item in loc:
            if isinstance(item, dict):
                arr = item.get("bbox2d") or item.get("bbox_2d")
                if isinstance(arr, (list | tuple)) and len(arr) == 4:
                    try:
                        boxes.append([float(v) for v in arr])
                    except Exception:
                        continue

    if boxes:
        return boxes

    # Regex fallback: extract any [x1,y1,x2,y2]
    try:
        pattern = r"\[\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)\s*\]"
        matches = re.findall(pattern, location_text)
        for m in matches:
            vals = [float(x) for x in m]
            if len(vals) == 4:
                boxes.append(vals)
        return boxes
    except Exception as e:
        logger.error(f"Failed to parse predicted bboxes: {e}")
        return []


def _extract_gt_bboxes(ground_truth, extra_info):
    """Extract ground truth bboxes list[[x1,y1,x2,y2]]."""
    boxes = []
    # Prefer ground_truth dict
    try:
        if isinstance(ground_truth, dict) and "bboxes" in ground_truth:
            gt_boxes = ground_truth["bboxes"]
            try:
                iterable = list(gt_boxes)
            except Exception:
                iterable = []
            if iterable:
                for item in iterable:
                    if isinstance(item, dict):
                        arr = item.get("bbox2d") or item.get("bbox_2d")
                        if isinstance(arr, list) and len(arr) == 4:
                            boxes.append([float(v) for v in arr])
        # Fallback to extra_info
        if not boxes and extra_info and "bboxes" in extra_info:
            gt_boxes = extra_info["bboxes"]
            try:
                iterable = list(gt_boxes)
            except Exception:
                iterable = []
            if iterable:
                for item in iterable:
                    if isinstance(item, dict):
                        arr = item.get("bbox2d") or item.get("bbox_2d")
                        if isinstance(arr, list) and len(arr) == 4:
                            boxes.append([float(v) for v in arr])
    except Exception as e:
        logger.error(f"Failed to extract GT bboxes: {e}")
    return boxes


def _bbox_iou_xyxy(box_a, box_b):
    """Compute IoU between two bboxes in xyxy format."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _improved_iou_reward(pred_boxes, gt_boxes, max_pred_boxes=3):
    """
    IoU-based reward calculation focusing on precision (how accurate predictions are).
    Returns the average IoU of predicted boxes matched to ground truth boxes.
    """
    # Limit predicted boxes to avoid excessive outputs
    if len(pred_boxes) > max_pred_boxes:
        pred_boxes = pred_boxes[:max_pred_boxes]

    # Empty cases
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0
    if len(gt_boxes) == 0 and len(pred_boxes) > 0:
        # Penalty for false positive detections
        return 0.0
    if len(gt_boxes) > 0 and len(pred_boxes) == 0:
        # Penalty for missing detections
        return 0.0

    # Compute proper IoU with intersection/union formula
    def compute_proper_iou(box1, box2):
        """Compute IoU = intersection / union"""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_area = max(0.0, x2_inter - x1_inter) * max(0.0, y2_inter - y1_inter)

        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    # Compute best matching IoU for each predicted box (precision metric)
    pred_ious = []
    for pred_box in pred_boxes:
        best_iou = 0.0
        for gt_box in gt_boxes:
            iou = compute_proper_iou(pred_box, gt_box)
            best_iou = max(best_iou, iou)
        pred_ious.append(best_iou)

    # Return average IoU directly (precision: how accurate are the predictions)
    precision = sum(pred_ious) / len(pred_ious) if pred_ious else 0.0
    return precision


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
        
        # Create prediction mask from bboxes
        pred_mask = np.zeros((height, width), dtype=np.uint8)
        
        for box in pred_boxes:
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
        intersection = np.logical_and(pred_mask, gt_mask_binary).sum()
        union = np.logical_or(pred_mask, gt_mask_binary).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        iou = float(intersection) / float(union)
        return iou
        
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


def _validate_tool_call_json(solution_str):
    """
    Validate that tool_call tags contain valid JSON.

    Args:
        solution_str: The solution string to check

    Returns:
        bool: True if all tool_call contents are valid JSON (or no tool_calls), False otherwise
    """
    import json

    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    tool_calls = re.findall(tool_call_pattern, solution_str, re.DOTALL)

    if not tool_calls:
        return True  # No tool calls is valid

    for tool_call_content in tool_calls:
        try:
            json.loads(tool_call_content.strip())
        except (json.JSONDecodeError, ValueError):
            return False

    return True


def _smooth_format_reward(format_errors_count, total_checks=4):
    """
    Compute smooth format reward based on the number of format errors.

    Args:
        format_errors_count: Number of format errors detected
        total_checks: Total number of format checks performed

    Returns:
        float: Smooth format reward in range [-0.5, 0.0]
    """
    if format_errors_count == 0:
        return 0.0

    # Use sigmoid-like function to smooth the penalty
    error_ratio = format_errors_count / total_checks
    # Map error ratio to smooth penalty using tanh
    smooth_penalty = -0.5 * math.tanh(2.0 * error_ratio)
    return smooth_penalty


def _smooth_acc_reward(llm_response, confidence_threshold=0.8):
    """
    Compute smooth accuracy reward with confidence consideration.

    Args:
        llm_response: Response from LLM judge
        confidence_threshold: Threshold for high confidence

    Returns:
        float: Smooth accuracy reward in range [0.0, 1.0]
    """
    if re.search(r"\bCORRECT\b", llm_response, re.IGNORECASE):
        # Check for confidence indicators in the response
        confidence_words = [
            "definitely",
            "clearly",
            "obviously",
            "certainly",
            "absolutely",
        ]
        uncertainty_words = [
            "maybe",
            "possibly",
            "might",
            "could",
            "uncertain",
            "unclear",
        ]

        confidence_score = 1.0
        if any(word in llm_response.lower() for word in uncertainty_words):
            confidence_score = 0.7
        elif any(word in llm_response.lower() for word in confidence_words):
            confidence_score = 1.0
        else:
            confidence_score = 0.85  # Default confidence for CORRECT

        return confidence_score
    elif re.search(r"\bINCORRECT\b", llm_response, re.IGNORECASE):
        return 0.0
    else:
        # Ambiguous response gets low reward
        return 0.1


def _clip_and_normalize_reward(reward, min_val=-1.0, max_val=1.0):
    """
    Clip and normalize reward to prevent extreme values.

    Args:
        reward: Raw reward value
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        float: Clipped and normalized reward
    """
    clipped = max(min_val, min(max_val, reward))
    # Apply tanh for additional smoothing
    return math.tanh(clipped)


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Compute reward score for defect detection task.

    The score consists of three components:
    1. Format reward: Hard reward (1.0 if no errors, 0.0 if any errors)
    2. Answer correctness reward: Hard reward (1.0 if match, 0.0 otherwise)
    3. Tool reward: Combination of tool usage (0.0 or 1.0) and bbox IoU

    Args:
        data_source: Source of the data (not used currently)
        solution_str: Model's solution string
        ground_truth: Ground truth answer (can be dict or string)
        extra_info: Additional information including question, bboxes, etc.

    Returns:
        dict: Dictionary containing score and reward components
    """
    import json

    # 1. Format reward: Check tag pairing
    is_format_error = False
    
    # Check <think> tags
    count_think_1 = solution_str.count("<think>")
    count_think_2 = solution_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True
    
    # Check <tool_call> and <tool_response> tags (only need pairing, not every turn has them)
    count_tool_call_1 = solution_str.count("<tool_call>")
    count_tool_call_2 = solution_str.count("</tool_call>")
    if count_tool_call_1 != count_tool_call_2:
        is_format_error = True
    
    count_tool_response_1 = solution_str.count("<tool_response>")
    count_tool_response_2 = solution_str.count("</tool_response>")
    if count_tool_response_1 != count_tool_response_2:
        is_format_error = True
    
    # Check <answer> tags
    count_answer_1 = solution_str.count("<answer>")
    count_answer_2 = solution_str.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True
    
    # Check <loc> tags (used instead of <location>)
    count_loc_1 = solution_str.count("<location>")
    count_loc_2 = solution_str.count("</location>")
    if count_loc_1 != count_loc_2:
        is_format_error = True
    
    # Check <type> tags
    count_type_1 = solution_str.count("<type>")
    count_type_2 = solution_str.count("</type>")
    if count_type_1 != count_type_2:
        is_format_error = True
    
    format_reward = 1.0 if not is_format_error else 0.0
    
    # 2. Extract answer and compute acc_reward
    answer_text = ""
    answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    
    # Extract ground truth answer
    ground_truth_answer = _extract_ground_truth_answer(ground_truth, extra_info)
    
    # Check if answer contains "yes" or "no" and matches ground_truth
    if answer_text:
        answer_normalized = answer_text.strip().lower()
        ground_truth_normalized = ground_truth_answer.strip().lower()
        
        # Check if ground_truth is "yes" or "no"
        if "yes" in ground_truth_normalized:
            acc_reward = 1.0 if "yes" in answer_normalized else 0.0
        elif "no" in ground_truth_normalized:
            acc_reward = 1.0 if "no" in answer_normalized else 0.0
        else:
            # Fallback to exact matching if ground_truth is not yes/no
            acc_reward = 1.0 if answer_normalized == ground_truth_normalized else -1.0
    else:
        acc_reward = 0.0
    
    # 3. Tool reward: combination of tool usage and bbox IoU
    # Part 1: Check if tools were called
    has_tool_usage = count_tool_call_1 > 0
    tool_usage_reward = 1.0 if has_tool_usage else 0.0
    
    # Part 2: Compute bbox IoU from tool_call parameters or mask image
    bbox_iou = 0.0
    if has_tool_usage:
        # Extract bbox from the last tool_call only
        tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
        tool_calls = re.findall(tool_call_pattern, solution_str, re.DOTALL)
        
        pred_boxes = []
        # Only process the last tool_call
        if tool_calls:
            tool_call_content = tool_calls[-1]
            try:
                tool_data = json.loads(tool_call_content.strip())
                # Extract bbox_2d from arguments
                if isinstance(tool_data, dict):
                    args = tool_data.get("arguments", {})
                    bbox = args.get("bbox_2d") or args.get("bbox2d")
                    if bbox and isinstance(bbox, list) and len(bbox) == 4:
                        pred_boxes.append([float(v) for v in bbox])
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        
        if pred_boxes:
            # Check if mask image is available in extra_info (only exists for defective samples)
            gt_mask_bytes = None
            if extra_info and isinstance(extra_info, dict) and "mask_image" in extra_info:
                gt_mask_bytes = extra_info.get("mask_image")
            
            # Determine which IoU calculation method to use
            # Priority: mask-based IoU > bbox-based IoU
            if gt_mask_bytes is not None and isinstance(gt_mask_bytes, bytes):
                # Use mask-based IoU (more accurate for defective samples)
                bbox_iou = _compute_mask_iou(pred_boxes, gt_mask_bytes, max_pred_boxes=3)
                logger.debug(f"Using mask-based IoU: {bbox_iou:.4f}")
            else:
                # Fallback to bbox-based IoU
                # This handles two cases:
                # 1. Good samples (label=0) without mask_image
                # 2. Legacy data without mask_image field
                
                gt_boxes = _extract_gt_bboxes(ground_truth, extra_info)
                bbox_iou = _improved_iou_reward(pred_boxes, gt_boxes, max_pred_boxes=3)
                if gt_boxes:
                    logger.debug(f"Using bbox-based IoU with {len(gt_boxes)} GT boxes: {bbox_iou:.4f}")
                else:
                    logger.debug(f"No GT boxes/mask found, IoU penalty: {bbox_iou:.4f}")
    
    # Combined tool reward
    tool_reward = 0.5 * tool_usage_reward + 0.5 * bbox_iou
    
    # Final score calculation
    # Weighted combination: format (0.5), acc (0.5), tool (1.0)
    final_score = 0.25 * format_reward + 0.25 * acc_reward + 0.5 * tool_reward
    
    # Log for debugging
    logger.debug(
        f"Score breakdown: format={format_reward:.2f}, acc={acc_reward:.2f}, "
        f"tool_usage={tool_usage_reward:.2f}, bbox_iou={bbox_iou:.2f}, "
        f"tool={tool_reward:.2f}, final={final_score:.2f}"
    )
    
    return {
        "score": final_score,
        "format_reward": format_reward,
        "acc_reward": acc_reward,
        "tool_reward": tool_reward,
        "bbox_iou": bbox_iou,
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
<think>By zooming in,我 observed that the protrusion was likely a shadow cast from the component housing rather than an actual defect in the terminal. Considering the typical manufacturing techniques and the orientation of the light within the image, the shadowing can be attributed to normal lighting conditions rather than a defect in the terminal itself.</think>
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
