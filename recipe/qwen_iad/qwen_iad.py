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
from PIL import Image, ImageDraw, ImageFont

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

SYSTEM_PROMPT: str = (
    "You are a highly precise and meticulous quality control inspector. Your primary mission is to "
    "analyze images and determine if any defects are present. Your decision must be grounded in visual evidence."
    "Carefully follow these instructions for your response format:"
    "**If you detect one or more defects:**"
    "Your response MUST be structured with the following four tags in this exact order:"
    "1.  `<think></think>`: Provide a step-by-step reasoning process. Describe the visual "
    'characteristics of the anomaly (e.g., "I observe a dark, irregular crack on the upper left surface...").'
    "2.  `<location></location>`: Provide a JSON list of all detected defect locations.Notice! You should give the location of the only defects not the full object."
    ' Each item in the list must be a JSON object with a "bbox2d" key and coordinates in `[x_min, y_min, x_max, y_max]` format. '
    'For example: `[{"bbox2d": [100, 150, 200, 250]}, {"bbox2d": [300, 350, 400, 450]}]`.Do not give more than 3 bounding boxes. If you are uncertain about the exact location, '
    "provide an approximate bounding box that best encompasses the defect area."
    '3.  `<type></type>`: Specify the type of defect found (e.g., "crack", "discoloration", "scratch", "hole", "surface" and "other"). If the type is uncertain, use "unspecified".'
    '4.  `<answer></answer>`: Conclude with "yes".'
    "**If you detect NO defects:**"
    "Your response MUST be structured with the following two tags:"
    "1.  `<think></think>`: Explain why you believe the object is defect-free. Describe the normal and healthy features you observed."
    "2.  `<location></location>`: Provide an empty JSON list."
    '3.  `<type></type>`: "good".'
    '4.  `<answer></answer>`: Conclude with "no".'
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
    Improved IoU-based reward calculation with proper union computation.
    Limits predicted boxes to max_pred_boxes and uses proper IoU formula.
    """
    # Limit predicted boxes to avoid excessive outputs
    if len(pred_boxes) > max_pred_boxes:
        pred_boxes = pred_boxes[:max_pred_boxes]

    # Empty cases
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0
    if len(gt_boxes) == 0 and len(pred_boxes) > 0:
        # Penalty for false positive detections
        return max(0.0, 1.0 - 0.2 * len(pred_boxes))
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

    # Compute best matching IoU for each GT box
    gt_ious = []
    for gt_box in gt_boxes:
        best_iou = 0.0
        for pred_box in pred_boxes:
            iou = compute_proper_iou(gt_box, pred_box)
            best_iou = max(best_iou, iou)
        gt_ious.append(best_iou)

    # Compute best matching IoU for each predicted box
    pred_ious = []
    for pred_box in pred_boxes:
        best_iou = 0.0
        for gt_box in gt_boxes:
            iou = compute_proper_iou(pred_box, gt_box)
            best_iou = max(best_iou, iou)
        pred_ious.append(best_iou)

    # Use the average of both directions, weighted by recall and precision
    recall = sum(gt_ious) / len(gt_ious) if gt_ious else 0.0
    precision = sum(pred_ious) / len(pred_ious) if pred_ious else 0.0

    # F1-score like combination
    if recall + precision > 0:
        f1_score = 2 * (recall * precision) / (recall + precision)
    else:
        f1_score = 0.0

    # Apply progressive IoU thresholds for better reward shaping
    if f1_score >= 0.7:
        return 1.0
    elif f1_score >= 0.5:
        return 0.8 + 0.2 * (f1_score - 0.5) / 0.2
    elif f1_score >= 0.3:
        return 0.5 + 0.3 * (f1_score - 0.3) / 0.2
    else:
        return f1_score / 0.3 * 0.5


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
        confidence_words = ["definitely", "clearly", "obviously", "certainly", "absolutely"]
        uncertainty_words = ["maybe", "possibly", "might", "could", "uncertain", "unclear"]

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
    1. Format reward: Checks if output follows expected format with proper tags
    2. Answer correctness reward: Uses LLM judge to evaluate answer accuracy
    3. Bbox correctness reward: Only computed when answer is correct, evaluates bbox IoU

    Args:
        data_source: Source of the data (not used currently)
        solution_str: Model's solution string
        ground_truth: Ground truth answer (can be dict or string)
        extra_info: Additional information including question, bboxes, etc.

    Returns:
        float: Final reward score
    """
    import json

    format_errors_count = 0

    # 1. Format checking - check all required tags and count errors
    count_think_1 = solution_str.count("<think>")
    count_think_2 = solution_str.count("</think>")
    if count_think_1 != count_think_2:
        format_errors_count += 1

    predict_no_think = solution_str.split("</think>")[-1].strip() if "</think>" in solution_str else solution_str

    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        format_errors_count += 1

    count_location_1 = predict_no_think.count("<location>")
    count_location_2 = predict_no_think.count("</location>")
    if count_location_1 != count_location_2:
        format_errors_count += 1

    count_type_1 = predict_no_think.count("<type>")
    count_type_2 = predict_no_think.count("</type>")
    if count_type_1 != count_type_2:
        format_errors_count += 1

    # Extract components
    answer_text = extract_answer(predict_no_think)
    location_text = extract_location(predict_no_think)
    type_text = extract_type(predict_no_think)  # noqa: F841 - extracted for format checking, not used in reward

    # Check if answer exists
    if not answer_text:
        format_errors_count += 1
        answer_text = ""

    # Check bbox format (3-box limit)
    bbox_format_ok = False
    if location_text:
        try:
            loc = json.loads(location_text)
            if isinstance(loc, list) and len(loc) <= 3:
                bbox_format_ok = all(
                    isinstance(item, dict)
                    and ("bbox2d" in item or "bbox_2d" in item)
                    and isinstance((item.get("bbox2d") or item.get("bbox_2d")), list)
                    and len(item.get("bbox2d") or item.get("bbox_2d")) == 4
                    for item in loc
                )
        except (json.JSONDecodeError, TypeError):
            pass

    # Smooth format reward based on error count
    format_reward = _smooth_format_reward(format_errors_count, total_checks=5)

    # 2. Answer correctness using LLM judge
    if not client or not model_name:
        logger.warning("Reward function client not initialized or model name not found.")
        return format_reward

    # Extract ground truth answer
    ground_truth_answer = _extract_ground_truth_answer(ground_truth, extra_info)
    question_text = extra_info.get("question", "") if extra_info else ""

    # Penalize excessively long answers
    if len(answer_text) >= 1000:
        acc_reward = 0.0
        format_errors_count += 1
        format_reward = _smooth_format_reward(format_errors_count, total_checks=5)
    elif not answer_text:
        acc_reward = 0.0
    else:
        # Use LLM judge to evaluate answer correctness
        system_prompt = (
            "You are an expert evaluator for industrial defect detection. Your task is to determine if a model's "
            "answer is semantically equivalent to the standard answer.\n"
            "The answer should be 'yes' or 'no' (or variations like 'Yes. There has been a defect detected.').\n"
            'You must provide your final judgement as a single word: either "CORRECT" or "INCORRECT". '
            'You may also include confidence indicators like "definitely", "clearly", "possibly", etc.'
        )

        user_prompt = (
            f"Evaluate if the model's answer matches the standard answer.\n\n"
            f"[Question]: {question_text}\n"
            f"[Standard Answer]: {ground_truth_answer}\n"
            f"[Model's Answer]: {answer_text}\n"
            f"[Your Judgement]:"
        )

        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                seed=random.randint(0, 1000000),
                temperature=0.1,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            response = chat_response.choices[0].message.content.strip()

            # Use smooth acc reward with confidence consideration
            acc_reward = _smooth_acc_reward(response)
        except Exception as e:
            logger.warning(f"LLM judge request failed: {e}")
            acc_reward = 0.0

    # 3. Bbox correctness reward - ONLY computed when answer is correct
    bbox_reward = 0.0
    if acc_reward > 0.5:  # Only compute bbox reward when answer is correct
        # Parse predicted and ground truth bboxes
        pred_boxes = _parse_predicted_bboxes(location_text)
        gt_boxes = _extract_gt_bboxes(ground_truth, extra_info)

        # Compute IoU-based reward
        bbox_iou_reward = _improved_iou_reward(pred_boxes, gt_boxes, max_pred_boxes=3)

        # Combine format and IoU rewards for bbox
        bbox_reward = 0.2 * (1.0 if bbox_format_ok else 0.0) + 0.8 * bbox_iou_reward

    # Final score calculation with smooth combination
    # Apply clipping and normalization to individual rewards
    # format_reward_clipped = _clip_and_normalize_reward(format_reward, min_val=-0.5, max_val=1.0)
    # acc_reward_clipped = _clip_and_normalize_reward(acc_reward, min_val=0.0, max_val=1.0)

    # Weighted combination with smooth blending
    raw_score = 0.5 * format_reward + 0.5 * acc_reward + 0.5 * bbox_reward
    # raw_score = 0.5 * format_reward + 0.5 * acc_reward

    # Apply final smoothing to prevent extreme gradients
    final_score = _clip_and_normalize_reward(raw_score, min_val=-0.5, max_val=1.0)

    # Log for debugging
    if extra_info:
        logger.debug(
            f"Score breakdown: format={format_reward:.2f}, acc={acc_reward:.2f}, "
            f"bbox={bbox_reward:.2f}, final={final_score:.2f}"
        )
        print(
            f"Score breakdown: format={format_reward:.2f}, acc={acc_reward:.2f}, "
            f"bbox={bbox_reward:.2f}, final={final_score:.2f}"
        )

    return {
        "score": final_score,
        "format_reward": format_reward,
        "acc_reward": acc_reward,
        "bbox_reward": bbox_reward,
    }


def draw_bboxes_on_image(image, pred_boxes, gt_boxes, save_path, sample_id="sample"):
    """
    Draw predicted and ground truth bboxes on image and save to file.

    Args:
        image: PIL Image object
        pred_boxes: List of predicted bboxes in format [[x1,y1,x2,y2], ...]
        gt_boxes: List of ground truth bboxes in format [[x1,y1,x2,y2], ...]
        save_path: Directory path to save the image
        sample_id: Unique identifier for the sample

    Returns:
        str: Path to the saved image file
    """
    try:
        # Create a copy of the image to avoid modifying the original
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)

        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except OSError:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

        # Draw ground truth boxes in green
        for i, gt_box in enumerate(gt_boxes):
            x1, y1, x2, y2 = gt_box
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
            # Draw label
            label = f"GT_{i + 1}"
            if font:
                draw.text((x1, y1 - 20), label, fill="green", font=font)
            else:
                draw.text((x1, y1 - 15), label, fill="green")

        # Draw predicted boxes in red
        for i, pred_box in enumerate(pred_boxes):
            x1, y1, x2, y2 = pred_box
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            # Draw label
            label = f"PRED_{i + 1}"
            if font:
                draw.text((x1, y1 - 40), label, fill="red", font=font)
            else:
                draw.text((x1, y1 - 30), label, fill="red")

        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save the image
        filename = f"{sample_id}_bbox_visualization.jpg"
        full_path = os.path.join(save_path, filename)
        img_copy.save(full_path, "JPEG", quality=95)

        logger.info(f"Saved bbox visualization to {full_path}")
        return full_path

    except Exception as e:
        logger.error(f"Failed to draw bboxes on image: {e}")
        return None


def save_validation_images_with_bboxes(batch_data, outputs, save_dir, global_step):
    """
    Save validation images with predicted and ground truth bboxes drawn.

    Args:
        batch_data: Batch data containing images and ground truth information
        outputs: Model outputs containing predicted bboxes
        save_dir: Directory to save the visualization images
        global_step: Current training step for organizing saved files

    Returns:
        List of saved image paths
    """
    saved_paths = []

    try:
        # Create step-specific directory
        step_dir = os.path.join(save_dir, f"step_{global_step}")
        os.makedirs(step_dir, exist_ok=True)

        # Process each sample in the batch
        for i, (sample_data, output_text) in enumerate(zip(batch_data, outputs, strict=False)):
            try:
                # Extract image from sample data
                image = None
                if hasattr(sample_data, "non_tensor_batch") and "multi_modal_data" in sample_data.non_tensor_batch:
                    multi_modal_data = sample_data.non_tensor_batch["multi_modal_data"]
                    if "image" in multi_modal_data and multi_modal_data["image"]:
                        image = multi_modal_data["image"][0]  # Get first image
                elif hasattr(sample_data, "batch") and "multi_modal_data" in sample_data.batch:
                    multi_modal_data = sample_data.batch["multi_modal_data"]
                    if "image" in multi_modal_data and multi_modal_data["image"]:
                        image = multi_modal_data["image"][0]

                if image is None:
                    logger.warning(f"No image found for sample {i}")
                    continue

                # Extract predicted bboxes from output text
                location_text = extract_location(output_text)
                pred_boxes = _parse_predicted_bboxes(location_text)

                # Extract ground truth bboxes
                ground_truth = None
                extra_info = None

                if hasattr(sample_data, "non_tensor_batch"):
                    if "reward_model" in sample_data.non_tensor_batch:
                        reward_model_data = sample_data.non_tensor_batch["reward_model"]
                        if isinstance(reward_model_data, dict):
                            ground_truth = reward_model_data.get("ground_truth")
                        elif hasattr(reward_model_data, "get"):
                            ground_truth = reward_model_data.get("ground_truth")

                    if "extra_info" in sample_data.non_tensor_batch:
                        extra_info = sample_data.non_tensor_batch["extra_info"]

                gt_boxes = _extract_gt_bboxes(ground_truth, extra_info)

                # Create unique sample ID
                sample_id = f"sample_{i:04d}"
                if hasattr(sample_data, "non_tensor_batch") and "uid" in sample_data.non_tensor_batch:
                    uid = sample_data.non_tensor_batch["uid"]
                    if isinstance(uid, (list | tuple)) and len(uid) > 0:
                        sample_id = f"uid_{uid[0]}"
                    elif isinstance(uid, str):
                        sample_id = f"uid_{uid}"

                # Draw bboxes and save image
                saved_path = draw_bboxes_on_image(
                    image=image, pred_boxes=pred_boxes, gt_boxes=gt_boxes, save_path=step_dir, sample_id=sample_id
                )

                if saved_path:
                    saved_paths.append(saved_path)

            except Exception as e:
                logger.error(f"Failed to process sample {i}: {e}")
                continue

    except Exception as e:
        logger.error(f"Failed to save validation images with bboxes: {e}")

    return saved_paths


if __name__ == "__main__":
    # Test case 1: Original test case
    predict_str = "The answer is 2 + 2 = 4 </think> <answer> right </answer> <answer> left </answer>"
    ground_truth = "left"
    extra_info = {
        "answer": "The woman is to the left of the man who is holding the camera.",
        "id": 0,
        "image": "/cpfs/user/honglingyi/DATA/LLM/Vstar/gqa/images/713270.jpg",
        "pred_ans": "The woman is to the right of the man who is holding the camera.",
        "question": "Is the woman to the left or to the right of the man who is holding the camera?",
    }
    print("=== Test Case 1: Original test ===")
    import time

    time_start = time.time()
    score = compute_score("common_reasoning", predict_str, ground_truth, extra_info)
    print(f"Score: {score}")
    time_end = time.time()
    print(f"Time: {time_end - time_start}")

    # Test case 2: Problematic case mentioned by user
    problematic_solution = """<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [226, 399, 265, 464], "label": "white van"}}
</tool_call>user
<tool_response>
Zoomed in on the image to the region [226, 399, 265, 464] with label white van.
</tool_response>
assistant
The white van is visible in the lower section of the image, near the diagonal road."""

    problematic_ground_truth = "Yes, the white van is indeed situated in the bottom part of the picture."
    problematic_extra_info = {
        "question": "Is the white van in the bottom part of the picture?",
    }

    print("\n=== Test Case 2: Problematic case (no answer tags) ===")
    print(f"Solution: {problematic_solution}")
    print(f"Ground truth: {problematic_ground_truth}")

    time_start = time.time()
    score2 = compute_score(
        "common_reasoning",
        problematic_solution,
        problematic_ground_truth,
        problematic_extra_info,
    )
    print(f"Score: {score2}")
    time_end = time.time()
    print(f"Time: {time_end - time_start}")

    # Test case 3: Well-formatted case with tools
    well_formatted_solution = """<think>
I need to use the image zoom tool to get a better look at the specific area.
</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [226, 399, 265, 464], "label": "white van"}}
</tool_call>
<tool_response>
Zoomed in on the image to the region [226, 399, 265, 464] with label white van.
</tool_response>
<answer>Yes, the white van is indeed situated in the bottom part of the picture.</answer>"""

    print("\n=== Test Case 3: Well-formatted case ===")
    time_start = time.time()
    score3 = compute_score(
        "common_reasoning",
        well_formatted_solution,
        problematic_ground_truth,
        problematic_extra_info,
    )
    print(f"Score: {score3}")
    time_end = time.time()
    print(f"Time: {time_end - time_start}")
