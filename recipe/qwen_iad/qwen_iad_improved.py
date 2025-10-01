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
import os
import random
import re

import numpy as np
import requests
from openai import OpenAI
from PIL import Image

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

# Improved system prompt with variations for data augmentation
SYSTEM_PROMPT_VARIATIONS = [
    (
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
    ),
    (
        "As a professional quality inspector, your task is to examine images for any defects or anomalies. "
        "Base your assessment on clear visual evidence and follow the structured response format below:"
        "**When defects are detected:**"
        "Structure your response with these four required tags:"
        "1.  `<think></think>`: Detail your analysis process. Describe what you see "
        '(e.g., "There is a visible crack extending across the surface...").'
        "2.  `<location></location>`: List defect locations in JSON format with 'bbox2d' coordinates as `[x_min, y_min, x_max, y_max]`. "
        'Example: `[{"bbox2d": [100, 150, 200, 250]}]`. Maximum 3 bounding boxes.'
        '3.  `<type></type>`: Classify the defect type ("crack", "scratch", "hole", etc.).'
        '4.  `<answer></answer>`: State "yes".'
        "**When no defects are found:**"
        "Use this format:"
        "1.  `<think></think>`: Explain your reasoning for finding no defects."
        "2.  `<location></location>`: Empty JSON list `[]`."
        '3.  `<type></type>`: "good".'
        '4.  `<answer></answer>`: State "no".'
    ),
]

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
    """
    Improved RLHF Dataset with prompt augmentation to reduce gradient variance
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_augmentation = kwargs.get("prompt_augmentation", True)

    def _augment_system_prompt(self) -> str:
        """
        Randomly select from prompt variations to increase diversity
        """
        if self.prompt_augmentation:
            return random.choice(SYSTEM_PROMPT_VARIATIONS)
        else:
            return SYSTEM_PROMPT_VARIATIONS[0]

    def _add_image_noise(self, image: Image.Image, noise_level: float = 0.01) -> Image.Image:
        """
        Add subtle noise to image to increase robustness
        """
        if random.random() < 0.3:  # 30% chance to add noise
            img_array = np.array(image)
            noise = np.random.normal(0, noise_level * 255, img_array.shape).astype(np.uint8)
            noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy_array)
        return image

    def __getitem__(self, item):
        """
        Enhanced getitem with data augmentation for training stability
        """
        row_dict: dict = self.dataframe[item]

        # Use augmented system prompt
        system_prompt = self._augment_system_prompt()

        row_dict[self.prompt_key] = [
            {
                "role": "system",
                "content": system_prompt,
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

                # Apply image augmentation during training
                if hasattr(self, "training") and self.training:
                    images = [self._add_image_noise(img) for img in images]

                multi_modal_data["image"] = images

            model_inputs = self.processor(text=[raw_prompt], images=images, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            row_dict["multi_modal_data"] = multi_modal_data

            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)
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
            ]

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

        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt

        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = {
            "image_zoom_in_tool": {
                "create_kwargs": {"image": images[0]},
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

    try:
        loc = json.loads(location_text)
    except Exception:
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


def _improved_iou_reward(pred_boxes, gt_boxes, max_pred_boxes=3):
    """
    Improved IoU-based reward calculation with smoother transitions
    """
    if len(pred_boxes) > max_pred_boxes:
        pred_boxes = pred_boxes[:max_pred_boxes]

    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0
    if len(gt_boxes) == 0 and len(pred_boxes) > 0:
        return max(0.0, 1.0 - 0.15 * len(pred_boxes))  # Reduced penalty
    if len(gt_boxes) > 0 and len(pred_boxes) == 0:
        return 0.1  # Small reward instead of 0 to avoid gradient cliff

    def compute_proper_iou(box1, box2):
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_area = max(0.0, x2_inter - x1_inter) * max(0.0, y2_inter - y1_inter)

        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    gt_ious = []
    for gt_box in gt_boxes:
        best_iou = 0.0
        for pred_box in pred_boxes:
            iou = compute_proper_iou(gt_box, pred_box)
            best_iou = max(best_iou, iou)
        gt_ious.append(best_iou)

    pred_ious = []
    for pred_box in pred_boxes:
        best_iou = 0.0
        for gt_box in gt_boxes:
            iou = compute_proper_iou(pred_box, gt_box)
            best_iou = max(best_iou, iou)
        pred_ious.append(best_iou)

    recall = sum(gt_ious) / len(gt_ious) if gt_ious else 0.0
    precision = sum(pred_ious) / len(pred_ious) if pred_ious else 0.0

    if recall + precision > 0:
        f1_score = 2 * (recall * precision) / (recall + precision)
    else:
        f1_score = 0.0

    # Smoother reward transitions
    if f1_score >= 0.7:
        return 1.0
    elif f1_score >= 0.5:
        return 0.7 + 0.3 * (f1_score - 0.5) / 0.2
    elif f1_score >= 0.3:
        return 0.4 + 0.3 * (f1_score - 0.3) / 0.2
    else:
        return 0.1 + 0.3 * (f1_score / 0.3)  # Minimum reward of 0.1


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


def compute_score_improved(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Improved reward computation with smoother transitions and better stability
    """
    import json

    is_format_error = False

    # 1. Format checking with more lenient penalties
    count_think_1 = solution_str.count("<think>")
    count_think_2 = solution_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    predict_no_think = solution_str.split("</think>")[-1].strip() if "</think>" in solution_str else solution_str

    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    count_location_1 = predict_no_think.count("<location>")
    count_location_2 = predict_no_think.count("</location>")
    if count_location_1 != count_location_2:
        is_format_error = True

    count_type_1 = predict_no_think.count("<type>")
    count_type_2 = predict_no_think.count("</type>")
    if count_type_1 != count_type_2:
        is_format_error = True

    answer_text = extract_answer(predict_no_think)
    location_text = extract_location(predict_no_think)
    # type_text = extract_type(predict_no_think)

    if not answer_text:
        is_format_error = True
        answer_text = ""

    # Check bbox format
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

    # Smoother format penalty: -0.5 instead of -1.0
    format_reward = -0.5 if is_format_error else 0.0

    # 2. Answer correctness with retry mechanism
    if not client or not model_name:
        logger.warning("Reward function client not initialized or model name not found.")
        return format_reward

    ground_truth_answer = _extract_ground_truth_answer(ground_truth, extra_info)
    question_text = extra_info.get("question", "") if extra_info else ""

    # Penalize excessively long answers but less severely
    if len(answer_text) >= 1000:
        acc_reward = 0.1  # Small reward instead of 0
        format_reward = -0.3  # Reduced penalty
    elif not answer_text:
        acc_reward = 0.0
    else:
        # Use LLM judge with retry
        system_prompt = (
            "You are an expert evaluator for industrial defect detection. Your task is to determine if a model's "
            "answer is semantically equivalent to the standard answer.\n"
            "The answer should be 'yes' or 'no' (or variations like 'Yes. There has been a defect detected.').\n"
            'You must provide your final judgement as a single word: either "CORRECT" or "INCORRECT".'
        )

        user_prompt = (
            f"Evaluate if the model's answer matches the standard answer.\n\n"
            f"[Question]: {question_text}\n"
            f"[Standard Answer]: {ground_truth_answer}\n"
            f"[Model Answer]: {answer_text}\n\n"
            "Consider semantic equivalence, not exact word matching."
        )

        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens=50,
                seed=random.randint(0, 1000000),
                temperature=0.1,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            response = chat_response.choices[0].message.content.strip()

            if re.search(r"\bCORRECT\b", response, re.IGNORECASE):
                acc_reward = 1.0
            elif re.search(r"\bINCORRECT\b", response, re.IGNORECASE):
                acc_reward = 0.2  # Small reward for attempting
            else:
                logger.warning(f"LLM judge format error. Response: {response}")
                acc_reward = 0.1  # Small default reward
        except Exception as e:
            logger.warning(f"LLM judge request failed: {e}")
            acc_reward = 0.1  # Small default reward

    # 3. Bbox correctness reward with smoother transitions
    bbox_reward = 0.0
    if acc_reward > 0.15:  # Lower threshold for bbox evaluation
        pred_boxes = _parse_predicted_bboxes(location_text)
        gt_boxes = _extract_gt_bboxes(ground_truth, extra_info)

        bbox_iou_reward = _improved_iou_reward(pred_boxes, gt_boxes, max_pred_boxes=3)
        bbox_reward = 0.3 * (1.0 if bbox_format_ok else 0.5) + 0.7 * bbox_iou_reward

    # Final score with adjusted weights and normalization
    final_score = 0.2 * format_reward + 0.5 * acc_reward + 0.3 * bbox_reward

    # Apply reward normalization to reduce variance
    final_score = np.tanh(final_score)  # Squash to [-1, 1] range

    # Log for debugging
    if extra_info:
        logger.debug(
            f"Score breakdown: format={format_reward:.2f}, acc={acc_reward:.2f}, "
            f"bbox={bbox_reward:.2f}, final={final_score:.2f}"
        )

    return {
        "score": final_score,
        "format_reward": format_reward,
        "acc_reward": acc_reward,
        "bbox_reward": bbox_reward,
    }


# Keep the original function for compatibility
def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """Original compute_score function for backward compatibility"""
    result = compute_score_improved(data_source, solution_str, ground_truth, extra_info)
    if isinstance(result, dict):
        return result["score"]
    return result


if __name__ == "__main__":
    # Test the improved scoring function
    predict_str = """<think>
    I need to analyze this image for defects. Looking at the surface, I can see some irregularities.
    </think>
    <location>[{"bbox2d": [100, 150, 200, 250]}]</location>
    <type>crack</type>
    <answer>yes</answer>"""

    ground_truth = "yes"
    extra_info = {"question": "Are there any defects in this image?", "bboxes": [{"bbox2d": [105, 155, 195, 245]}]}

    print("=== Testing Improved Scoring Function ===")
    import time

    time_start = time.time()
    score = compute_score_improved("defect_detection", predict_str, ground_truth, extra_info)
    print(f"Score: {score}")
    time_end = time.time()
    print(f"Time: {time_end - time_start}")
