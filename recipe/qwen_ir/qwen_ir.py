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

import requests
from openai import OpenAI
from PIL import Image

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

SYSTEM_PROMPT: str = (
    "You are an expert system specialized in identifying image degradation.\n"
    "Task: Analyze the provided image and identify all degradation types present. Constraints: You must choose only from the following list:\n"
    "[haze, rain, low resolution, defocus blur, motion blur, dark, noise, jpeg compression artifact]\n"
    "Your response must follow the format below:\n"
    "1.  **`<think></think>`**: Provide your step-by-step reasoning about the image quality and degradation.\n"
    '2.  **`<answer></answer>`**: Provide the degradation type(s). If no degradation is found, use "null". '
    'If multiple degradation types are found, list them separated by commas (e.g., "motion blur, noise"). '
    'If only one degradation type is found, provide just that type (e.g., "motion blur"). '
    'Degradation types must be from the list: "haze", "rain", "low resolution", "defocus blur", "motion blur", "dark", "noise", "jpeg compression artifact".\n'
    "** Example 1: Degradation Found (single type):**\n"
    "<think>The blur pattern shows directional streaks characteristic of motion blur. "
    "I can confidently identify this degradation type.</think>\n"
    "<answer>motion blur</answer>\n"
    "** Example 2: Degradation Found (multiple types):**\n"
    "<think>I observe both motion blur in the moving objects and noise artifacts throughout the image. "
    "Both degradation types are present.</think>\n"
    "<answer>motion blur, noise</answer>\n"
    "** Example 3: No Degradation:**\n"
    "<think>I have thoroughly analyzed the entire image. The image is sharp, clear, and shows no signs of haze, blur, noise, "
    "or any other degradation artifacts. The image quality meets high standards.</think>\n"
    "<answer>null</answer>\n"
)

USER_PROMPT: list[str] = [
    "<image>.\nPlease analyze the following image and identify all degradation types present. Return the degradation type(s) in the format specified below.",
    "<image>.\nAnalyze the image and identify all degradation types present.",
    "<image>.\nWhat are the degradation types present in the image?",
    "<image>.\nCan you identify the degradation types present in the image?",
    "<image>.\nAnalyze the image and identify all degradation types present. Return the degradation type(s) in the format specified below.",
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
            logger.warning(
                "No models found at the specified API base for reward scoring."
            )
    except (requests.exceptions.RequestException, KeyError, IndexError) as e:
        logger.warning(
            f"Failed to get model from {openai_api_base}: {e}. Reward scoring will be disabled."
        )


class CustomRLHFDataset(RLHFDataset):
    def __getitem__(self, item):
        """
        Modified to only extract images and env_name from data.
        images: image data
        env_name: degradation types present in the image
        """
        row_dict: dict = self.dataframe[item]

        # Extract env_name (degradation types)
        env_name = row_dict.get("env_name", "")

        # Build prompt with system and user messages
        row_dict[self.prompt_key] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                # randomly select a user prompt
                "content": random.choice(USER_PROMPT),
            },
        ]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                images = [
                    Image.open(io.BytesIO(image["bytes"])) for image in row_dict_images
                ]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205  # noqa: E501
                multi_modal_data["image"] = images

            model_inputs = self.processor(
                text=[raw_prompt], images=images, return_tensors="pt"
            )

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            # Don't pop second_per_grid_ts here - we need it for get_rope_index later

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            model_inputs = self.tokenizer(
                raw_prompt, return_tensors="pt", add_special_tokens=False
            )
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

        if (
            self.processor is not None
            and "Qwen2VLImageProcessor"
            in self.processor.image_processor.__class__.__name__
        ):
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
                raw_prompt_ids = (
                    raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
                )
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                )

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # Store env_name as ground_truth in reward_model for reward computation
        row_dict["reward_model"] = {"ground_truth": env_name}

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        row_dict["agent_name"] = "tool_agent"

        return row_dict


def extract_answer(text):
    """Extract content from <answer></answer> tags."""
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def compute_score(
    data_source: str, solution_str: str, ground_truth: str, extra_info=None
) -> float:
    """
    Compute reward score for image degradation identification task.

    The score consists of two components:
    1. Format reward: Hard reward (1.0 if no errors, 0.0 if any errors)
    2. Answer correctness reward: Compare predicted degradation types with env_name

    Args:
        data_source: Source of the data (not used currently)
        solution_str: Model's solution string
        ground_truth: Ground truth answer (env_name from dataset, contains degradation types)
        extra_info: Additional information (optional)

    Returns:
        dict: Dictionary containing score and reward components
    """
    # 1. Format reward: Check tag pairing
    is_format_error = False

    # Check <think> tags
    count_think_1 = solution_str.count("<think>")
    count_think_2 = solution_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    # Check <answer> tags
    count_answer_1 = solution_str.count("<answer>")
    count_answer_2 = solution_str.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    format_reward = 1.0 if not is_format_error else 0.0

    # 2. Extract answer and compute acc_reward
    answer_text = ""
    answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()

    # Parse answer as degradation types (multiple types separated by commas, or "null" if none)
    if answer_text:
        answer_normalized = answer_text.strip().lower()

        # Parse answer: split by comma and normalize
        if answer_normalized == "null":
            pred_types = set()
        else:
            pred_types = set([t.strip() for t in answer_normalized.split(",")])
            pred_types.discard("")  # Remove empty strings

        # Parse ground_truth: ground_truth parameter contains env_name (degradation types)
        # ground_truth is passed from dataset where env_name is stored as ground_truth in reward_model
        # Can be str (comma-separated like "dark, haze, rain"), list, or dict
        # First, normalize ground_truth to a list format
        gt_list = []
        if isinstance(ground_truth, list):
            # If ground_truth is already a list, use it directly
            gt_list = ground_truth
        elif isinstance(ground_truth, dict):
            # If ground_truth is a dict, extract env_name first
            env_name = ground_truth.get("env_name", "")
            if isinstance(env_name, list):
                gt_list = env_name
            elif isinstance(env_name, str):
                # Convert comma-separated string to list
                env_name_normalized = env_name.strip().lower() if env_name else ""
                if env_name_normalized and env_name_normalized != "null":
                    gt_list = [t.strip() for t in env_name_normalized.split(",")]
        elif isinstance(ground_truth, str):
            # Convert comma-separated string to list
            env_name_normalized = ground_truth.strip().lower() if ground_truth else ""
            if env_name_normalized and env_name_normalized != "null":
                gt_list = [t.strip() for t in env_name_normalized.split(",")]
        else:
            # For other types, convert to string first, then to list
            env_name = str(ground_truth) if ground_truth is not None else ""
            env_name_normalized = env_name.strip().lower() if env_name else ""
            if env_name_normalized and env_name_normalized != "null":
                gt_list = [t.strip() for t in env_name_normalized.split(",")]

        # print(f"gt_list: {gt_list}")

        # Now convert the list to a normalized set (lowercase, no empty strings)
        gt_types = set([str(item).strip().lower() for item in gt_list if item])
        gt_types.discard("")

        # Compute accuracy: match if predicted types exactly match ground truth types
        if pred_types == gt_types:
            acc_reward = 1.0
        elif len(pred_types) == 0 and len(gt_types) == 0:
            acc_reward = 1.0  # Both are null/empty
        else:
            # Partial credit based on intersection over union
            intersection = pred_types & gt_types
            union = pred_types | gt_types
            if len(union) > 0:
                acc_reward = float(len(intersection)) / float(len(union))
            else:
                acc_reward = 0.0
    else:
        acc_reward = 0.0

    # 3. Check length limit: if <think> or <answer> content exceeds 512, set acc_reward to 0
    reasoning_text = ""
    reasoning_match = re.search(r"<think>(.*?)</think>", solution_str, re.DOTALL)
    if reasoning_match:
        reasoning_text = reasoning_match.group(1).strip()

    if len(reasoning_text) > 2048 or len(answer_text) > 120:
        acc_reward = 0.0
        logger.debug(
            f"Length exceeded: reasoning_len={len(reasoning_text)}, "
            f"answer_len={len(answer_text)}. Setting acc_reward to 0."
        )

    # Final score calculation
    # Weighted combination: format (0.5), acc (0.5)
    final_score = 0.2 * format_reward + 0.8 * acc_reward

    # Log for debugging
    logger.debug(
        f"Score breakdown: format={format_reward:.2f}, acc={acc_reward:.2f}, "
        f"final={final_score:.2f}"
    )

    return {
        "score": final_score,
        "format_reward": format_reward,
        "acc_reward": acc_reward,
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

    ground_truth_1 = {
        "answer": "no",
        "bboxes": [
            {"bbox_2d": [590, 670, 280, 700]},
            {"bbox_2d": [590, 770, 280, 795]},
        ],
    }
    extra_info_1 = {
        "question": "Does this image contain any defects?",
        "bboxes": [
            {"bbox_2d": [590, 670, 280, 700]},
            {"bbox_2d": [590, 770, 280, 795]},
        ],
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
    score2 = compute_score(
        "defect_detection", test_case_2, ground_truth_2, extra_info_2
    )
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
    score3 = compute_score(
        "defect_detection", test_case_3, ground_truth_2, extra_info_2
    )
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
    score4 = compute_score(
        "defect_detection", test_case_4, ground_truth_1, extra_info_1
    )
    print(f"Score: {score4}")
    time_end = time.time()
    print(f"Time: {time_end - time_start}")
