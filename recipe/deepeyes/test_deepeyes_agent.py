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
"""
Test script for DeepEyes React Agent Loop.
"""

import json
import os

import numpy as np
import pytest
import ray
from omegaconf import DictConfig

from tests.experimental.agent_loop.agent_utils import init_agent_loop_manager
from verl.protocol import DataProto
from verl.utils import hf_tokenizer


@pytest.fixture
def init_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(config_name="ppo_trainer")

    # Use a vision-language model (Qwen2-VL or similar)
    model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2-VL-2B-Instruct")
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.name = os.getenv("ROLLOUT_NAME", "vllm")
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.prompt_length = 8192
    config.actor_rollout_ref.rollout.response_length = 4096
    config.actor_rollout_ref.rollout.n = 2
    config.actor_rollout_ref.rollout.agent.num_workers = 2

    # Enable multi-turn for agent loop
    config.actor_rollout_ref.rollout.multi_turn.enable = True
    config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns = 5
    config.actor_rollout_ref.rollout.multi_turn.max_user_turns = 5
    config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls = 1
    config.actor_rollout_ref.rollout.multi_turn.format = "hermes"

    config.actor_rollout_ref.actor.use_dynamic_bsz = True

    return config


def test_deepeyes_agent(init_config):
    """Test DeepEyes agent loop with image zoom in tool."""
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )

    # Create agent loop config
    agent_loop_config = [
        {
            "_target_": "recipe.deepeyes.deepeyes_react_agent_loop.DeepEyesReactAgentLoop",
            "name": "deepeyes_agent",
        },
    ]
    agent_loop_config_path = "/tmp/deepeyes_agent_loop_config.json"
    with open(agent_loop_config_path, "w") as f:
        json.dump(agent_loop_config, f)

    # Update config
    n = 2
    init_config.actor_rollout_ref.rollout.n = n
    init_config.actor_rollout_ref.rollout.agent.agent_loop_config_path = agent_loop_config_path

    # Initialize agent loop manager
    agent_loop_manager = init_agent_loop_manager(init_config)

    # Create test prompts with vision questions
    raw_prompts = [
        [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. You can call functions to assist with the user query. "
                    "Important: You must call only one function at a time. After each function call, "
                    "wait for the execution result before making the next function call if needed."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in the top-left corner of this image?",
                    },
                    {"type": "image"},
                ],
            },
        ],
        [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. You can call functions to assist with the user query. "
                    "Important: You must call only one function at a time."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Can you zoom in on the region [100, 100, 300, 300] and tell me what you see?",
                    },
                    {"type": "image"},
                ],
            },
        ],
    ]

    # Create dummy images for testing (you should replace with real images)
    from PIL import Image

    dummy_image = Image.new("RGB", (640, 480), color="red")

    # Create batch
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array([np.array(prompt) for prompt in raw_prompts], dtype=object),
            "agent_name": np.array(["deepeyes_agent"] * len(raw_prompts)),
            "data_source": np.array(["deepeyes"] * len(raw_prompts)),
            "multi_modal_data": np.array([{"image": [dummy_image]}] * len(raw_prompts), dtype=object),
            "reward_model": np.array([{"style": "rule", "ground_truth": "test"}] * len(raw_prompts)),
        },
    )
    batch = batch.repeat(n)

    # Generate sequences
    result = agent_loop_manager.generate_sequences(prompts=batch)
    assert len(result) == len(raw_prompts) * n

    # Check num_turns
    num_turns = result.non_tensor_batch["__num_turns__"]
    print(f"num_turns: {num_turns}")

    # Decode and print responses
    tokenizer = hf_tokenizer(init_config.actor_rollout_ref.model.path)
    responses = result.batch["responses"]
    response_mask = result.batch["response_mask"]
    attention_mask = result.batch["attention_mask"]

    for i in range(len(responses)):
        response_length = response_mask.size(1)
        valid_tokens = responses[i][attention_mask[i][-response_length:].bool()]
        response_with_obs = tokenizer.decode(valid_tokens)

        valid_tokens = responses[i][response_mask[i].bool()]
        response_without_obs = tokenizer.decode(valid_tokens)

        print("=" * 50)
        print(f"Sample {i}:")
        print(f"Response with observations:\n{response_with_obs}")
        print("-" * 50)
        print(f"Response without observations:\n{response_without_obs}")
        print("=" * 50)

    print("Test passed!")
    ray.shutdown()


if __name__ == "__main__":
    # Run test
    pytest.main([__file__, "-v", "-s"])
