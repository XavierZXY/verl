#!/usr/bin/env python3
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
Example usage of DeepEyes React Agent Loop.

This script demonstrates how to use the DeepEyes agent loop for vision-based tasks.
"""

import asyncio

from PIL import Image


# Example 1: Simple standalone usage
async def example_standalone():
    """Example of using the agent loop in standalone mode."""
    from recipe.deepeyes.deepeyes_react_agent_loop import (
        create_langchain_tool_from_basetool,
    )
    from verl.tools.image_zoom_in_tool import ImageZoomInTool
    from verl.tools.schemas import OpenAIFunctionToolSchema

    print("=" * 60)
    print("Example 1: Standalone Agent Loop Usage")
    print("=" * 60)

    # 1. Create ImageZoomInTool
    tool_config = {
        "num_workers": 20,
        "rate_limit": 50,
        "timeout": 30,
        "enable_global_rate_limit": False,  # Disable for simple usage
    }

    tool_schema = OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "image_zoom_in_tool",
                "description": "Zoom in on a specific region of an image by cropping it based on a bounding box.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bbox_2d": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                            "description": "Bounding box as [x1, y1, x2, y2]",
                        },
                        "label": {
                            "type": "string",
                            "description": "Object label (optional)",
                        },
                    },
                    "required": ["bbox_2d"],
                },
            },
        }
    )

    image_tool = ImageZoomInTool(config=tool_config, tool_schema=tool_schema)

    # 2. Create a test image
    test_image = Image.new("RGB", (640, 480), color="blue")

    # 3. Create LangChain wrapper
    langchain_tool = create_langchain_tool_from_basetool(image_tool, instance_kwargs={"image": test_image})

    # 4. Test tool execution
    result = await langchain_tool.ainvoke({"bbox_2d": [100, 100, 300, 300], "label": "test_region"})
    print(f"Tool result: {result}")

    print("\nExample 1 completed!\n")


# Example 2: Dataset integration
def example_dataset():
    """Example of using the agent loop with dataset."""

    print("=" * 60)
    print("Example 2: Dataset Integration")
    print("=" * 60)

    print("""
The CustomRLHFDataset has been updated to work with the agent loop:

1. Before (ToolAgentLoop):
   ```python
   tools_kwargs = {
       "image_zoom_in_tool": {
           "create_kwargs": {"image": images[0]},
       }
   }
   row_dict["tools_kwargs"] = tools_kwargs
   row_dict["agent_name"] = "tool_agent"
   ```

2. After (ReactAgentLoop):
   ```python
   # Image is automatically extracted from multi_modal_data
   row_dict["agent_name"] = "deepeyes_agent"
   ```

The agent loop will:
- Extract the image from `multi_modal_data`
- Initialize the tool with the image
- Manage the tool lifecycle automatically
    """)

    print("\nExample 2 completed!\n")


# Example 3: Training configuration
def example_training_config():
    """Example of training configuration."""
    print("=" * 60)
    print("Example 3: Training Configuration")
    print("=" * 60)

    print("""
To use the DeepEyes agent loop in training:

1. Create agent_loop_config.json:
   ```json
   [
     {
       "_target_": "recipe.deepeyes.deepeyes_react_agent_loop.DeepEyesReactAgentLoop",
       "name": "deepeyes_agent"
     }
   ]
   ```

2. Update your training script:
   ```bash
   actor_rollout_ref.rollout.multi_turn.enable=True \\
   actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \\
   actor_rollout_ref.rollout.multi_turn.max_user_turns=5 \\
   actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \\
   actor_rollout_ref.rollout.multi_turn.format=hermes \\
   actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/deepeyes/configs/agent_loop_config.json
   ```

3. Run training:
   ```bash
   bash recipe/deepeyes/run_deepeyes_agent_grpo.sh
   ```
    """)

    print("\nExample 3 completed!\n")


# Example 4: Comparison with langgraph_agent
def example_comparison():
    """Compare with langgraph_agent implementation."""
    print("=" * 60)
    print("Example 4: Comparison with langgraph_agent")
    print("=" * 60)

    print("""
Similarities with langgraph_agent:
----------------------------------
1. Both use LangGraph's StateGraph for workflow
2. Both use LangChain's tool interface
3. Both support multi-turn conversations
4. Both integrate with verl's training pipeline

Key Differences:
----------------
1. DeepEyes uses stateful BaseTool (ImageZoomInTool)
   - Requires instance lifecycle management
   - Needs image initialization per trajectory

2. LangGraph example uses stateless tools
   - Simple function decorators
   - No state management needed

3. DeepEyes adds tool wrapper layer
   - Converts BaseTool to LangChain StructuredTool
   - Manages instance_id internally

Tool Integration Pattern:
------------------------
langgraph_agent:
    @tool
    def calculate(a: int, b: int) -> int:
        return a + b

deepeyes:
    class ImageZoomInTool(BaseTool):
        async def create(self, instance_id, **kwargs):
            # Initialize with image
        async def execute(self, instance_id, parameters):
            # Perform zoom operation
        async def release(self, instance_id):
            # Cleanup
    
    # Wrapped as LangChain tool:
    langchain_tool = create_langchain_tool_from_basetool(tool, kwargs)
    """)

    print("\nExample 4 completed!\n")


# Main execution
def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("DeepEyes React Agent Loop - Usage Examples")
    print("=" * 60 + "\n")

    # Run async example
    print("Running async example...")
    asyncio.run(example_standalone())

    # Run other examples
    example_dataset()
    example_training_config()
    example_comparison()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
