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
DeepEyes React Agent Loop with LangGraph integration.

This implementation uses LangGraph's workflow system with the ImageZoomInTool.
"""

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from recipe.langgraph_agent.react_agent_loop import ReactAgentLoop
from verl.tools.image_zoom_in_tool import ImageZoomInTool
from verl.tools.schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Global tool instance storage for stateful BaseTool integration
_tool_instances = {}


class ImageZoomInInput(BaseModel):
    """Input schema for image zoom in tool."""

    bbox_2d: list[float] = Field(
        description=(
            "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is "
            "the top-left corner and (x2, y2) is the bottom-right corner."
        ),
        min_length=4,
        max_length=4,
    )
    label: Optional[str] = Field(
        default=None,
        description="The name or label of the object in the specified bounding box (optional).",
    )


def create_langchain_tool_from_basetool(base_tool: Any, instance_kwargs: dict[str, Any]) -> StructuredTool:
    """
    Create a LangChain StructuredTool from a verl BaseTool.

    Args:
        base_tool: An instance of BaseTool (e.g., ImageZoomInTool)
        instance_kwargs: kwargs to pass to tool.create() for initialization

    Returns:
        A LangChain StructuredTool that wraps the BaseTool
    """

    async def tool_func(**kwargs) -> str:
        """Execute the tool with the given parameters."""
        # Get or create instance_id from RunnableConfig
        config: RunnableConfig = kwargs.pop("config", None)
        instance_id = None

        if config and "configurable" in config:
            instance_id = config["configurable"].get("instance_id")

        # Create instance if not exists
        if instance_id is None:
            instance_id = str(uuid4())
            logger.info(f"Creating new tool instance: {instance_id}")

        # Store instance_id in global registry if not exists
        if instance_id not in _tool_instances:
            try:
                created_id, _ = await base_tool.create(instance_id=instance_id, **instance_kwargs)
                _tool_instances[instance_id] = created_id
                logger.info(f"Tool instance created: {created_id}")
            except Exception as e:
                logger.error(f"Failed to create tool instance: {e}")
                return f"Error: Failed to initialize tool - {e}"

        # Execute tool
        try:
            tool_response, reward, metrics = await base_tool.execute(instance_id, kwargs)
            logger.info(f"Tool executed successfully. Reward: {reward}, Metrics: {metrics}")

            # Format response text
            result_text = tool_response.text or "Tool executed successfully."

            # Note: LangChain tools typically return strings
            # Image data will be handled separately in the agent loop
            return result_text

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"Error: Tool execution failed - {e}"

    # Create the tool with proper schema
    tool_name = base_tool.name
    tool_description = base_tool.tool_schema.function.description

    return StructuredTool(
        name=tool_name,
        description=tool_description,
        args_schema=ImageZoomInInput,
        coroutine=tool_func,
    )


class DeepEyesReactAgentLoop(ReactAgentLoop):
    """
    DeepEyes-specific React Agent Loop with ImageZoomInTool integration.

    This class extends ReactAgentLoop to use the ImageZoomInTool with LangGraph.
    """

    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        """Initialize the DeepEyes agent loop with ImageZoomInTool."""
        if cls._class_initialized:
            return
        print("Performing class-level DeepEyesReactAgentLoop initialization")

        # Initialize ImageZoomInTool
        tool_config = {
            "num_workers": 20,
            "rate_limit": 50,
            "timeout": 30,
            "enable_global_rate_limit": True,
        }

        # Create tool schema
        tool_schema = OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "image_zoom_in_tool",
                    "description": (
                        "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an "
                        "optional object label."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bbox_2d": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 4,
                                "maxItems": 4,
                                "description": (
                                    "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is "
                                    "the top-left corner and (x2, y2) is the bottom-right corner."
                                ),
                            },
                            "label": {
                                "type": "string",
                                "description": "The name or label of the object in the specified bounding box (optional).",
                            },
                        },
                        "required": ["bbox_2d"],
                    },
                },
            }
        )

        # Create BaseTool instance
        cls.image_zoom_tool = ImageZoomInTool(config=tool_config, tool_schema=tool_schema)

        # Store instance kwargs (will be populated per trajectory)
        cls.default_instance_kwargs = {}

        # Don't create LangChain tools yet - will be done in run() with specific instance kwargs
        cls.base_tools = [cls.image_zoom_tool]

        # Call parent init to build graph
        cls._class_initialized = True
        cls.graph = cls.build_graph()

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> Any:
        """
        Run the agent loop with image-specific initialization.

        Args:
            sampling_params: Sampling parameters for generation
            **kwargs: Should contain 'raw_prompt', 'multi_modal_data' with image, etc.

        Returns:
            AgentLoopOutput
        """
        # Extract image from kwargs
        multi_modal_data = kwargs.get("multi_modal_data", {})
        image = multi_modal_data.get("image", None)

        if image is None:
            logger.warning("No image provided in multi_modal_data, tool may fail")
            instance_kwargs = {}
        else:
            # Get the first image if it's a list
            first_image = image[0] if isinstance(image, list) else image
            instance_kwargs = {"image": first_image}

        # Create LangChain tool wrapper with instance-specific kwargs
        langchain_tool = create_langchain_tool_from_basetool(self.image_zoom_tool, instance_kwargs)

        # Temporarily set cls.tools for this run
        self.tools = [langchain_tool]

        # Call parent run method
        result = await super().run(sampling_params, **kwargs)

        # Cleanup: release tool instances
        # Note: This is a simplified cleanup - in production you might want more sophisticated lifecycle management
        for instance_id in list(_tool_instances.keys()):
            try:
                await self.image_zoom_tool.release(instance_id)
                del _tool_instances[instance_id]
            except Exception as e:
                logger.warning(f"Failed to release tool instance {instance_id}: {e}")

        return result
