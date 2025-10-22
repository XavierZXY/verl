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
from typing import Any, Optional
from uuid import uuid4

from PIL import Image

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ImageReferenceTool(BaseTool):
    """A tool for retrieving the reference image for quality comparison.

    This tool provides access to a reference image of the same class,
    which can be used for comparison during quality control inspection.
    The reference image can be either defect-free or annotated with defect markers.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the reference image retrieval operation
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "image_reference_tool",
                "description": (
                    "Retrieve a defect-free reference image of the same object class for comparison. "
                    "This helps identify anomalies by comparing the current image with a known good sample."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": (
                                "Optional explanation of why you need the reference image "
                                "(e.g., 'to compare surface texture', 'to verify if the spot is normal')."
                            ),
                        },
                    },
                    "required": [],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        logger.info(f"Initialized ImageReferenceTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """
        Creates a new instance for image reference tool.

        This method initializes a new session with a reference image, which can then
        be retrieved when needed for comparison.

        Args:
            instance_id: An optional unique identifier for the instance. If not
                provided, a new UUID will be generated.
            **kwargs: Should contain 'reference_image' key with image bytes data,
                or 'create_kwargs' containing {'reference_image': image_bytes}.
                The image bytes should be in a format that PIL can read (JPEG, PNG, etc.)

        Returns:
            Tuple of (instance_id, ToolResponse)
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Handle create_kwargs parameter if passed
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        # Get reference image from kwargs
        reference_image_bytes = kwargs.get("reference_image")
        
        if reference_image_bytes is None:
            logger.warning(f"No reference_image provided for instance {instance_id}")
            # Store None to indicate no reference image available
            self._instance_dict[instance_id] = {
                "reference_image": None,
                "available": False,
            }
            return instance_id, ToolResponse()

        # Load the reference image from bytes
        try:
            if isinstance(reference_image_bytes, bytes):
                reference_img = Image.open(io.BytesIO(reference_image_bytes))
            else:
                logger.error(f"Invalid reference_image type: {type(reference_image_bytes)}")
                self._instance_dict[instance_id] = {
                    "reference_image": None,
                    "available": False,
                }
                return instance_id, ToolResponse()
            
            self._instance_dict[instance_id] = {
                "reference_image": reference_img,
                "available": True,
            }
            logger.info(f"Successfully loaded reference image for instance {instance_id}, size: {reference_img.size}")
            
        except Exception as e:
            logger.error(f"Failed to load reference image: {e}")
            self._instance_dict[instance_id] = {
                "reference_image": None,
                "available": False,
            }
            
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """
        Execute the reference image retrieval operation.

        Args:
            instance_id: The instance identifier
            parameters: Dictionary containing optional 'reason' parameter
            **kwargs: Additional keyword arguments

        Returns:
            Tuple of (ToolResponse, reward, metrics_dict)
            - ToolResponse: Contains the reference image and descriptive text
            - reward: Small positive reward (0.0) for using the reference tool
            - metrics_dict: Dictionary with success status and metadata
        """
        reason = parameters.get("reason", "")
        
        if instance_id not in self._instance_dict:
            return (
                ToolResponse(text="Error: Tool instance not found. Please create an instance first."),
                -0.05,
                {"success": False, "error": "instance_not_found"},
            )

        instance_data = self._instance_dict[instance_id]
        
        if not instance_data.get("available", False):
            return (
                ToolResponse(
                    text="No reference image is available for this sample. "
                         "This might be a good sample without a reference, or the reference image failed to load."
                ),
                0.0,
                {"success": False, "reference_available": False},
            )

        reference_image = instance_data["reference_image"]
        
        if reference_image is None:
            return (
                ToolResponse(text="Error: Reference image is not available."),
                -0.05,
                {"success": False, "reference_available": False},
            )

        # Create a copy of the reference image to avoid "Operation on closed image" error
        # This ensures the returned image is independent of the stored reference
        try:
            reference_image_copy = reference_image.copy()
            logger.info(f"Returning reference image copy for instance {instance_id}, size: {reference_image_copy.size}")
        except Exception as e:
            logger.error(f"Failed to copy reference image for instance {instance_id}: {e}")
            return (
                ToolResponse(text="Error: Failed to copy reference image."),
                -0.05,
                {"success": False, "error": "copy_failed"},
            )

        # Construct response text
        # response_text = "Here is a reference image of the same object class for comparison."
        
        if reason:
            response_text = f"Retrieved reference image for: {reason}"
        # for bad reference image, we need to add a note that the reference image is a bad reference image
        response_text = (
            "Here is a reference image of the same object class for comparison.The defect is marked in red rectangle."
            " You should pay attention to the defect. Learn the defect patterns."
        )

        return (
            ToolResponse(
                image=[reference_image_copy],
                text=response_text,
            ),
            0.0,  # Neutral reward for retrieving reference
            {"success": True, "reference_available": True, "reason": reason},
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance and clean up resources."""
        if instance_id in self._instance_dict:
            # Close the image if it exists
            instance_data = self._instance_dict[instance_id]
            if instance_data.get("reference_image") is not None:
                try:
                    instance_data["reference_image"].close()
                except Exception:
                    pass
            del self._instance_dict[instance_id]
            logger.debug(f"Released instance {instance_id}")

