# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import asyncio
import copy
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Any,
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # Save original image for tracking and current image for tool access
        # Get the first image if it's a list, otherwise use the image directly
        if image_data is not None:
            if isinstance(image_data, list) and len(image_data) > 0:
                self.original_image = image_data[0]
                self.current_image = image_data[0]  # Start with original, update after each zoom
            else:
                self.original_image = image_data
                self.current_image = image_data  # Start with original, update after each zoom
            logger.info(
                f"[DEBUG] AgentData initialized with original_image: type={type(self.original_image)}, "
                f"is_PIL={hasattr(self.original_image, 'size')}, "
                f"size={getattr(self.original_image, 'size', None)}"
            )
        else:
            self.original_image = None
            self.current_image = None
            logger.info(f"[DEBUG] AgentData initialized with original_image=None")

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        # Multi-turn conversation tracking for logging
        self.conversation_history: list[dict[str, Any]] = []
        
        # Zoom tracking for coordinate transformation
        self.zoom_offsets: list[tuple[float, float]] = []  # List of (x_offset, y_offset) from each zoom call


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        print(f"Initialized tools: {cls.tools}")

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )
        # Initialize interactions from config file
        cls.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if cls.interaction_config_file:
            cls.interaction_map: dict[str, BaseInteraction] = cls._initialize_interactions(cls.interaction_config_file)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        # Always use original_image to match the image used during generation
        # This ensures image tokens in prompt_ids match the image features in multi_modal_data
        # (Different image sizes generate different numbers of vision tokens in Qwen2-VL)
        multi_modal_data = {"image": [agent_data.original_image]} if agent_data.original_image is not None else {}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={},
        )
        # Debug: log conversation_history before returning
        logger.debug(f"[DEBUG] Final conversation_history has {len(agent_data.conversation_history)} entries")
        for i, entry in enumerate(agent_data.conversation_history):
            has_original_img = "original_image" in entry
            has_cropped_imgs = "cropped_images" in entry
            original_img_type = type(entry.get("original_image")) if has_original_img else None
            logger.debug(
                f"[DEBUG] conversation_history[{i}]: role={entry.get('role')}, "
                f"has_original_image={has_original_img}, original_img_type={original_img_type}, "
                f"has_cropped_images={has_cropped_imgs}, "
                f"cropped_images_count={len(entry.get('cropped_images', []))}"
            )

        output.extra_fields.update(
            {
                "turn_scores": agent_data.turn_scores,
                "tool_rewards": agent_data.tool_rewards,
                "conversation_history": agent_data.conversation_history,
                "zoom_offsets": agent_data.zoom_offsets,  # For coordinate transformation in reward calculation
            }
        )
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # Always use original_image to maintain consistent vision token counts across turns
            # (Qwen2-VL generates different token counts for different image sizes)
            # current_image is used only for zoom tool's progressive cropping
            image_for_generation = [agent_data.original_image] if agent_data.original_image is not None else None
            if image_for_generation:
                logger.info(
                    f"[PENDING] Using original_image for generation, size: {agent_data.original_image.size}, "
                    f"current_image_size: {agent_data.current_image.size if agent_data.current_image else 'None'}"
                )
            model_inputs = self.processor(text=[raw_prompt], images=image_for_generation, return_tensors="pt")
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        # Always use original_image to maintain consistent vision token counts across turns
        # (Qwen2-VL generates different token counts for different image sizes)
        # current_image is used only for zoom tool's progressive cropping
        image_for_generation = [agent_data.original_image] if agent_data.original_image is not None else None
        if image_for_generation:
            logger.info(
                f"[GENERATING] Turn {agent_data.assistant_turns + 1}: Using original_image for generation, "
                f"size: {agent_data.original_image.size}, "
                f"current_image_size: {agent_data.current_image.size if agent_data.current_image else 'None'}"
            )
        
        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=image_for_generation,
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        # Decode assistant response for logging
        assistant_message_text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        )

        # Record to conversation history for logging
        agent_data.conversation_history.append(
            {
                "role": "assistant",
                "content": assistant_message_text,
                "turn": agent_data.assistant_turns,
            }
        )

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        # Handle interaction if needed
        if self.interaction_config_file:
            add_messages.append({"role": "assistant", "content": assistant_message_text})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []

        tasks = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data))

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for idx, (tool_response, tool_reward, tool_metadata) in enumerate(responses):
            # Create message from tool response
            # IMPORTANT: Don't include image reference in tool message to avoid multiple image tokens in prompt_ids
            # The model will see the updated current_image through agent_data.current_image
            # Only keep text description in the message
            message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)
            agent_data.messages.extend(add_messages)

            # Record tool response to conversation history for logging
            tool_history_entry = {
                "role": "tool",
                "content": tool_response.text or "",
                "turn": agent_data.user_turns + 1,  # Will be incremented later
                "tool_name": agent_data.tool_calls[idx].name if idx < len(agent_data.tool_calls) else "unknown",
                "tool_success": tool_metadata.get("success", True),
            }

            # Add original image directly from agent_data for tracking
            # This ensures we always have the correct original image for comparison
            if agent_data.original_image is not None:
                tool_history_entry["original_image"] = agent_data.original_image
                logger.debug(
                    f"[DEBUG] Added original_image to tool_history_entry[{idx}]: "
                    f"type={type(agent_data.original_image)}, "
                    f"is_PIL={hasattr(agent_data.original_image, 'size')}, "
                    f"size={getattr(agent_data.original_image, 'size', None)}"
                )
            else:
                logger.warning(f"[DEBUG] agent_data.original_image is None for tool_history_entry[{idx}]")

            # Handle image data for logging purposes
            # Note: Images are NOT included in the message to avoid multiple image tokens in prompt_ids
            if tool_response.image:
                if agent_data.image_data is None:
                    agent_data.image_data = []
                elif not isinstance(agent_data.image_data, list):
                    agent_data.image_data = [agent_data.image_data]

                # Add new image data to history for logging
                cropped_images_for_logging = []
                if isinstance(tool_response.image, list):
                    for img in tool_response.image:
                        if img is not None:
                            agent_data.image_data.append(img)
                            cropped_images_for_logging.append(img)
                else:
                    if tool_response.image is not None:
                        agent_data.image_data.append(tool_response.image)
                        cropped_images_for_logging.append(tool_response.image)

                # Add cropped images to history entry for logging
                tool_history_entry["cropped_images"] = cropped_images_for_logging

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)
                tool_history_entry["tool_reward"] = tool_reward

            # Add to conversation history
            agent_data.conversation_history.append(tool_history_entry)

        # Update prompt with tool responses
        if self.processor is not None:
            raw_tool_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # Don't pass images when encoding tool response since we removed image references from messages
            # This ensures prompt_ids only contains one image token (from initial user message)
            model_inputs = self.processor(text=[raw_tool_response], images=None, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
            )
        response_ids = response_ids[len(self.system_prompt) :]
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        if self.processor is not None:
            raw_user_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_user_response], images=None, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
            )
        response_ids = response_ids[len(self.system_prompt) :]

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(self, tool_call: FunctionCall, agent_data: AgentData) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]

            # Prepare create_kwargs with image data
            kwargs = agent_data.tools_kwargs.get(tool_name, {})
            create_kwargs = kwargs.get("create_kwargs", {})

            # For zoom tool, use current_image (which may be a cropped image from previous zoom)
            # For other tools, use original_image
            if tool_name == "image_zoom_in_tool":
                if agent_data.current_image is not None:
                    create_kwargs["image"] = agent_data.current_image
            else:
                # For other tools like image_reference_tool, use original image
                if agent_data.original_image is not None:
                    create_kwargs["image"] = agent_data.original_image

            instance_id, _ = await tool.create(create_kwargs=create_kwargs)
            tool_execution_response, tool_reward, res = await tool.execute(instance_id, tool_args)
            
            # If this is a zoom tool and it succeeded, update current_image and track offset
            if tool_name == "image_zoom_in_tool" and res.get("success", False):
                # The cropped image is in tool_execution_response.image
                if tool_execution_response.image:
                    if isinstance(tool_execution_response.image, list):
                        agent_data.current_image = tool_execution_response.image[0]
                    else:
                        agent_data.current_image = tool_execution_response.image
                    logger.info(f"Updated current_image after zoom, new size: {agent_data.current_image.size}")
                
                # Track the offset for coordinate transformation
                offset = res.get("offset", [0.0, 0.0])
                agent_data.zoom_offsets.append((float(offset[0]), float(offset[1])))
                logger.info(f"Recorded zoom offset: {offset}, total offsets: {len(agent_data.zoom_offsets)}")
                
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    @classmethod
    def _initialize_interactions(cls, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        logger.info(f"Initialize interactions from configuration: interaction_map: {list(interaction_map.keys())}")
        return interaction_map
