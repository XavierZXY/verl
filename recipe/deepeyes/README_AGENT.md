# DeepEyes Agent Loop Implementation

This directory contains an implementation of the DeepEyes training pipeline using LangGraph-style agent loops, similar to the `langgraph_agent` recipe.

## Overview

The agent loop implementation enables the model to interact with the `ImageZoomInTool` in a multi-turn conversation format using LangGraph's workflow system.

## Architecture

### Key Components

1. **DeepEyesReactAgentLoop** (`deepeyes_react_agent_loop.py`)
   - Extends `ReactAgentLoop` from `langgraph_agent`
   - Integrates `ImageZoomInTool` as a LangChain StructuredTool
   - Manages tool instance lifecycle with image data
   - Uses LangGraph's state machine for agent execution

2. **Tool Wrapper**
   - Converts verl's `BaseTool` (stateful) to LangChain's `StructuredTool` (stateless)
   - Manages tool instance creation and cleanup
   - Handles image data initialization per trajectory

3. **Dataset Integration** (`deepeyes.py`)
   - Updated `CustomRLHFDataset` to work with agent loops
   - Removes manual `tools_kwargs` configuration
   - Passes image data via `multi_modal_data` for agent loop initialization

## Files

- `deepeyes_react_agent_loop.py` - Main agent loop implementation
- `configs/agent_loop_config.json` - Agent configuration for training
- `test_deepeyes_agent.py` - Test script for the agent loop
- `run_deepeyes_agent_grpo.sh` - Training script using agent loop
- `README_AGENT.md` - This file

## Usage

### Training with Agent Loop

Use the updated training script:

```bash
bash recipe/deepeyes/run_deepeyes_agent_grpo.sh
```

Key differences from the original script:
- Uses `actor_rollout_ref.rollout.agent.agent_loop_config_path` instead of `tool_config_path`
- No longer needs separate tool configuration
- Agent loop handles tool initialization automatically

### Testing

Run the test script to verify agent loop functionality:

```bash
python recipe/deepeyes/test_deepeyes_agent.py
```

Or with pytest:

```bash
pytest recipe/deepeyes/test_deepeyes_agent.py -v -s
```

## Comparison: Tool Agent vs React Agent

### Original ToolAgentLoop Approach

- Uses verl's `BaseTool` directly
- Maintains tool state via `instance_id`
- Requires `tools_kwargs` for tool initialization
- Tool lifecycle: create → execute → release

### New ReactAgentLoop Approach

- Uses LangChain's `StructuredTool` interface
- Wraps `BaseTool` with stateful instance management
- Tool initialization via agent loop parameters
- Integrated with LangGraph's workflow system

## Configuration

### Agent Loop Config (`configs/agent_loop_config.json`)

```json
[
  {
    "_target_": "recipe.deepeyes.deepeyes_react_agent_loop.DeepEyesReactAgentLoop",
    "name": "deepeyes_agent"
  }
]
```

### Training Config Updates

In your training script, ensure these parameters are set:

```bash
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5
actor_rollout_ref.rollout.multi_turn.max_user_turns=5
actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
actor_rollout_ref.rollout.multi_turn.format=hermes
actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/deepeyes/configs/agent_loop_config.json
```

## How It Works

### 1. Initialization

```python
class DeepEyesReactAgentLoop(ReactAgentLoop):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        # Initialize ImageZoomInTool
        cls.image_zoom_tool = ImageZoomInTool(config=tool_config, tool_schema=tool_schema)
        # Build LangGraph workflow
        cls.graph = cls.build_graph()
```

### 2. Per-Trajectory Execution

```python
async def run(self, sampling_params: dict[str, Any], **kwargs):
    # Extract image from multi_modal_data
    image = kwargs.get("multi_modal_data", {}).get("image")
    
    # Create LangChain tool wrapper with image
    langchain_tool = create_langchain_tool_from_basetool(
        self.image_zoom_tool, 
        instance_kwargs={"image": image}
    )
    
    # Run LangGraph workflow
    result = await super().run(sampling_params, **kwargs)
    
    # Cleanup tool instances
    await cleanup_tool_instances()
```

### 3. Tool Execution Flow

```
User Query → Agent (LLM) → Tool Call Decision → Tool Execution → Tool Response → Agent → Final Answer
```

## Key Differences from Original Implementation

### Dataset (`deepeyes.py`)

**Before:**
```python
tools_kwargs = {
    "image_zoom_in_tool": {
        "create_kwargs": {"image": images[0]},
    }
}
row_dict["tools_kwargs"] = tools_kwargs
row_dict["agent_name"] = "tool_agent"
```

**After:**
```python
# Image passed via multi_modal_data
# Agent loop handles tool initialization
row_dict["agent_name"] = "deepeyes_agent"
```

### Tool Integration

**ToolAgentLoop (Original):**
- Direct BaseTool usage
- Manual instance lifecycle management
- Requires tools_kwargs in dataset

**ReactAgentLoop (New):**
- LangChain StructuredTool wrapper
- Automatic instance lifecycle
- Image data from multi_modal_data

## Benefits

1. **Cleaner Interface**: No need to manage `tools_kwargs` in dataset
2. **Better Integration**: Uses standard LangChain/LangGraph patterns
3. **Flexibility**: Easy to add more tools or modify workflow
4. **Maintainability**: Follows established agent framework patterns

## Notes

- The agent loop automatically creates and releases tool instances
- Image data is extracted from `multi_modal_data` for each trajectory
- Tool responses are integrated back into the conversation flow
- Supports multi-turn conversations with multiple tool calls

## Troubleshooting

### Issue: Tool instance not found

**Cause**: Tool instance may not be properly initialized with image data.

**Solution**: Ensure `multi_modal_data` contains the image in the batch:
```python
row_dict["multi_modal_data"] = {"image": images}
```

### Issue: LangGraph workflow fails

**Cause**: Missing or invalid tool configuration.

**Solution**: Verify `agent_loop_config.json` is correctly formatted and accessible.

### Issue: Image not passed to tool

**Cause**: Image extraction may fail if format is unexpected.

**Solution**: Check image format in `multi_modal_data` - should be PIL Image or list of PIL Images.

## Future Enhancements

- [ ] Support for multiple images per trajectory
- [ ] Add more vision tools (e.g., object detection, segmentation)
- [ ] Implement tool response caching
- [ ] Add structured output support
- [ ] Enhance error handling and retry logic

