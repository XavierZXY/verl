# DeepEyes Agent Loop Migration Guide

## 概述

本指南说明如何从原始的 `ToolAgentLoop` 实现迁移到新的 `ReactAgentLoop` 实现。新实现参考了 `langgraph_agent` 的架构，使用 LangGraph 的工作流系统。

## 变更摘要

### 新增文件

1. **deepeyes_react_agent_loop.py** - 主要的 Agent Loop 实现
   - `DeepEyesReactAgentLoop` 类：继承自 `ReactAgentLoop`
   - `create_langchain_tool_from_basetool()` 函数：工具包装器
   - `ImageZoomInInput` 模型：工具输入 schema

2. **configs/agent_loop_config.json** - Agent 配置文件
   ```json
   [
     {
       "_target_": "recipe.deepeyes.deepeyes_react_agent_loop.DeepEyesReactAgentLoop",
       "name": "deepeyes_agent"
     }
   ]
   ```

3. **run_deepeyes_agent_grpo.sh** - 使用 Agent Loop 的训练脚本
   - 替换 `tool_config_path` 为 `agent_loop_config_path`

4. **test_deepeyes_agent.py** - 测试脚本
   - 验证 Agent Loop 功能

5. **README_AGENT.md** - 详细文档
   - 架构说明
   - 使用指南
   - 故障排除

6. **example_usage.py** - 使用示例
   - 独立使用示例
   - 数据集集成示例
   - 与 langgraph_agent 的对比

7. **MIGRATION_GUIDE.md** - 本文档

### 修改文件

#### deepeyes.py

**修改位置**: `CustomRLHFDataset.__getitem__()` 方法（第 166-173 行）

**之前**:
```python
tools_kwargs = {
    "image_zoom_in_tool": {
        "create_kwargs": {"image": images[0]},
    }
}
row_dict["index"] = index
row_dict["tools_kwargs"] = tools_kwargs
row_dict["agent_name"] = "tool_agent"
```

**之后**:
```python
# For agent loop mode: pass image via multi_modal_data
# The agent loop will handle tool initialization with the image
row_dict["index"] = index
row_dict["agent_name"] = "deepeyes_agent"  # Use the new DeepEyes agent loop
```

**原因**: 
- 移除了 `tools_kwargs` 的手动配置
- Agent loop 将自动从 `multi_modal_data` 中提取图像
- 简化了数据集代码

## 架构对比

### 原始架构 (ToolAgentLoop)

```
Dataset 
  └─> tools_kwargs: {"image_zoom_in_tool": {"create_kwargs": {...}}}
       └─> ToolAgentLoop
            └─> BaseTool.create(create_kwargs)
            └─> BaseTool.execute(parameters)
            └─> BaseTool.release()
```

### 新架构 (ReactAgentLoop)

```
Dataset
  └─> multi_modal_data: {"image": [...]}
       └─> DeepEyesReactAgentLoop
            └─> Extract image from multi_modal_data
            └─> Create LangChain wrapper
                 └─> Wrap BaseTool as StructuredTool
                      └─> Instance lifecycle management
            └─> LangGraph StateGraph
                 └─> call_model (agent node)
                 └─> tools (tool execution node)
            └─> Automatic cleanup
```

### 关键差异

| 方面     | ToolAgentLoop           | ReactAgentLoop           |
| -------- | ----------------------- | ------------------------ |
| 工具接口 | 直接使用 BaseTool       | LangChain StructuredTool |
| 状态管理 | 手动 instance_id        | 自动包装管理             |
| 工作流   | 自定义状态机            | LangGraph StateGraph     |
| 配置     | tool_config_path        | agent_loop_config_path   |
| 初始化   | tools_kwargs in dataset | multi_modal_data         |

## 迁移步骤

### 步骤 1: 更新数据集

修改 `deepeyes.py` 中的 `CustomRLHFDataset`:

```python
# 删除
tools_kwargs = {...}
row_dict["tools_kwargs"] = tools_kwargs

# 改为
row_dict["agent_name"] = "deepeyes_agent"
```

### 步骤 2: 创建 Agent 配置

创建 `configs/agent_loop_config.json`:

```json
[
  {
    "_target_": "recipe.deepeyes.deepeyes_react_agent_loop.DeepEyesReactAgentLoop",
    "name": "deepeyes_agent"
  }
]
```

### 步骤 3: 更新训练脚本

修改训练脚本参数:

```bash
# 删除
actor_rollout_ref.rollout.multi_turn.tool_config_path=...

# 改为
actor_rollout_ref.rollout.multi_turn.format=hermes \
actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/deepeyes/configs/agent_loop_config.json
```

### 步骤 4: 测试

运行测试脚本验证:

```bash
python recipe/deepeyes/test_deepeyes_agent.py
```

或运行示例:

```bash
python recipe/deepeyes/example_usage.py
```

## 配置参数说明

### 必需参数

```bash
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5
actor_rollout_ref.rollout.multi_turn.max_user_turns=5
actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
actor_rollout_ref.rollout.multi_turn.format=hermes  # 工具解析格式
actor_rollout_ref.rollout.agent.agent_loop_config_path=...  # Agent 配置路径
```

### 工具配置

工具配置现在在 `DeepEyesReactAgentLoop.init_class()` 中硬编码:

```python
tool_config = {
    "num_workers": 20,
    "rate_limit": 50,
    "timeout": 30,
    "enable_global_rate_limit": True,
}
```

如需自定义，可以通过配置文件传递或修改代码。

## 常见问题

### Q1: 为什么要迁移到 ReactAgentLoop?

**答**: 
1. 更符合标准的 Agent 框架模式（LangChain/LangGraph）
2. 简化数据集代码，无需手动配置 tools_kwargs
3. 更好的可维护性和扩展性
4. 与 langgraph_agent 保持一致的架构

### Q2: 迁移后性能有变化吗?

**答**: 
- 功能上等价，性能应该相近
- 额外的包装层开销很小
- LangGraph 的状态管理可能略有优化

### Q3: 可以混用两种实现吗?

**答**: 
- 技术上可以，但不推荐
- 需要维护两套配置
- 建议统一使用新的 ReactAgentLoop

### Q4: 如何添加更多工具?

**答**: 
在 `DeepEyesReactAgentLoop.init_class()` 中添加:

```python
@classmethod
def init_class(cls, config, tokenizer, **kwargs):
    # 现有的 image_zoom_tool
    cls.image_zoom_tool = ImageZoomInTool(...)
    
    # 添加新工具
    cls.new_tool = NewTool(...)
    
    # 在 run() 中创建包装器
    cls.base_tools = [cls.image_zoom_tool, cls.new_tool]
```

### Q5: 原始的 tool_config_path 还能用吗?

**答**: 
- 如果继续使用 ToolAgentLoop，可以
- 但推荐迁移到 ReactAgentLoop 使用 agent_loop_config_path

## 兼容性说明

### 向后兼容

- 原始的 `run_deepeyes_grpo.sh` 仍然可用（使用 ToolAgentLoop）
- 新的 `run_deepeyes_agent_grpo.sh` 使用 ReactAgentLoop
- 两者可以共存

### 数据集

- CustomRLHFDataset 已更新为默认使用 ReactAgentLoop
- 如需使用 ToolAgentLoop，需要恢复 tools_kwargs 配置

## 技术细节

### 工具包装器实现

`create_langchain_tool_from_basetool()` 函数的工作原理:

1. 接收 BaseTool 实例和初始化参数
2. 创建异步工具函数
3. 在工具函数内部:
   - 管理 instance_id
   - 调用 BaseTool.create()
   - 调用 BaseTool.execute()
   - 处理错误
4. 返回 LangChain StructuredTool

### 实例生命周期

```
run() 开始
  ├─> 从 multi_modal_data 提取图像
  ├─> 创建工具包装器（传入图像）
  ├─> 执行 LangGraph 工作流
  │    ├─> 工具调用时：
  │    │    ├─> 自动创建 instance_id
  │    │    ├─> BaseTool.create(instance_id, image=...)
  │    │    ├─> BaseTool.execute(instance_id, params)
  │    │    └─> 返回结果
  │    └─> 工作流结束
  └─> 清理：BaseTool.release(instance_id)
run() 结束
```

### LangGraph 工作流

```python
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)  # LLM 调用
workflow.add_node("tools", ToolNode)    # 工具执行
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
```

## 参考资料

1. **langgraph_agent**: `/home/zxy/codes/working/RL/verl/recipe/langgraph_agent/`
   - `react_agent_loop.py` - ReactAgentLoop 基类
   - `chat_model.py` - 自定义 ChatModel

2. **相关文档**:
   - LangGraph: https://langchain-ai.github.io/langgraph/
   - LangChain Tools: https://python.langchain.com/docs/modules/agents/tools/

3. **verl 文档**:
   - BaseTool: `/home/zxy/codes/working/RL/verl/verl/tools/base_tool.py`
   - ToolAgentLoop: `/home/zxy/codes/working/RL/verl/verl/experimental/agent_loop/tool_agent_loop.py`

## 贡献

如有问题或改进建议，请:
1. 查看 README_AGENT.md
2. 运行 test_deepeyes_agent.py
3. 参考 example_usage.py

## 更新日志

- 2024-XX-XX: 初始版本，实现 DeepEyes ReactAgentLoop
- 添加了完整的测试和示例
- 创建了迁移指南和文档

