# DeepEyes Agent Loop 实现 - 变更摘要

## 任务完成情况

✅ 已完成：参考 `langgraph_agent` 格式，将 deepeyes 中的 tool 调用改成 agent loop 的形式

## 新增文件列表

### 1. 核心实现文件

#### `deepeyes_react_agent_loop.py` (236 行)
- **功能**: DeepEyes 的 React Agent Loop 主实现
- **关键组件**:
  - `DeepEyesReactAgentLoop` 类：继承自 `ReactAgentLoop`
  - `create_langchain_tool_from_basetool()`: BaseTool → LangChain 工具转换器
  - `ImageZoomInInput`: Pydantic 输入模型
  - 全局 `_tool_instances` 字典：管理工具实例
- **设计要点**:
  - 包装 `ImageZoomInTool` 为 LangChain `StructuredTool`
  - 自动管理工具实例生命周期
  - 从 `multi_modal_data` 提取图像初始化工具

### 2. 配置文件

#### `configs/agent_loop_config.json`
```json
[
  {
    "_target_": "recipe.deepeyes.deepeyes_react_agent_loop.DeepEyesReactAgentLoop",
    "name": "deepeyes_agent"
  }
]
```
- 定义 Agent 配置，用于训练时加载

### 3. 训练脚本

#### `run_deepeyes_agent_grpo.sh` (74 行)
- 基于原始 `run_deepeyes_grpo.sh` 修改
- **关键变更**:
  - 移除: `tool_config_path`
  - 添加: `actor_rollout_ref.rollout.multi_turn.format=hermes`
  - 添加: `actor_rollout_ref.rollout.agent.agent_loop_config_path`

### 4. 测试文件

#### `test_deepeyes_agent.py` (141 行)
- pytest 测试套件
- **测试内容**:
  - Agent loop 初始化
  - 工具调用执行
  - 多轮对话
  - Response mask 验证
- 使用 dummy images 进行测试

### 5. 文档文件

#### `README_AGENT.md` (250+ 行)
- **包含内容**:
  - 架构概览
  - 组件说明
  - 使用指南
  - 配置说明
  - 对比分析（ToolAgentLoop vs ReactAgentLoop）
  - 故障排除
  - 未来改进方向

#### `MIGRATION_GUIDE.md` (300+ 行)
- **包含内容**:
  - 详细的迁移步骤
  - 架构对比图
  - 配置参数说明
  - 常见问题解答
  - 技术细节说明
  - 实例生命周期图
  - 参考资料

#### `CHANGES_SUMMARY.md` (本文档)
- 变更摘要
- 文件清单
- 关键改动说明

### 6. 示例文件

#### `example_usage.py` (280 行)
- **包含示例**:
  1. 独立使用示例（async）
  2. 数据集集成示例
  3. 训练配置示例
  4. 与 langgraph_agent 对比

## 修改的现有文件

### `deepeyes.py`

**修改位置**: `CustomRLHFDataset.__getitem__()` 方法

**删除的代码** (第 168-177 行):
```python
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
```

**新增的代码** (第 167-172 行):
```python
# add index for each prompt
index = row_dict.get("extra_info", {}).get("index", 0)

# For agent loop mode: pass image via multi_modal_data
# The agent loop will handle tool initialization with the image
row_dict["index"] = index
row_dict["agent_name"] = "deepeyes_agent"  # Use the new DeepEyes agent loop
```

**变更原因**:
- 移除手动 `tools_kwargs` 配置
- 图像通过 `multi_modal_data` 传递（已存在）
- Agent loop 自动处理工具初始化
- 简化数据集代码

## 架构设计

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Training Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│  CustomRLHFDataset (deepeyes.py)                            │
│    └─> agent_name: "deepeyes_agent"                         │
│    └─> multi_modal_data: {"image": [PIL.Image]}            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  DeepEyesReactAgentLoop (deepeyes_react_agent_loop.py)     │
├─────────────────────────────────────────────────────────────┤
│  1. init_class():                                           │
│     - Create ImageZoomInTool (BaseTool)                     │
│     - Build LangGraph StateGraph                            │
│                                                              │
│  2. run():                                                   │
│     - Extract image from multi_modal_data                   │
│     - Wrap BaseTool as LangChain StructuredTool            │
│     - Execute LangGraph workflow                            │
│     - Cleanup tool instances                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   LangGraph Workflow                         │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐   tool call?   ┌──────────┐                 │
│  │  Agent   │───────YES──────▶│  Tools   │                 │
│  │  (LLM)   │◀────────────────│  (Exec)  │                 │
│  └──────────┘                  └──────────┘                 │
│      │                                                       │
│      NO (final answer)                                      │
│      ▼                                                       │
│    [END]                                                     │
└─────────────────────────────────────────────────────────────┘
```

### 工具包装层

```
┌─────────────────────────────────────────────────────────────┐
│  LangChain StructuredTool                                   │
│  (create_langchain_tool_from_basetool)                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  async def tool_func(**kwargs):                             │
│    1. Get/Create instance_id                                │
│    2. BaseTool.create(instance_id, image=...)              │
│    3. BaseTool.execute(instance_id, parameters)            │
│    4. Return text result                                    │
│                                                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  ImageZoomInTool (BaseTool)                                 │
│  - Stateful: maintains _instance_dict                       │
│  - create(): initialize with image                          │
│  - execute(): perform zoom operation                        │
│  - release(): cleanup                                        │
└─────────────────────────────────────────────────────────────┘
```

## 关键设计决策

### 1. 为什么使用 LangGraph?

- ✅ 与 `langgraph_agent` 保持一致的架构
- ✅ 标准的 Agent 框架模式
- ✅ 更好的可维护性和扩展性
- ✅ 支持复杂的多轮对话工作流

### 2. 为什么需要工具包装器?

- 🔹 `ImageZoomInTool` 是有状态的 `BaseTool`
- 🔹 LangChain 工具是无状态的函数
- 🔹 包装器桥接两种模式
- 🔹 保持向后兼容性（仍使用 BaseTool）

### 3. 实例生命周期管理

```python
# 全局字典存储实例
_tool_instances = {}

# 创建实例（按需）
if instance_id not in _tool_instances:
    created_id, _ = await base_tool.create(instance_id, **instance_kwargs)
    _tool_instances[instance_id] = created_id

# 执行工具
await base_tool.execute(instance_id, parameters)

# 清理（run() 结束时）
for instance_id in _tool_instances:
    await base_tool.release(instance_id)
    del _tool_instances[instance_id]
```

### 4. 图像数据流

```
Dataset (CustomRLHFDataset)
  └─> multi_modal_data: {"image": [PIL.Image]} ───┐
                                                    │
DeepEyesReactAgentLoop.run()                      │
  └─> Extract: image = kwargs["multi_modal_data"]["image"] ◀┘
       └─> Pass to: create_langchain_tool_from_basetool(
                      base_tool, 
                      instance_kwargs={"image": image}
                    )
            └─> Tool wrapper calls: 
                  base_tool.create(instance_id, image=image)
```

## 与 langgraph_agent 的对比

| 方面         | langgraph_agent        | deepeyes (新实现)      |
| ------------ | ---------------------- | ---------------------- |
| **基类**     | ReactAgentLoop         | ReactAgentLoop (继承)  |
| **工具类型** | LangChain @tool        | BaseTool (包装)        |
| **状态**     | 无状态                 | 有状态 (instance_id)   |
| **初始化**   | 无需特殊初始化         | 需要图像初始化         |
| **示例工具** | calculate()            | ImageZoomInTool        |
| **配置文件** | agent_loop_config.json | agent_loop_config.json |
| **工作流**   | LangGraph StateGraph   | LangGraph StateGraph   |

## 使用方式

### 快速开始

1. **测试 Agent Loop**:
   ```bash
   python recipe/deepeyes/test_deepeyes_agent.py
   ```

2. **查看示例**:
   ```bash
   python recipe/deepeyes/example_usage.py
   ```

3. **运行训练**:
   ```bash
   bash recipe/deepeyes/run_deepeyes_agent_grpo.sh
   ```

### 配置要点

```bash
# 关键参数
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.multi_turn.format=hermes
actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/deepeyes/configs/agent_loop_config.json
```

## 代码统计

| 文件                         | 行数      | 说明              |
| ---------------------------- | --------- | ----------------- |
| deepeyes_react_agent_loop.py | 236       | 主实现            |
| test_deepeyes_agent.py       | 141       | 测试代码          |
| example_usage.py             | 280       | 示例代码          |
| README_AGENT.md              | ~250      | 详细文档          |
| MIGRATION_GUIDE.md           | ~300      | 迁移指南          |
| run_deepeyes_agent_grpo.sh   | 74        | 训练脚本          |
| agent_loop_config.json       | 6         | 配置文件          |
| **总计**                     | **~1300** | **新增代码+文档** |

**修改现有文件**: deepeyes.py (-10 行，+5 行)

## 验证清单

- ✅ 代码实现完成
- ✅ 无 linting 错误
- ✅ 测试脚本就绪
- ✅ 示例代码完整
- ✅ 文档齐全
  - ✅ README_AGENT.md
  - ✅ MIGRATION_GUIDE.md
  - ✅ CHANGES_SUMMARY.md
- ✅ 配置文件创建
- ✅ 训练脚本更新

## 下一步建议

### 短期
1. 在小规模数据上测试 agent loop
2. 验证工具调用和响应的正确性
3. 对比与原始实现的性能

### 中期
1. 添加更多视觉工具（检测、分割等）
2. 优化工具响应截断策略
3. 增强错误处理和重试逻辑

### 长期
1. 支持多图像输入
2. 实现工具响应缓存
3. 添加结构化输出支持
4. 与其他 Agent 框架集成

## 参考链接

- **LangGraph 文档**: https://langchain-ai.github.io/langgraph/
- **LangChain Tools**: https://python.langchain.com/docs/modules/agents/tools/
- **verl 文档**: 项目内部文档

## 总结

本次实现成功将 deepeyes 从传统的 `ToolAgentLoop` 迁移到基于 LangGraph 的 `ReactAgentLoop`：

1. ✅ **架构一致**: 与 `langgraph_agent` 保持一致的设计模式
2. ✅ **简化配置**: 移除了数据集中的 `tools_kwargs` 手动配置
3. ✅ **标准化**: 使用 LangChain/LangGraph 的标准工具接口
4. ✅ **可扩展**: 易于添加新工具和修改工作流
5. ✅ **文档完善**: 提供详细的使用指南和迁移文档

所有代码、测试和文档已准备就绪，可以开始使用！🎉

