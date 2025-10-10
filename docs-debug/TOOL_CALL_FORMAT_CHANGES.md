# Tool Call Format Reward 修改说明

## 修改概述

参考 `recipe/deepeyes/deepeyes.py` 的实现，为 `recipe/qwen_iad/qwen_iad.py` 增加了 tool call 相关的格式奖励检查。

## 主要修改内容

### 1. 更新 SYSTEM_PROMPT

在系统提示词中增加了关于 tool call 使用的说明：
- 说明可以使用工具来放大特定区域
- 明确 tool calls 应该放在 `<tool_call></tool_call>` 标签内
- 说明 tool responses 会在 `<tool_response></tool_response>` 标签内

### 2. 新增格式检查项

在 `compute_score()` 函数中增加了以下格式检查：

#### a. Tool Call 标签匹配检查
```python
count_tool_call_1 = solution_str.count("<tool_call>")
count_tool_call_2 = solution_str.count("</tool_call>")
if count_tool_call_1 != count_tool_call_2:
    format_errors_count += 1
```

#### b. Tool Response 标签匹配检查
```python
count_tool_response_1 = solution_str.count("<tool_response>")
count_tool_response_2 = solution_str.count("</tool_response>")
if count_tool_response_1 != count_tool_response_2:
    format_errors_count += 1
```

#### c. Tool Call JSON 格式验证
新增 `_validate_tool_call_json()` 辅助函数，验证 tool_call 标签内的内容是否为有效的 JSON：
```python
def _validate_tool_call_json(solution_str):
    """验证 tool_call 标签内容是否为有效 JSON"""
    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    tool_calls = re.findall(tool_call_pattern, solution_str, re.DOTALL)
    
    if not tool_calls:
        return True  # 没有 tool calls 是有效的
    
    for tool_call_content in tool_calls:
        try:
            json.loads(tool_call_content.strip())
        except (json.JSONDecodeError, ValueError):
            return False
    
    return True
```

### 3. 更新格式奖励计算

- 将 `total_checks` 从 5 增加到 8
- 新增的检查项：
  1. `<tool_call>` 标签匹配
  2. `<tool_response>` 标签匹配
  3. tool_call JSON 格式有效性

### 4. 增强日志记录

在调试输出中增加了 tool call 相关的状态信息：
```python
has_tool_calls = count_tool_call_1 > 0
tool_call_valid = _validate_tool_call_json(solution_str) if has_tool_calls else True
print(
    f"Score breakdown: format={format_reward:.2f}, acc={acc_reward:.2f}, "
    f"bbox={bbox_reward:.2f}, final={final_score:.2f}, "
    f"format_errors={format_errors_count}, has_tools={has_tool_calls}, tool_json_valid={tool_call_valid}"
)
```

### 5. 添加测试用例

添加了 4 个测试用例来验证功能：
1. **Test Case 1**: 完整格式的响应，包含有效的 tool calls
2. **Test Case 2**: 缺失 `</tool_call>` 闭合标签（格式错误）
3. **Test Case 3**: tool call 中包含无效的 JSON（格式错误）
4. **Test Case 4**: 不使用 tool calls 的有效响应

## 与 deepeyes.py 的区别

1. **不计算 tool_reward**: 按照要求，只检查格式，不对是否使用 tool 给予额外奖励
2. **更细粒度的检查**: 增加了 JSON 格式验证，deepeyes.py 只检查标签匹配
3. **平滑奖励函数**: 使用 `_smooth_format_reward()` 函数，根据错误数量给予平滑的惩罚

## 影响

- 模型现在会因为 tool call 格式错误而受到惩罚
- 鼓励模型在使用工具时遵循正确的格式
- format_errors_count 增加会降低 format_reward（最多 -0.5）

## 运行测试

可以通过以下命令运行测试用例：
```bash
cd /home/takisobe@amd.com/zxy/codes/verl
python recipe/qwen_iad/qwen_iad.py
```

注意：测试需要配置 `LLM_AS_A_JUDGE_BASE` 环境变量来调用 LLM judge 进行准确率评估。

