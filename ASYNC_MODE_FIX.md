# vLLM Async Mode 乱码问题分析和解决方案

## 问题描述

- **Sync模式**: 使用`vLLMRollout`类，直接创建`LLM()`引擎。没有tool calling支持，但生成正常。
- **Async模式**: 使用`vLLMAsyncRollout` + `vLLMHttpServer` + `AsyncLLM`引擎。支持tool calling，但生成乱码。

## 根本原因

在async模式下使用`load_format='dummy'`时：

1. **架构差异**: 
   - vLLM v1的AsyncLLM引擎在`EngineCore`进程中运行模型
   - `vLLMAsyncRollout` workers只是ZeroMQ通信代理
   - 权重同步到workers后，**没有正确传递到EngineCore进程**

2. **权重同步失败**:
   - 配置中设置了`free_cache_engine=False`
   - 这导致`AgentLoopManager.generate_sequences()`不调用`wake_up()`
   - 即使调用了，`vLLMHttpServer.wake_up()`也只是唤醒workers，不会触发FSDP到vLLM的权重同步
   - 结果：AsyncLLM引擎一直使用dummy随机权重 → 产生乱码

3. **Vocab size修正缺失**:
   - `_monkey_patch_compute_logits`只在`vLLMAsyncRollout._load_model()`中应用
   - 但这个补丁应用到的是WorkerWrapperBase的模型，不是EngineCore中的模型
   - 即使权重正确，vocab size超出部分也可能产生无效token

## 解决方案

### 方案1：启用free_cache_engine（推荐用于HYBRID模式）

修改训练脚本:
```bash
actor_rollout_ref.rollout.free_cache_engine=True \
```

**优点**: 
- 确保每次生成前同步权重 (`wake_up()`会被调用)
- 内存管理更好（释放KV cache）

**缺点**:
- 可能稍慢（每次都要sync权重）

### 方案2：使用auto load_format（推荐用于async模式）

修改训练脚本，从训练好的checkpoint或预训练权重开始：
```bash
# 移除或注释掉这行（使用默认值'auto'）
# actor_rollout_ref.rollout.load_format=dummy
```

或显式设置:
```bash
actor_rollout_ref.rollout.load_format=auto \
```

**优点**: 
- 直接加载真实权重，不依赖同步
- 对async模式更可靠

**缺点**:
- 需要有预训练权重可用
- 首次加载稍慢

### 方案3：修复vLLMAsyncRollout.update_weights()（需要代码修改）

需要确保vLLMAsyncRollout.update_weights()能够将权重同步到vLLM v1的EngineCore进程。这是一个架构性问题，需要深入修改。

## 关于tool calling的支持

- **Sync模式**不支持tool calling是因为它使用`vLLMRollout.generate_sequences()`直接生成，没有agent loop
- **Async模式**支持tool calling是因为它使用`AgentLoopManager` → `AgentLoopWorker` → agent loop，在agent loop中可以调用tool

## 建议

对于当前的多轮tool calling场景：
1. 使用async模式（已配置）
2. 设置`free_cache_engine=True`以确保权重正确同步
3. 或者使用`load_format=auto`从实际权重开始

## 验证

修改后，检查日志中：
- 应该看到权重同步日志："After update_weights"等
- 生成的文本应该是有意义的，而不是乱码
- Tool calling应该正常工作

