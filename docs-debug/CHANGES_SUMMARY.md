# 修复vLLM Async模式乱码问题 - 更改总结

## 问题回顾

您遇到的问题：
- **Sync模式**: 没有tool calling逻辑，无法调用tool
- **Async模式**: 支持tool calling，但rollout结果出现乱码

## 根本原因

使用async模式时，配置中设置了：
```bash
actor_rollout_ref.rollout.load_format=dummy  # 使用dummy随机权重
actor_rollout_ref.rollout.free_cache_engine=False  # 禁用权重同步
```

这导致：
1. vLLM引擎使用dummy随机权重初始化
2. `free_cache_engine=False`导致`wake_up()`不被调用
3. 权重没有从FSDP训练模型同步到vLLM推理引擎
4. 结果：模型使用随机权重生成 → 输出乱码

## 修复内容

### 1. 修改训练脚本 (`scripts/iad/train_vllm.sh`)

**修改前**:
```bash
actor_rollout_ref.rollout.free_cache_engine=False \
```

**修改后**:
```bash
actor_rollout_ref.rollout.free_cache_engine=True \  # Must be True for async mode to enable weight sync from FSDP to vLLM
```

### 2. 增强错误处理 (`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`)

在`vLLMAsyncRollout._load_model()`中添加了错误处理和日志：
- 捕获monkey patch应用失败的异常
- 记录vocab_size信息用于调试
- 如果worker结构不匹配，记录警告而不是崩溃

### 3. 添加配置警告 (`verl/workers/rollout/vllm_rollout/vllm_async_server.py`)

在`vLLMHttpServer.run_server()`中添加警告：
- 检测`load_format='dummy'`与async HYBRID模式的组合
- 提示用户可能的乱码问题和解决方案
- 建议设置`free_cache_engine=True`或使用`load_format='auto'`

## 为什么这样修复

### free_cache_engine的作用

当`free_cache_engine=True`时：

1. **在AgentLoopManager中** (verl/experimental/agent_loop/agent_loop.py:849-850):
   ```python
   if self.config.actor_rollout_ref.rollout.free_cache_engine:
       self.wake_up()
   ```

2. **wake_up()调用链**:
   - `AgentLoopManager.wake_up()` 
   - → `RolloutReplica.wake_up()`
   - → `vLLMHttpServer.wake_up()`
   - → `worker.wake_up()` for each worker

3. **在ActorRolloutRefWorker.rollout_mode()中** (verl/workers/fsdp_workers.py:689-701):
   ```python
   if self.config.rollout.free_cache_engine:
       await self.rollout.resume(tags=["weights"])
   
   await self.rollout.update_weights(per_tensor_param, ...)  # 同步权重
   ```

这确保了每次生成前，FSDP训练好的权重都会同步到vLLM推理引擎。

## 关于Sync vs Async模式

| 特性 | Sync模式 | Async模式 |
|------|----------|-----------|
| 实现类 | `vLLMRollout` | `vLLMAsyncRollout` + `vLLMHttpServer` + `AgentLoopManager` |
| 引擎类型 | `LLM()` | `AsyncLLM()` + HTTP Server |
| Tool Calling | ❌ 不支持 | ✅ 支持 (通过AgentLoopWorker) |
| 生成方式 | 直接调用`generate()` | 通过HTTP API或token-in-token-out |
| 适用场景 | 简单单轮生成 | 多轮对话、tool calling、agent loop |

## 测试建议

运行修改后的脚本：
```bash
bash scripts/iad/train_vllm.sh
```

检查日志：
1. 应该看到"After update_weights"等权重同步日志
2. 验证时生成的文本应该有意义（不是"uate Exhibition"重复）
3. Tool calling应该正常工作

## 备选方案

如果`free_cache_engine=True`导致内存或性能问题，可以考虑：

```bash
actor_rollout_ref.rollout.load_format=auto \  # 直接加载真实权重，不依赖同步
actor_rollout_ref.rollout.free_cache_engine=False \  # 可以关闭cache释放
```

但这需要在启动时有可用的预训练权重。

## 参考

- vLLM Issue #13175: https://github.com/vllm-project/vllm/issues/13175 (vocab size问题)
- 详细分析: 参见 `ASYNC_MODE_FIX.md`

