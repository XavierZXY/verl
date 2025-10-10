# AMD ROCm cumem 分配器修复

## 问题描述

在AMD ROCm平台上运行verl时，会遇到以下错误：

```
AssertionError: cumem allocator is not available
```

这个错误发生在vLLM尝试使用CUDA特定的`cumem`（CUDA Memory）分配器时。`cumem`是NVIDIA CUDA的内存管理功能，在AMD ROCm平台上不可用。

## 错误堆栈

```
File "/usr/local/lib/python3.12/dist-packages/vllm/device_allocator/cumem.py", line 140, in get_instance
    assert cumem_available, "cumem allocator is not available"
```

特别是在使用 `free_cache_engine=True` 时，vLLM的 `sleep()` 和 `wake_up()` 方法会调用cumem allocator。

## 解决方案

### 1. 禁用NCCL cumem分配器

通过设置环境变量 `NCCL_CUMEM_ENABLE=0` 来禁用NCCL的cumem分配器。这个环境变量告诉NCCL不要使用CUDA内存池功能，从而避免在ROCm平台上的兼容性问题。

### 2. 跳过vLLM的sleep/wake_up调用

在AMD ROCm平台上，自动检测并跳过vLLM引擎的 `sleep()` 和 `wake_up()` 调用，因为这些方法需要cumem allocator。

## 修改的文件

### 自动检测ROCm平台

1. **verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py**
   - 在 `vLLMRollout.__init__()` 方法中设置环境变量
   - 在 `vLLMAsyncRollout.__init__()` 方法中设置环境变量
   - 在 `vLLMRollout.resume()` 和 `release()` 中检测ROCm并跳过sleep/wake_up
   - 在 `vLLMAsyncRollout.resume()` 和 `release()` 中检测ROCm并跳过sleep/wake_up

2. **verl/workers/rollout/vllm_rollout/vllm_async_server.py**
   - 在 `vLLMHttpServer.__init__()` 方法中设置环境变量

### 环境变量设置

所有初始化方法中都添加了：
```python
# Disable NCCL cumem allocator to avoid compatibility issues with ROCm and other platforms
# This prevents "cumem allocator is not available" errors on AMD ROCm
os.environ["NCCL_CUMEM_ENABLE"] = "0"
```

### ROCm检测逻辑

在 `resume()` 和 `release()` 方法中添加了：
```python
# Skip sleep/wake_up on AMD ROCm as cumem allocator is not available
if torch.version.hip is not None:
    logger.warning("Skipping wake_up/sleep on AMD ROCm platform (cumem allocator not available)")
    return
```

这样，当检测到ROCm平台时（`torch.version.hip` 不为None），会自动跳过cumem相关的调用。

## 参考资料

这个修复方法已经在其他地方使用：
- `verl/workers/rollout/sglang_rollout/sglang_rollout.py` (line 94)
- `verl/workers/roles/reward_model_engine/sglang_reward_model.py` (line 50)
- `verl/trainer/constants_ppo.py` (line 31) - 在Ray runtime环境中设置

vLLM官方文档也提到了这个问题：
https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues

## 测试

修复后，您应该能够在AMD ROCm平台上正常运行verl训练任务，不会再遇到cumem分配器错误。

在启动训练时，您会看到警告信息（这是正常的）：
```
Skipping wake_up on AMD ROCm platform (cumem allocator not available)
Skipping sleep on AMD ROCm platform (cumem allocator not available)
```

## 注意事项

### 性能影响

- 这个设置会禁用NCCL的内存池功能，可能会略微影响性能
- 但这是在AMD ROCm平台上运行的必要设置
- 跳过 `sleep()`/`wake_up()` 意味着GPU内存不会在训练和推理之间自动释放/恢复
  - 在AMD ROCm平台上，您需要确保有足够的GPU内存来同时容纳训练和推理模型
  - 或者考虑使用 `free_cache_engine=False` 来完全禁用缓存管理

### 跨平台兼容性

- 这些修改会自动检测平台，对NVIDIA CUDA平台不会有任何影响
- 在NVIDIA平台上，正常的 `sleep()`/`wake_up()` 功能会继续工作
- 环境变量设置对两个平台都是安全的

### 配置建议

对于AMD ROCm平台，推荐配置：
```yaml
actor_rollout_ref.rollout.free_cache_engine=False  # 或者 True (会跳过sleep/wake_up)
actor_rollout_ref.rollout.gpu_memory_utilization=0.4  # 降低内存使用以避免OOM
```

