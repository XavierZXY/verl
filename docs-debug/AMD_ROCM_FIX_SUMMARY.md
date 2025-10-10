# AMD ROCm平台修复总结

## 问题

在AMD ROCm平台上运行verl训练时遇到错误：
```
AssertionError: cumem allocator is not available
```

## 根本原因

vLLM v1在调用 `sleep()` 和 `wake_up()` 方法时会尝试使用CUDA Memory (cumem) 分配器，这是NVIDIA CUDA特有的功能，在AMD ROCm平台上不可用。

## 解决方案

### 1. 自动检测ROCm平台

修改了以下文件，添加ROCm平台检测逻辑：

**verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py**

在 `vLLMRollout` 和 `vLLMAsyncRollout` 类中：
- `resume()` 方法：检测ROCm平台，跳过 `wake_up()` 调用
- `release()` 方法：检测ROCm平台，跳过 `sleep()` 调用

检测逻辑：
```python
if torch.version.hip is not None:
    logger.warning("Skipping wake_up/sleep on AMD ROCm platform (cumem allocator not available)")
    return
```

### 2. 禁用NCCL cumem分配器

在以下类的 `__init__()` 方法中设置环境变量：
- `vLLMRollout`
- `vLLMAsyncRollout`  
- `vLLMHttpServer`

```python
os.environ["NCCL_CUMEM_ENABLE"] = "0"
```

## 测试

运行训练脚本：
```bash
bash scripts/iad/train_vllm.sh
```

您会看到警告信息（这是正常的）：
```
Skipping wake_up on AMD ROCm platform (cumem allocator not available)
Skipping sleep on AMD ROCm platform (cumem allocator not available)
```

## 配置

您的配置已经正确设置：
```yaml
actor_rollout_ref.rollout.free_cache_engine=True
actor_rollout_ref.rollout.gpu_memory_utilization=0.4
```

## 跨平台兼容性

✅ 这些修改会自动检测平台：
- **AMD ROCm**: 自动跳过 `sleep()`/`wake_up()`
- **NVIDIA CUDA**: 正常使用 `sleep()`/`wake_up()`

## 已知限制

在AMD ROCm平台上：
- GPU内存不会在训练和推理之间自动释放/恢复
- 需要确保有足够的GPU内存来同时容纳训练和推理模型
- 或者使用 `free_cache_engine=False` 来完全禁用缓存管理

## 修改文件列表

1. ✅ `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
   - 添加 `NCCL_CUMEM_ENABLE=0` 环境变量
   - 添加ROCm检测逻辑到 `resume()` 和 `release()` 方法

2. ✅ `verl/workers/rollout/vllm_rollout/vllm_async_server.py`
   - 添加 `NCCL_CUMEM_ENABLE=0` 环境变量

3. ✅ `scripts/iad/train_vllm.sh`
   - 设置 `free_cache_engine=True`

4. ✅ `AMD_ROCM_FIX.md`
   - 详细的修复文档

## 下一步

现在可以重新运行训练脚本，应该不会再遇到cumem错误了：

```bash
cd /home/takisobe@amd.com/zxy/codes/verl
bash scripts/iad/train_vllm.sh
```

