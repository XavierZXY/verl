# GPU 内存分配不均衡问题修复说明

## 问题描述

在使用 SGLang async/hybrid 模式进行训练时，出现以下症状：
- **GPU 0** 负载突然涨到 30%，然后报 OOM 错误
- **GPU 1-7** 利用率只有 2%，几乎空闲
- 错误信息：`RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.`

## 根本原因

在 `verl/workers/rollout/sglang_rollout/async_sglang_server.py` 文件中，所有的 SGLang 服务器实例的 `base_gpu_id` 都被硬编码为 `0`：

```python
args = {
    ...
    "base_gpu_id": 0,  # ❌ 所有 8 个 replica 都使用 GPU 0！
    "gpu_id_step": 1,
    "tp_size": self.config.tensor_model_parallel_size,
    ...
}
```

当配置 `actor_rollout_ref.rollout.n=8` 时，会创建 8 个独立的 SGLang 服务器（replica 0-7）。由于 `base_gpu_id` 都是 0，所有服务器都尝试在 **GPU 0** 上分配内存：

- replica_rank=0 → GPU 0 ❌
- replica_rank=1 → GPU 0 ❌  
- replica_rank=2 → GPU 0 ❌
- ...
- replica_rank=7 → GPU 0 ❌

这导致：
1. GPU 0 上有 8 个服务器竞争内存
2. 大部分服务器因内存不足而启动失败
3. GPU 1-7 完全空闲

## 解决方案

将 `base_gpu_id` 改为使用 `self.replica_rank`，让每个 replica 使用不同的 GPU：

```python
args = {
    ...
    "base_gpu_id": self.replica_rank,  # ✅ 每个 replica 使用对应的 GPU
    "gpu_id_step": 1,
    "tp_size": self.config.tensor_model_parallel_size,
    ...
}
```

修复后的 GPU 分配：
- replica_rank=0 → GPU 0 ✅
- replica_rank=1 → GPU 1 ✅
- replica_rank=2 → GPU 2 ✅
- ...
- replica_rank=7 → GPU 7 ✅

## 修改的文件

**文件**: `verl/workers/rollout/sglang_rollout/async_sglang_server.py`  
**行号**: 131  
**修改**: `"base_gpu_id": 0,` → `"base_gpu_id": self.replica_rank,`

## 验证方法

修复后重新运行训练脚本，检查：

1. **所有 8 个 SGLang 服务器都能成功启动**
   ```
   SGLang http server: rollout_mode=<RolloutMode.HYBRID: 'hybrid'>, replica_rank=0, ...
   SGLang http server: rollout_mode=<RolloutMode.HYBRID: 'hybrid'>, replica_rank=1, ...
   ...
   SGLang http server: rollout_mode=<RolloutMode.HYBRID: 'hybrid'>, replica_rank=7, ...
   ```

2. **GPU 利用率均衡**
   - 使用 `nvidia-smi` 或 `rocm-smi` 监控
   - 每张 GPU 应该有大致相同的内存占用

3. **不再出现 OOM 错误**
   - 不再有 `RuntimeError: Not enough memory` 错误
   - 所有 replica 都能正常工作

## 适用场景

此修复适用于：
- ✅ SGLang async 模式 (`actor_rollout_ref.rollout.mode=async`)
- ✅ SGLang hybrid 模式 (训练和推理引擎融合)
- ✅ 多 replica 配置 (`actor_rollout_ref.rollout.n > 1`)
- ✅ 单节点多 GPU 环境

**不影响**：
- ❌ SGLang sync 模式 (该模式使用不同的代码路径，已正确处理)
- ❌ vLLM rollout (使用不同的实现)

## 注意事项

1. **确保配置匹配**：`rollout.n` 应该等于可用的 GPU 数量
   - 例如：8 张卡时，设置 `n=8`
   
2. **内存配置**：如果单个 replica 仍然内存不足，调整：
   ```bash
   actor_rollout_ref.rollout.gpu_memory_utilization=0.5  # 增加到 0.6 或 0.7
   ```

3. **Tensor Parallelism**：如果使用 TP，每个 replica 会使用多张卡：
   - `tp_size=2, n=4` → 需要 8 张 GPU
   - `base_gpu_id` 应该是 `replica_rank * tp_size`
   - 当前修复假设 `tp_size=1`

## 相关配置

```bash
# 推荐配置 (8 GPU, 8 replicas)
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.n=8
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.gpu_memory_utilization=0.4  # 可根据模型大小调整
```

## 参考

- Issue: GPU 0 过载，GPU 1-7 空闲
- 影响版本: 所有使用 SGLang async/hybrid 模式的版本
- 修复日期: 2025-10-09

