# 快速测试指南

## 测试修复后的async模式

### 1. 运行训练脚本

```bash
cd /home/takisobe@amd.com/zxy/codes/verl
bash scripts/iad/train_vllm.sh
```

### 2. 检查关键日志

#### ✅ 正确的日志应该包含：

**权重同步日志**:
```
Before state_dict() in sharding manager memory
After state_dict() in sharding manager memory
Before offload_fsdp_model_to_cpu
After offload_fsdp_model_to_cpu
After resume weights
After update_weights
```

**生成结果有意义**:
```
[Validation Step 0] Sample generations (Total: 10 samples):
  Sample 1: <实际有意义的回答，而不是"uate Exhibition"重复>
```

**Tool calling工作**:
```
tool_calls: {...}
tool_response: {...}
```

#### ❌ 如果仍然看到问题：

**乱码输出**:
```
uate Exhibitionuate Exhibitionuate Exhibition...
```
或其他重复的无意义文本

**缺少权重同步日志** - 检查是否有：
```
After update_weights
```

### 3. 验证配置

检查实际运行的配置（在日志开头）：

```python
'free_cache_engine': True,  # 应该是True
'load_format': 'dummy',     # HYBRID模式下可以是dummy
'mode': 'async',            # async模式
```

### 4. 故障排除

#### 问题：仍然看到乱码

**解决方案A**: 确认free_cache_engine已启用
```bash
grep "free_cache_engine" logs/vllm-debug.log
# 应该看到: 'free_cache_engine': True
```

**解决方案B**: 改用auto load_format

在`scripts/iad/train_vllm.sh`中添加：
```bash
actor_rollout_ref.rollout.load_format=auto \
```

并从checkpoint或预训练模型开始。

#### 问题：内存不足

如果`free_cache_engine=True`导致OOM：

1. 减少GPU memory utilization:
```bash
ROLLOUT_UTIL=0.3  # 从0.4降低
```

2. 或增加param_offload:
```bash
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
```

#### 问题：性能下降

如果weight sync导致速度变慢，可以考虑：
- 使用`load_format=auto`加载真实权重，避免每次sync
- 减少batch size
- 使用更多GPUs分担负载

### 5. 对比sync模式

如果想快速验证非tool calling的生成是否正常，可以临时切换到sync模式：

```bash
# 修改train_vllm.sh中的mode
actor_rollout_ref.rollout.mode=sync \  # 从async改为sync
```

但注意：sync模式**不支持tool calling**，只能用于验证基础生成功能。

## 预期结果

修复成功后：
- ✅ Validation生成有意义的文本
- ✅ Tool calling正常工作（调用image_zoom_in_tool等）
- ✅ 训练正常进行，loss下降
- ✅ 没有重复的乱码输出

## 需要帮助？

如果修复后仍有问题，请检查：
1. 完整的日志文件 `logs/vllm-debug.log`
2. 搜索"ERROR"或"Exception"
3. 检查GPU内存使用情况 `nvidia-smi` 或 `rocm-smi`

