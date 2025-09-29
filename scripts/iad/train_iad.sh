#!/bin/bash

set -x

export LLM_AS_A_JUDGE_BASE="http://10.21.239.180:9091/v1"
export WANDB_API_KEY="18fd5045bbb1e4c755167b1897f38a2c786250ed"

PROJECT_NAME="iad-grounding"
EXPERIMENT_NAME="debug_for_vllm"

BASEDIR=/home/takisobe@amd.com/zxy/codes/verl
SAVE_CHECKPOINT_DIR=/home/takisobe@amd.com/zxy/models/verl_checkpoints
# DATASET_TRAIN=${BASEDIR}/data/train.parquet
# DATASET_VAL=${BASEDIR}/data/val.parquet
DATASET_TRAIN=/home/takisobe@amd.com/zxy/data/deepeyes/output_first_300.parquet
DATASET_VAL=/home/takisobe@amd.com/zxy/data/deepeyes/output_first_300.parquet
REF_MODEL_PATH=/home/takisobe@amd.com/zxy/models/Qwen2.5-VL-3B-Instruct
# ---------------- Train config -----------------
WORLD_SIZE=1
BATCH_SIZE=128
PPO_BATCH_SIZE=128
MICRO_BATCH_SIZE=64
LR=1e-6
LOG_PER_GPU_BATCH_SIZE=2048
ROLLOUT_PARALLELISM=1
ROLLOUT_UTIL=0.6
N_GPUS_PER_NODE=8
N_ROLLOUT=16
# ---------------- Train config -----------------
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path=${BASEDIR}/recipe/deepeyes/configs \
    --config-name='deepeyes_multiturn_grpo' \
    data.train_files=${DATASET_TRAIN} \
    data.val_files=[${DATASET_VAL}] \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=8192 \
    data.max_response_length=16384 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PER_GPU_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_PARALLELISM} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.n=${N_ROLLOUT} \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_UTIL} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PER_GPU_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=recipe/deepeyes/configs/image_zoom_in_tool_config.yaml \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.rollout_data_dir=${BASEDIR}/logs/rollout_data/${EXPERIMENT_NAME} \
    trainer.total_epochs=1 2>&1 | tee ./logs/${EXPERIMENT_NAME}.log