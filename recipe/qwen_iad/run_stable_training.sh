#!/bin/bash

# Stable training script for Qwen IAD with improved gradient stability
# This script uses the improved configuration to handle identical prompts

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/home/zxy/codes/working/RL/verl:$PYTHONPATH

# Training parameters
CONFIG_FILE="/home/zxy/codes/working/RL/verl/recipe/qwen_iad/qiad_multiturn_grpo_stable.yaml"
MODEL_PATH="/home/zxy/codes/working/RL/verl/recipe/qwen_iad/Qwen2-VL-2B-Instruct"
DATA_PATH="/home/zxy/codes/working/RL/verl/recipe/qwen_iad/data/thinklite_train.jsonl"
OUTPUT_DIR="/home/zxy/codes/working/RL/verl/recipe/qwen_iad/output_stable"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting stable training with improved gradient handling..."
echo "Config: $CONFIG_FILE"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"

# Run training with improved configuration
python -m verl.trainer.main_ppo \
    --config_file $CONFIG_FILE \
    --config_data.train_files $DATA_PATH \
    --config_data.val_files $DATA_PATH \
    --config_model.model.path $MODEL_PATH \
    --config_model.model.peft_config.r 64 \
    --config_model.model.peft_config.lora_alpha 16 \
    --config_model.model.peft_config.target_modules '["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]' \
    --config_model.model.peft_config.lora_dropout 0.05 \
    --config_trainer.default_hdfs_dir $OUTPUT_DIR \
    --config_trainer.project_name "qwen_iad_stable" \
    --config_trainer.experiment_name "stable_training_$(date +%Y%m%d_%H%M%S)" \
    --config_trainer.logger.tracking_project_name "qwen_iad_stable" \
    --config_trainer.save_freq 100 \
    --config_trainer.eval_freq 50 \
    --config_trainer.log_freq 10 \
    --config_trainer.gradient_checkpointing true \
    --config_trainer.mixed_precision "bf16" \
    --config_trainer.monitor_grad_norm true \
    --config_trainer.early_stopping_patience 5 \
    --config_trainer.early_stopping_threshold 0.01 \
    2>&1 | tee $OUTPUT_DIR/training.log

echo "Training completed. Check logs at: $OUTPUT_DIR/training.log"
echo "Model checkpoints saved to: $OUTPUT_DIR"

# Optional: Run evaluation after training
if [ -f "$OUTPUT_DIR/final_model" ]; then
    echo "Running post-training evaluation..."
    python -m verl.trainer.evaluate \
        --model_path $OUTPUT_DIR/final_model \
        --data_path $DATA_PATH \
        --output_path $OUTPUT_DIR/evaluation_results.json
fi