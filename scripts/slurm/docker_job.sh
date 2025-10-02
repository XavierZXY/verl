#!/bin/bash
#SBATCH --job-name=deepeyes_train      # 任务名称
#SBATCH --partition=Silo_Customer_Engineering  # 分区名称
#SBATCH --nodelist=tw008               # 可用节点，可替换为其他可用节点
#SBATCH --ntasks=1                     # 任务数量
#SBATCH --cpus-per-task=16             # CPU核心数，根据需求调整
#SBATCH --mem=200G                     # 内存需求，略高于shm-size
#SBATCH --time=8:00:00                 # 任务超时时间，根据训练时长调整
#SBATCH --output=deepeyes_%j.out       # 输出日志文件
#SBATCH --error=deepeyes_%j.err        # 错误日志文件

# 打印任务信息
echo "=========================================================="
echo "开始DeepEyes训练任务"
echo "任务ID (Job ID): $SLURM_JOB_ID"
echo "任务名称 (Job Name): $SLURM_JOB_NAME"
echo "执行节点 (Execution Node): $(hostname)"
echo "提交目录 (Submission Directory): $SLURM_SUBMIT_DIR"
echo "开始时间 (Start Time): $(date)"
echo "=========================================================="

# 设置容器名称
CONTAINER_NAME="vllm-deep"

# 检查容器是否正在运行
if [ ! "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "错误：容器 $CONTAINER_NAME 未在运行。"
    exit 1
fi

echo "在容器 $CONTAINER_NAME 中执行训练脚本..."

# 在指定的Docker容器中执行训练命令
# 使用 /bin/bash -c 将多个命令串联起来
# "&&" 确保只有前一个命令成功完成后，才会执行下一个命令
docker exec "$CONTAINER_NAME" /bin/bash -c "cd codes/verl/ && bash scripts/iad/train_iad.sh"

# 检查上一个命令的退出状态
if [ $? -eq 0 ]; then
    echo "训练脚本成功完成。"
else
    echo "错误：训练脚本执行失败。"
fi

echo "=========================================================="
echo "任务结束时间 (End Time): $(date)"
echo "DeepEyes训练任务完成。"
echo "=========================================================="