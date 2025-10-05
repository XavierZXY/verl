#!/bin/bash
#SBATCH --job-name=deepeyes_train      # 任务名称
#SBATCH --partition=AIG_Models  # 分区名称
#SBATCH --nodelist=tw011
#SBATCH --ntasks=1                     # 任务数量
#SBATCH --cpus-per-task=64             # CPU核心数，根据需求调整
#SBATCH --gres=gpu:8
#SBATCH --mem=400G                     # 内存需求，略高于shm-size
#SBATCH --exclusive                   # 独占节点
#SBATCH --time=8:00:00                 # 任务超时时间，根据训练时长调整
#SBATCH --output=log/deepeyes_%j.out       # 输出日志文件
#SBATCH --error=log/deepeyes_%j.err        # 错误日志文件

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

# Function to install required packages in container
install_packages() {
    echo "在容器中安装必要的Python包..."
    docker exec "$CONTAINER_NAME" /bin/bash -c "pip install qwen_vl_utils torchdata tensordict hydra-core math_verify codetiming wandb swanlab"
    if [ $? -eq 0 ]; then
        echo "Python包安装成功。"
    else
        echo "警告：Python包安装失败，但继续执行训练脚本。"
    fi
}

# 检查容器是否正在运行
if [ ! "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "检测到容器 $CONTAINER_NAME 未在运行。"
    
    # 检查容器是否存在（但未运行）
    if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
        echo "尝试启动现有容器 $CONTAINER_NAME..."
        docker start "$CONTAINER_NAME"
        if [ $? -eq 0 ]; then
            echo "容器 $CONTAINER_NAME 启动成功。"
        else
            echo "错误：无法启动容器 $CONTAINER_NAME。"
            exit 1
        fi
    else
        echo "容器 $CONTAINER_NAME 不存在，正在创建新容器..."
        
        # 创建新容器
        docker run -d --name=vllm-deep \
            --volume /home/takisobe@amd.com/zxy:/home/takisobe@amd.com/zxy \
            --device /dev/dri:/dev/dri \
            --device /dev/kfd:/dev/kfd \
            --shm-size=400g \
            --cap-add SYS_PTRACE \
            --privileged \
            --security-opt seccomp=unconfined \
            --group-add video \
            -w /home/takisobe@amd.com/zxy \
            -p 9091:9091 \
            -p 9092:9092 \
            -t rocm/vllm:rocm6.4.1_vllm_0.10.0_20250812
        
        if [ $? -eq 0 ]; then
            echo "容器 $CONTAINER_NAME 创建成功。"
            
            # 等待容器完全启动
            sleep 5
            
            # 安装必要的Python包
            install_packages
        else
            echo "错误：无法创建容器 $CONTAINER_NAME。"
            exit 1
        fi
    fi
else
    echo "容器 $CONTAINER_NAME 正在运行。"
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