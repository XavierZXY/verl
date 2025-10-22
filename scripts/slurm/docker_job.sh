#!/bin/bash
#SBATCH --job-name=Mvtec-64    # 任务名称
#SBATCH --partition=Silo_Customer_Engineering  # 分区名称
#SBATCH --ntasks=1                     # 任务数量
#SBATCH --cpus-per-task=64             # CPU核心数，根据需求调整
#SBATCH --gres=gpu:8
#SBATCH --mem=400G                     # 内存需求，略高于shm-size
#SBATCH --exclusive                   # 独占节点
#SBATCH --time=8:00:00                 # 任务超时时间，根据训练时长调整
#SBATCH --output=log/deepeyes_%j.out       # 输出日志文件
#SBATCH --error=log/deepeyes_%j.err        # 错误日志文件
#--nodelist=tw051
# Silo_Customer_Engineering,AIG_Models
# 设置容器名称
CONTAINER_NAME="vllm-deep"

# 监控间隔（秒）
MONITOR_INTERVAL=300  # 5分钟 = 300秒

# 存储监控进程的PID
MONITOR_PID=""

# Docker监控函数 - 每5分钟检查并停止其他docker容器
monitor_docker_containers() {
    echo "=========================================================="
    echo "启动Docker容器监控服务"
    echo "监控间隔: ${MONITOR_INTERVAL}秒 (5分钟)"
    echo "当前容器: $CONTAINER_NAME"
    echo "=========================================================="
    
    while true; do
        # 获取所有正在运行的容器，排除当前容器
        other_containers=$(docker ps --format "{{.Names}}" | grep -v "^${CONTAINER_NAME}$")
        
        if [ -n "$other_containers" ]; then
            echo "=========================================================="
            echo "[$(date)] 检测到其他正在运行的Docker容器："
            echo "$other_containers"
            echo "正在停止这些容器..."
            
            # 停止每个其他容器
            echo "$other_containers" | while read -r container; do
                if [ -n "$container" ]; then
                    echo "  - 停止容器: $container"
                    docker stop "$container" 2>&1
                    if [ $? -eq 0 ]; then
                        echo "    ✓ 容器 $container 已成功停止"
                    else
                        echo "    ✗ 停止容器 $container 时出现错误"
                    fi
                fi
            done
            echo "=========================================================="
        else
            echo "[$(date)] Docker容器监控检查: 没有检测到其他运行的容器 ✓"
        fi
        
        # 等待指定的时间间隔
        sleep $MONITOR_INTERVAL
    done
}

# Cleanup function to stop and remove container
cleanup() {
    echo "=========================================================="
    echo "执行清理操作..."
    echo "清理时间 (Cleanup Time): $(date)"
    
    # 停止监控进程
    if [ -n "$MONITOR_PID" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
        echo "正在停止Docker监控进程 (PID: $MONITOR_PID)..."
        kill "$MONITOR_PID" 2>/dev/null
        wait "$MONITOR_PID" 2>/dev/null
        echo "Docker监控进程已停止。"
    fi
    
    # Check if container is running and stop it
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        echo "正在停止容器 $CONTAINER_NAME..."
        docker stop "$CONTAINER_NAME"
        if [ $? -eq 0 ]; then
            echo "容器 $CONTAINER_NAME 已成功停止。"
        else
            echo "警告：停止容器 $CONTAINER_NAME 时出现错误。"
        fi
    else
        echo "容器 $CONTAINER_NAME 未在运行。"
    fi
    
    # Optionally remove the container (uncomment if you want to remove it)
    # if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    #     echo "正在删除容器 $CONTAINER_NAME..."
    #     docker rm "$CONTAINER_NAME"
    #     if [ $? -eq 0 ]; then
    #         echo "容器 $CONTAINER_NAME 已成功删除。"
    #     else
    #         echo "警告：删除容器 $CONTAINER_NAME 时出现错误。"
    #     fi
    # fi
    
    echo "清理操作完成。"
    echo "=========================================================="
}

# Signal handler function
handle_signal() {
    echo ""
    echo "=========================================================="
    echo "收到终止信号，正在清理资源..."
    echo "信号接收时间 (Signal Received Time): $(date)"
    cleanup
    exit 1
}

# Set up signal traps for SIGTERM and SIGINT
trap handle_signal SIGTERM SIGINT

# Also ensure cleanup runs on normal script exit
trap cleanup EXIT

# 打印任务信息
echo "=========================================================="
echo "开始DeepEyes训练任务"
echo "任务ID (Job ID): $SLURM_JOB_ID"
echo "任务名称 (Job Name): $SLURM_JOB_NAME"
echo "执行节点 (Execution Node): $(hostname)"
echo "提交目录 (Submission Directory): $SLURM_SUBMIT_DIR"
echo "开始时间 (Start Time): $(date)"
echo "=========================================================="

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
            echo "错误：无法启动容器 $CONTAINER_NAME。正在删除并重新创建..."
            docker rm -f "$CONTAINER_NAME"
            
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
                echo "容器 $CONTAINER_NAME 重新创建成功。"
                
                # 等待容器完全启动
                sleep 5
                
                # 安装必要的Python包
                install_packages
            else
                echo "错误：无法重新创建容器 $CONTAINER_NAME。"
                exit 1
            fi
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
    # 重启容器
    echo "正在重启容器 $CONTAINER_NAME..."
    docker restart "$CONTAINER_NAME"
fi

# 启动Docker容器监控进程（后台运行）
echo "=========================================================="
echo "启动后台监控进程..."
monitor_docker_containers &
MONITOR_PID=$!
echo "监控进程已启动 (PID: $MONITOR_PID)"
echo "=========================================================="

echo "在容器 $CONTAINER_NAME 中执行训练脚本..."

# 在指定的Docker容器中执行训练命令
# 使用 /bin/bash -c 将多个命令串联起来
# "&&" 确保只有前一个命令成功完成后，才会执行下一个命令
docker exec "$CONTAINER_NAME" /bin/bash -c "cd codes/verl-compare/ && bash scripts/iad/train_iad.sh"
# docker exec "$CONTAINER_NAME" /bin/bash -c "cd codes/verl/ && bash scripts/iad/train_deepeyes.sh"

# 检查上一个命令的退出状态
TRAINING_EXIT_CODE=$?
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "训练脚本成功完成。"
    SCRIPT_SUCCESS=true
else
    echo "错误：训练脚本执行失败，退出码: $TRAINING_EXIT_CODE"
    SCRIPT_SUCCESS=false
fi

echo "=========================================================="
echo "任务结束时间 (End Time): $(date)"
if [ "$SCRIPT_SUCCESS" = true ]; then
    echo "DeepEyes训练任务成功完成。"
else
    echo "DeepEyes训练任务执行失败。"
fi
echo "=========================================================="

# 等待6小时后再退出
WAIT_HOURS=6
WAIT_SECONDS=$((WAIT_HOURS * 3600))
echo ""
echo "=========================================================="
echo "任务已完成，将在 ${WAIT_HOURS} 小时后退出"
echo "等待开始时间: $(date)"
echo "预计退出时间: $(date -d "+${WAIT_HOURS} hours" 2>/dev/null || date -v+${WAIT_HOURS}H 2>/dev/null || echo "N/A")"
echo "在等待期间，Docker监控进程将继续运行..."
echo "如需提前退出，可以手动取消任务 (scancel $SLURM_JOB_ID)"
echo "=========================================================="

# 等待指定时间
sleep $WAIT_SECONDS

echo ""
echo "=========================================================="
echo "等待时间已到，准备退出..."
echo "实际退出时间: $(date)"
echo "=========================================================="

# Exit with the same code as the training script
exit $TRAINING_EXIT_CODE