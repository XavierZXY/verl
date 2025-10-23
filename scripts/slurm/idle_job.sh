#!/bin/bash
#SBATCH --job-name=machine_idle        # 任务名称
#SBATCH --partition=Silo_Customer_Engineering         # 分区名称
#SBATCH --nodelist=tw029               # 指定节点
#SBATCH --ntasks=1                     # 任务数量
#SBATCH --cpus-per-task=64             # CPU核心数，根据需求调整
#SBATCH --gres=gpu:8                   # GPU数量
#SBATCH --mem=400G                     # 内存需求
#SBATCH --exclusive                    # 独占节点
#SBATCH --time=8:00:00                # 任务超时时间，设置为24小时
#SBATCH --output=log/idle_%j.out       # 输出日志文件
#SBATCH --error=log/idle_%j.err        # 错误日志文件

# 打印任务信息
echo "=========================================================="
echo "开始机器占用任务"
echo "任务ID (Job ID): $SLURM_JOB_ID"
echo "任务名称 (Job Name): $SLURM_JOB_NAME"
echo "执行节点 (Execution Node): $(hostname)"
echo "提交目录 (Submission Directory): $SLURM_SUBMIT_DIR"
echo "开始时间 (Start Time): $(date)"
echo "=========================================================="

sleep 36000

# 设置容器名称
CONTAINER_NAME="vllm-deep-idle"

# Function to create CPU load
create_cpu_load() {
    echo "创建CPU负载..."
    # 启动多个后台进程来占用CPU
    for i in $(seq 1 8); do
        yes > /dev/null &
    done
    echo "CPU负载进程已启动"
}

# Function to create GPU load
create_gpu_load() {
    echo "创建GPU负载..."
    docker exec "$CONTAINER_NAME" /bin/bash -c "
        python3 -c \"
import torch
import time
import threading

def gpu_stress_test(device_id):
    '''Create GPU load on specified device'''
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Starting GPU stress test on device {device_id}')
    
    # Create large tensors to occupy GPU memory
    try:
        # Allocate about 80% of GPU memory
        mem_size = int(torch.cuda.get_device_properties(device_id).total_memory * 0.8 // 4)  # float32 = 4 bytes
        tensor = torch.randn(mem_size, device=device)
        
        while True:
            # Perform matrix operations to keep GPU busy
            result = torch.matmul(tensor[:1000, :1000], tensor[:1000, :1000])
            time.sleep(0.1)  # Small delay to prevent overwhelming
    except Exception as e:
        print(f'GPU {device_id} stress test error: {e}')
        # Fallback to smaller operations
        while True:
            tensor = torch.randn(1000, 1000, device=device)
            result = torch.matmul(tensor, tensor)
            time.sleep(0.1)

# Start stress test on all available GPUs
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f'Found {gpu_count} GPUs, starting stress test...')
    
    threads = []
    for i in range(gpu_count):
        thread = threading.Thread(target=gpu_stress_test, args=(i,))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(60)
            print(f'GPU stress test running... {time.strftime(\\\"%Y-%m-%d %H:%M:%S\\\")}')
    except KeyboardInterrupt:
        print('GPU stress test interrupted')
else:
    print('No CUDA GPUs available')
    while True:
        time.sleep(60)
        print(f'Idle process running... {time.strftime(\\\"%Y-%m-%d %H:%M:%S\\\")}')
\"
    " &
    echo "GPU负载进程已启动"
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
        docker run -d --name=$CONTAINER_NAME \
            --volume /home/takisobe@amd.com/zxy:/home/takisobe@amd.com/zxy \
            --device /dev/dri:/dev/dri \
            --device /dev/kfd:/dev/kfd \
            --shm-size=400g \
            --cap-add SYS_PTRACE \
            --privileged \
            --security-opt seccomp=unconfined \
            --group-add video \
            -w /home/takisobe@amd.com/zxy \
            -t rocm/vllm:rocm6.4.1_vllm_0.10.0_20250812
        
        if [ $? -eq 0 ]; then
            echo "容器 $CONTAINER_NAME 创建成功。"
            
            # 等待容器完全启动
            sleep 10
            
            # 安装必要的Python包
            echo "在容器中安装必要的Python包..."
            docker exec "$CONTAINER_NAME" /bin/bash -c "pip install torch"
        else
            echo "错误：无法创建容器 $CONTAINER_NAME。"
            exit 1
        fi
    fi
else
    echo "容器 $CONTAINER_NAME 正在运行。"
fi

echo "开始创建系统负载以占用机器..."

# 创建CPU负载
create_cpu_load

# 等待一下让CPU负载稳定
sleep 5

# 创建GPU负载
create_gpu_load

echo "=========================================================="
echo "机器占用任务已启动"
echo "CPU和GPU负载进程正在运行"
echo "任务将持续运行直到时间限制或手动停止"
echo "=========================================================="

# 主循环 - 保持脚本运行并定期输出状态
COUNTER=0
while true; do
    sleep 300  # 每5分钟输出一次状态
    COUNTER=$((COUNTER + 1))
    ELAPSED_TIME=$((COUNTER * 5))
    
    echo "=========================================================="
    echo "机器占用状态报告 #$COUNTER"
    echo "已运行时间: ${ELAPSED_TIME} 分钟"
    echo "当前时间: $(date)"
    echo "节点信息: $(hostname)"
    echo "CPU使用情况:"
    top -bn1 | grep "Cpu(s)" | head -1
    echo "内存使用情况:"
    free -h | grep "Mem:"
    echo "GPU状态:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    elif command -v rocm-smi &> /dev/null; then
        rocm-smi --showuse --showmemuse
    else
        echo "GPU监控工具不可用"
    fi
    echo "容器状态:"
    docker ps --filter name=$CONTAINER_NAME --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo "=========================================================="
done

# 清理函数（当脚本被中断时调用）
cleanup() {
    echo "=========================================================="
    echo "收到中断信号，开始清理..."
    echo "停止CPU负载进程..."
    pkill -f "yes"
    echo "停止容器..."
    docker stop "$CONTAINER_NAME" 2>/dev/null
    echo "清理完成"
    echo "任务结束时间: $(date)"
    echo "=========================================================="
    exit 0
}

# 设置信号处理
trap cleanup SIGINT SIGTERM

# 等待信号
wait