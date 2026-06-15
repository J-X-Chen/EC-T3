#!/usr/bin/env bash
# 查看请用
PORT=7682
SESSION="run2"

if [ $# -lt 1 ]; then
    export MUJOCO_GL=egl
    echo "Usage:"
    echo "  bash run.sh [start|stop|watch] <cmd> [args...]"
    exit 1
fi

mode="$1"

# 👉 判断模式
if [[ "$mode" == "start" || "$mode" == "stop" || "$mode" == "watch" ]]; then
    shift
else
    mode="start"
fi

# =========================
# STOP
# =========================
if [ "$mode" == "stop" ]; then
    echo "🛑 Stopping..."

    tmux kill-session -t "$SESSION" 2>/dev/null || true
    pkill -f "ttyd -p $PORT" 2>/dev/null || true

    echo "✅ Stopped"
    exit 0
fi

# =========================
# 构造命令
# =========================
if [ $# -lt 1 ]; then
    echo "❌ No command provided"
    exit 1
fi

cmd="$1"
shift

if [[ "$cmd" == *.py ]]; then
    full_cmd="MUJOCO_GL=egl python $cmd $@"
elif [[ "$cmd" == "python" ]]; then
    full_cmd="MUJOCO_GL=egl $cmd $@"
else
    full_cmd="MUJOCO_GL=egl $cmd $@"
fi

if [ "$mode" == "start" ]; then
    echo "🧹 Cleaning..."
    pkill -f "ttyd -p $PORT" 2>/dev/null || true
    tmux kill-session -t "$SESSION" 2>/dev/null || true
fi

echo "🚀 Running:"
echo "$full_cmd"
echo "--------------------------------"

# =========================
# WATCH 模式（dashboard）
# =========================
if [ "$mode" == "watch" ]; then

    # 👉 如果 run 不存在，报错
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "❌ No running session: $SESSION"
        echo "👉 请先用 start 启动"
        exit 1
    fi

    DASH="dash"

    tmux kill-session -t $DASH 2>/dev/null || true

    # 新建 dashboard session
    tmux new-session -d -s $DASH

    # =========================
    # 布局结构：
    # ┌───────────────────┬───┐
    # │  左上 (代码输出)   │GPU│  ← 上面两个占更大面积 (75%)
    # │                   │   │
    # ├───────────────────┼───┤
    # │  左下 (交互终端)  │CPU│  ← 下面两个占较小面积 (25%)
    # └───────────────────┴───┘
    # =========================

    # 步骤 1：先分成上下两行（上面占 75%，下面占 25%）
    # -p 指定新窗口的大小百分比，所以下面是 25%
    tmux split-window -v -t $DASH:0.0 -p 25
    # 现在：0.0 = 上半部分（75%），0.1 = 下半部分（25%）

    # 步骤 2：上半部分分成左右两列
    tmux select-pane -t $DASH:0.0
    tmux split-window -h -t $DASH:0.0
    # 现在：0.0 = 左上，0.1 = 右上，0.2 = 下半部分

    # 步骤 3：下半部分分成左右两列
    tmux select-pane -t $DASH:0.2
    tmux split-window -h -t $DASH:0.2
    # 现在：0.0 = 左上，0.1 = 右上，0.2 = 左下，0.3 = 右下

    # 步骤 4：填充内容
    # 左上：attach 到 run 的 output（使用 capture-pane 方式，不霸屏）
    tmux link-window -s $SESSION:0 -t $DASH:0
    # 右上：GPU 监控
    tmux send-keys -t $DASH:0.1 "
watch -n 1 nvidia-smi
" C-m

    # 左下：可交互终端
    tmux send-keys -t $DASH:0.2 "
echo '===== INTERACTIVE SHELL ====='
echo 'You can run commands here'
" C-m

    # 右下：CPU / 内存监控
    tmux send-keys -t $DASH:0.3 "
watch -n 2 free -h
" C-m

    # attach dashboard
    tmux attach -t $DASH

    exit 0
fi

# =========================
# START 模式（原本逻辑）
# =========================

# 创建日志目录
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"

# 启动 tmux（输出重定向到日志文件，同时保留在 pane 中）
tmux new-session -d -s "$SESSION" "bash -lc '
echo \"===== Starting: $full_cmd =====\"
echo \"Start time: \$(date)\"
echo \"Log file: $LOG_FILE\"
echo \"\"

# 同时输出到屏幕和日志文件
exec > >(tee -a $LOG_FILE) 2>&1

$full_cmd
EXIT_CODE=\$?

echo \"\"
echo \"===== Process finished with exit code: \$EXIT_CODE =====\"
echo \"End time: \$(date)\"
echo \"Log saved to: $LOG_FILE\"

if [ \$EXIT_CODE -ne 0 ]; then
    echo \"❌ Command failed! See log: $LOG_FILE\"
    echo \"Error log saved to: $LOG_FILE\" >> $LOG_FILE.err
else
    echo \"✅ Command completed successfully.\"
fi

echo \"\"
echo \"Shell will remain open. Press Ctrl+C to exit.\"
exec bash
'"

# 启动 ttyd（如果你环境允许）
nohup ttyd -p $PORT env -i HOME=$HOME TERM=xterm-256color tmux attach -t $SESSION > /dev/null 2>&1 &

sleep 1

echo ""
echo "=============================="
echo "浏览器打开："
echo "http://localhost:$PORT"
echo "日志文件：$LOG_FILE"
echo "（如果你环境支持 ttyd）"
echo "可以运行tmux attach -t run"
echo "=============================="