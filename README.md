# XOLang

## Installation

### System Setup

```bash
sudo apt update
sudo apt install python3-venv -y
python3 -m venv xolangenv
source xolangenv/bin/activate
```

### Clone and Install

```bash
git clone https://github.com/xoul-ai/XOLang
cd XOLang
git checkout prod

pip install --upgrade pip
pip install -e "python[all]"
pip uninstall -y pynvml  # Remove warning
```

### Python 3.12 Additional Requirements

If you're using Python 3.12, install the development headers:

```bash
sudo apt install -y build-essential python3.12-dev
```

## Running the Server

```bash
source xolangenv/bin/activate
export TORCH_CUDA_ARCH_LIST="90"

nohup python3 -m sglang.launch_server \
  --model-path {model_name} \
  --tensor-parallel 8 \
  --host 0.0.0.0 \
  --port 8000 \
  --schedule-conservativeness 0.24 \
  --context-length 32768 \
  --mem-fraction-static 0.9 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend fa3 \
  --decode-attention-backend fa3 \
  --prefill-attention-backend fa3 \
  --disable-shared-experts-fusion \
  --enable-bigram-start-guard-the-word \
  --enable-metrics \
  > nohup.out &
```

## Optional: Performance Tuning

For optimal performance, you can tune the fused MoE kernels:

```bash
pip install "ray[default]"==2.34.0
cd /home/user/XOLang/benchmark/kernels/fused_moe_triton
python3 tuning_fused_moe_triton.py --model {model_name} --tp-size 8 --dtype fp8_w8a8 --tune
```

## SGLang Health Monitor and Auto-Restart Script

This script monitors the SGLang engine and automatically restarts it if it fails.

### Usage

**Run in foreground:**
```bash
./monitor.sh
```

**Run as daemon:**
```bash
./monitor.sh --daemon
```

### Script

Create a file named `monitor.sh` with the following content:

```bash
#!/bin/bash

# Configuration
MODEL_NAME="{model_name}"
SGLANG_HOST="0.0.0.0"
SGLANG_PORT="8000"
HEALTH_CHECK_URL="http://${SGLANG_HOST}:${SGLANG_PORT}/health"
SERVER_INFO_URL="http://${SGLANG_HOST}:${SGLANG_PORT}/get_server_info"
CHECK_INTERVAL=30
MAX_RETRIES=3      # Number of failed checks before restart
RETRY_COUNT=0
LOG_DIR="$HOME/sglang_logs"
MONITOR_LOG="$LOG_DIR/monitor.log"

mkdir -p "$LOG_DIR"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MONITOR_LOG"
}

check_health() {
    response=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$HEALTH_CHECK_URL" 2>/dev/null)

    if [ "$response" = "200" ]; then
        # Also verify the server can actually respond to info requests
        info_response=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$SERVER_INFO_URL" 2>/dev/null)
        if [ "$info_response" = "200" ]; then
            return 0
        else
            log_message "WARNING: Health check passed but server info failed (HTTP $info_response)"
            return 1
        fi
    else
        log_message "Health check failed (HTTP $response)"
        return 1
    fi
}

kill_sglang_processes() {
    log_message "Killing existing sglang processes..."

    pkill -9 -f sglang
    sleep 3

    port_pid=$(lsof -ti:$SGLANG_PORT 2>/dev/null)
    if [ -n "$port_pid" ]; then
        log_message "Killing process using port $SGLANG_PORT: $port_pid"
        kill -9 "$port_pid" 2>/dev/null
        sleep 2
    fi
}

start_sglang() {
    log_message "Starting sglang engine..."

    LOG_FILE="$LOG_DIR/sglang_$(date +%Y%m%d_%H%M%S).log"

    nohup python3 -m sglang.launch_server \
        --model-path $MODEL_NAME \
        --tensor-parallel 8 \
        --host $SGLANG_HOST \
        --port $SGLANG_PORT \
        --schedule-conservativeness 0.24 \
        --context-length 32768 \
        --mem-fraction-static 0.95 \
        --kv-cache-dtype fp8_e4m3 \
        --enable-metrics \
        > "$LOG_FILE" 2>&1 &

    local sglang_pid=$!
    disown $sglang_pid

    log_message "Started sglang with PID $sglang_pid"
    log_message "Logs: $LOG_FILE"

    log_message "Waiting for sglang to be ready (this may take several minutes for model loading)..."
    local wait_count=0
    local max_wait=600  # Wait up to 10 minutes for startup

    while [ $wait_count -lt $max_wait ]; do
        if check_health; then
            log_message "sglang is ready and healthy!"
            return 0
        fi

        # Check if process is still running
        if ! kill -0 $sglang_pid 2>/dev/null; then
            log_message "ERROR: sglang process died during startup"
            log_message "Check logs at: $LOG_FILE"
            tail -n 50 "$LOG_FILE" >> "$MONITOR_LOG"
            return 1
        fi

        sleep 5
        wait_count=$((wait_count + 5))

        if [ $((wait_count % 30)) -eq 0 ]; then
            log_message "Still waiting for sglang to start... ($wait_count seconds elapsed)"
        fi
    done

    log_message "ERROR: sglang failed to start within $max_wait seconds"
    return 1
}

restart_sglang() {
    log_message "Initiating sglang restart..."

    kill_sglang_processes

    if start_sglang; then
        log_message "sglang restarted successfully"
        RETRY_COUNT=0
        return 0
    else
        log_message "ERROR: Failed to restart sglang"
        return 1
    fi
}

main() {
    log_message "Starting sglang monitor (PID: $$)"
    log_message "Health check URL: $HEALTH_CHECK_URL"
    log_message "Check interval: ${CHECK_INTERVAL}s"
    log_message "Max retries before restart: $MAX_RETRIES"

    if check_health; then
        log_message "sglang is already running and healthy"
    else
        log_message "sglang is not running, starting it..."
        if ! start_sglang; then
            log_message "ERROR: Failed to start sglang on monitor startup"
            exit 1
        fi
    fi

    while true; do
        sleep $CHECK_INTERVAL

        if check_health; then
            if [ $RETRY_COUNT -gt 0 ]; then
                log_message "sglang recovered, resetting retry count"
                RETRY_COUNT=0
            fi
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            log_message "Health check failed ($RETRY_COUNT/$MAX_RETRIES)"

            if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
                log_message "Max retries reached, restarting sglang..."
                if ! restart_sglang; then
                    log_message "ERROR: Restart failed, will retry in next cycle"
                fi
            fi
        fi
    done
}

trap 'log_message "Monitor received shutdown signal"; exit 0' SIGTERM SIGINT

if [ "$1" = "--daemon" ]; then
    log_message "Starting monitor in daemon mode..."
    nohup "$0" > /dev/null 2>&1 &
    echo "Monitor started in background with PID: $!"
    echo "Logs: $MONITOR_LOG"
else
    main
fi
```

Make the script executable:

```bash
chmod +x monitor.sh
```

### Monitor Features

- **Health Monitoring**: Checks both `/health` and `/get_server_info` endpoints every 30 seconds
- **Auto-Restart**: Automatically restarts SGLang after 3 consecutive failed health checks
- **Logging**: All events logged to `~/sglang_logs/monitor.log`
- **Daemon Mode**: Can run in background with `--daemon` flag
- **Graceful Shutdown**: Handles SIGTERM and SIGINT signals properly

### Configuration

Edit the configuration variables at the top of the script:

- `MODEL_NAME`: Path to your model
- `SGLANG_PORT`: Server port (default: 8000)
- `CHECK_INTERVAL`: Seconds between health checks (default: 30)
- `MAX_RETRIES`: Failed checks before restart (default: 3)
