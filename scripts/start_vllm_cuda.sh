#!/bin/bash
# start_vllm_cuda.sh
# Start vLLM servers on CUDA (RTX 4090 × 4) using configs/models.yaml

CONFIG_FILE="$(dirname $0)/../configs/models.yaml"

# 모델별 GPU 및 포트 매핑
declare -A GPU_MAP=(
  [mistral]="0"   # GPU0
  [deepseek]="1"  # GPU1
  [gemma]="2"     # GPU2
  [yi]="3"        # GPU3
)

declare -A PORT_MAP=(
  [mistral]="3000"   # GPU0 → port 3000
  [deepseek]="3001"  # GPU1 → port 3001
  [gemma]="3002"     # GPU2 → port 3002
  [yi]="3003"        # GPU3 → port 3003
)

# 실행
for key in mistral deepseek gemma yi; do
  MODEL_NAME=$(yq e ".${key}.name" $CONFIG_FILE)
  GPU=${GPU_MAP[$key]}
  PORT=${PORT_MAP[$key]}

  if [ "$MODEL_NAME" = "null" ]; then
    echo "⚠️  Skipping $key (not found in config)"
    continue
  fi

  LOGFILE="${key}.log"

  echo "🚀 Starting $MODEL_NAME on GPU $GPU (port=$PORT)"
  CUDA_VISIBLE_DEVICES=$GPU nohup python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --port $PORT \
    --dtype float16 > $LOGFILE 2>&1 &
done

echo "✅ Mistral, DeepSeek, Gemma, Yi launched (CUDA mode)"
