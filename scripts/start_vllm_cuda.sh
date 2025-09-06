#!/bin/bash
# start_vllm_cuda.sh
# Start all vLLM servers on CUDA (RTX 4090 × 4) using configs/models.yaml

CONFIG_FILE="$(dirname $0)/../configs/models.yaml"

# 모델별 GPU 및 포트 매핑
declare -A GPU_MAP=(
  [mistral]="0"
  [deepseek]="0"
  [llama]="1"
  [gemma]="2"
  [qwen]="2,3"
  [yi]="3"
)

declare -A PORT_MAP=(
  [mistral]="3001"
  [deepseek]="3005"
  [llama]="3003"
  [gemma]="3004"
  [qwen]="3002"
  [yi]="3006"
)

# 실행
for key in mistral deepseek llama gemma qwen yi; do
  MODEL_NAME=$(yq e ".${key}.name" $CONFIG_FILE)
  GPU=${GPU_MAP[$key]}
  PORT=${PORT_MAP[$key]}

  if [ "$MODEL_NAME" = "null" ]; then
    echo "⚠️  Skipping $key (not found in config)"
    continue
  fi

  LOGFILE="${key}.log"

  echo "🚀 Starting $MODEL_NAME on GPU $GPU (port=$PORT)"
  if [ "$key" = "qwen" ]; then
    CUDA_VISIBLE_DEVICES=$GPU nohup python -m vllm.entrypoints.openai.api_server \
      --model $MODEL_NAME \
      --port $PORT \
      --dtype float16 \
      --tensor-parallel-size 2 > $LOGFILE 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$GPU nohup python -m vllm.entrypoints.openai.api_server \
      --model $MODEL_NAME \
      --port $PORT \
      --dtype float16 > $LOGFILE 2>&1 &
  fi
done

echo "✅ All vLLM servers launched (CUDA mode)"
