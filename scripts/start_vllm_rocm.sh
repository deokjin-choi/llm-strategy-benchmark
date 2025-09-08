#!/bin/bash
# start_vllm_rocm.sh
# Start all vLLM servers on ROCm (MI300X) using configs/models.yaml

CONFIG_FILE="$(dirname $0)/../configs/models.yaml"

# 모델별 포트 매핑
declare -A PORT_MAP=(
  [mistral]="3000"
  [deepseek]="3001"
  [gemma]="3002"
  [yi]="3003"
  [qwen]="3004"
  [llama]="3005"  
)

# 실행
for key in mistral deepseek llama gemma qwen yi; do
  MODEL_NAME=$(yq e ".${key}.name" $CONFIG_FILE)
  PORT=${PORT_MAP[$key]}

  if [ "$MODEL_NAME" = "null" ]; then
    echo "⚠️  Skipping $key (not found in config)"
    continue
  fi

  LOGFILE="${key}_rocm.log"

  echo "🚀 Starting $MODEL_NAME on HIP_VISIBLE_DEVICES=0 (port=$PORT)"
  HIP_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --port $PORT \
    --dtype bfloat16 > $LOGFILE 2>&1 &
done

echo "✅ All vLLM servers launched (ROCm mode)"
