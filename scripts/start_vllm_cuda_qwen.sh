#!/bin/bash
# start_qwen_cuda.sh
# Start Qwen 14B (tensor parallel) on GPU2+3, port 3004

CONFIG_FILE="$(dirname $0)/../configs/models.yaml"

MODEL_NAME=$(yq e ".qwen.name" $CONFIG_FILE)
GPU="2,3"
PORT="3004"

if [ "$MODEL_NAME" = "null" ]; then
  echo "⚠️  Qwen not configured in models.yaml"
  exit 1
fi

LOGFILE="qwen.log"

echo "🚀 Starting $MODEL_NAME on GPU $GPU (port=$PORT)"
CUDA_VISIBLE_DEVICES=$GPU nohup python -m vllm.entrypoints.openai.api_server \
  --model $MODEL_NAME \
  --port $PORT \
  --dtype float16 \
  --tensor-parallel-size 2 > $LOGFILE 2>&1 &

echo "✅ Qwen launched on GPU2+3 (port=$PORT)"
