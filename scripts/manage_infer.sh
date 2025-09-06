#!/bin/bash
# scripts/manage_infer.sh
# Unified inference manager for home (Ollama), company (CUDA vLLM), amd_cloud (ROCm vLLM)

set -e

if [ $# -lt 2 ]; then
  echo "Usage: ENVIRONMENT=<home|company|amd_cloud> $0 <start|stop> <model_key>"
  echo "Example: ENVIRONMENT=home $0 start mistral"
  exit 1
fi

ACTION=$1       # start or stop
MODEL_KEY=$2    # mistral, qwen, gemma, etc.
ENVIRONMENT=${ENVIRONMENT:-home}   # default to home

MODELS_FILE="configs/models.yaml"

# -----------------------------
# Helper: ensure yq installed
# -----------------------------
if ! command -v yq &> /dev/null; then
  echo "❌ yq is required (https://github.com/mikefarah/yq)"
  exit 1
fi

echo "🌍 Environment: $ENVIRONMENT"
echo "⚙️  Action: $ACTION | Model: $MODEL_KEY"

# -----------------------------
# ENV = home (Ollama)
# -----------------------------
if [ "$ENVIRONMENT" = "home" ]; then
  if [ "$ACTION" = "start" ]; then
    echo "🏠 Running on Home (Ollama)"
    MODEL_ALIAS=$MODEL_KEY
    echo "🚀 Starting Ollama model: $MODEL_ALIAS"
    ollama pull $MODEL_ALIAS || true   # 모델이 이미 있으면 그냥 통과
    ollama run $MODEL_ALIAS
  elif [ "$ACTION" = "stop" ]; then
    echo "🛑 Stopping Ollama model: $MODEL_KEY"
    ollama stop $MODEL_KEY || true
  else
    echo "❌ Unknown action: $ACTION"
    exit 1
  fi

# -----------------------------
# ENV = company (CUDA vLLM)
# -----------------------------
elif [ "$ENVIRONMENT" = "company" ]; then
  MODEL_NAME=$(yq ".${MODEL_KEY}.name" $MODELS_FILE)
  if [ "$ACTION" = "start" ]; then
    echo "🏢 Running on Company (CUDA vLLM)"
    ./scripts/start_vllm_cuda.sh $MODEL_KEY "$MODEL_NAME"
  elif [ "$ACTION" = "stop" ]; then
    ./scripts/stop_vllm_cuda.sh $MODEL_KEY
  else
    echo "❌ Unknown action: $ACTION"
    exit 1
  fi

# -----------------------------
# ENV = amd_cloud (ROCm vLLM)
# -----------------------------
elif [ "$ENVIRONMENT" = "amd_cloud" ]; then
  MODEL_NAME=$(yq ".${MODEL_KEY}.name" $MODELS_FILE)
  if [ "$ACTION" = "start" ]; then
    echo "☁️  Running on AMD Cloud (ROCm vLLM)"
    ./scripts/start_vllm_rocm.sh $MODEL_KEY "$MODEL_NAME"
  elif [ "$ACTION" = "stop" ]; then
    ./scripts/stop_vllm_rocm.sh $MODEL_KEY
  else
    echo "❌ Unknown action: $ACTION"
    exit 1
  fi

else
  echo "❌ Unknown environment: $ENVIRONMENT"
  exit 1
fi
