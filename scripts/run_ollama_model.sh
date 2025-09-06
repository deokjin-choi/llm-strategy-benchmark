#!/bin/bash
# run_ollama_model.sh
# Ollama 환경에서 모델 로딩 / 언로드 관리
# configs/models.yaml을 읽어 실제 모델명 매핑

CONFIG_FILE="$(dirname $0)/../configs/models.yaml"

if [ $# -lt 2 ]; then
  echo "Usage: $0 <action> <model-key>"
  echo "Actions: start | stop | pull"
  echo "Models: mistral | deepseek | yi | gemma"
  exit 1
fi

ACTION=$1
MODEL_KEY=$2

# --------------------------
# 모델 실제 이름 가져오기 (yq 필요)
# --------------------------
if ! command -v yq &> /dev/null; then
  echo "❌ yq is required (https://github.com/mikefarah/yq)"
  exit 1
fi

MODEL_NAME=$(yq e ".${MODEL_KEY}.name" $CONFIG_FILE)

if [ "$MODEL_NAME" = "null" ]; then
  echo "❌ Unknown model key: $MODEL_KEY"
  exit 1
fi

# --------------------------
# 동작 수행
# --------------------------
if [ "$ACTION" = "pull" ]; then
  echo "⬇️ Pulling Ollama model: $MODEL_NAME"
  ollama pull "$MODEL_NAME"

elif [ "$ACTION" = "start" ]; then
  echo "🚀 Starting Ollama model: $MODEL_NAME"
  ollama run "$MODEL_NAME" &

elif [ "$ACTION" = "stop" ]; then
  echo "🛑 Stopping Ollama model: $MODEL_NAME"
  ollama rm "$MODEL_NAME"

else
  echo "❌ Unknown action: $ACTION"
  exit 1
fi
