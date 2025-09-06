#!/bin/bash
# stop_vllm_rocm.sh
# Stop all vLLM servers on ROCm

echo "🛑 Stopping all vLLM ROCm servers..."

# vLLM 프로세스 찾아 종료
PIDS=$(ps aux | grep "vllm.entrypoints.openai.api_server" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
  echo "⚠️  No vLLM servers running."
else
  echo "Killing PIDs: $PIDS"
  kill $PIDS
  sleep 2
  # 혹시 남아있으면 강제 종료
  PIDS=$(ps aux | grep "vllm.entrypoints.openai.api_server" | grep -v grep | awk '{print $2}')
  if [ ! -z "$PIDS" ]; then
    echo "Force killing remaining PIDs: $PIDS"
    kill -9 $PIDS
  fi
fi

echo "✅ All vLLM ROCm servers stopped."
