#!/bin/bash
# run_vllm_rocm.sh
# Start vLLM servers on ROCm (AMD MI300X 192GB)

HIP_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --port 3001 --dtype bfloat16 > mistral7b.log 2>&1 &

HIP_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/deepseek-7b-instruct \
  --port 3005 --dtype bfloat16 > deepseek7b.log 2>&1 &

HIP_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 3003 --dtype bfloat16 > llama8b.log 2>&1 &

HIP_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2-9b-it \
  --port 3004 --dtype bfloat16 > gemma9b.log 2>&1 &

HIP_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --port 3002 --dtype bfloat16 > qwen14b.log 2>&1 &

HIP_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model 01-ai/Yi-9B-Chat \
  --port 3006 --dtype bfloat16 > yi9b.log 2>&1 &
