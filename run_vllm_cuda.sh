#!/bin/bash
# run_vllm_cuda.sh
# Start vLLM servers on CUDA (RTX 4090 × 4)

# GPU0: Mistral-7B + DeepSeek-7B
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --port 3001 --dtype float16 > mistral7b.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/deepseek-7b-instruct \
  --port 3005 --dtype float16 > deepseek7b.log 2>&1 &

# GPU1: Llama-3.1-8B
CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 3003 --dtype float16 > llama8b.log 2>&1 &

# GPU2: Gemma-2-9B
CUDA_VISIBLE_DEVICES=2 nohup python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2-9b-it \
  --port 3004 --dtype float16 > gemma9b.log 2>&1 &

# GPU3+GPU2: Qwen-14B (2GPU 병렬)
CUDA_VISIBLE_DEVICES=2,3 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --port 3002 --dtype float16 --tensor-parallel-size 2 > qwen14b.log 2>&1 &

# GPU3: Yi-9B (단독 실행)
CUDA_VISIBLE_DEVICES=3 nohup python -m vllm.entrypoints.openai.api_server \
  --model 01-ai/Yi-9B-Chat \
  --port 3006 --dtype float16 > yi9b.log 2>&1 &
