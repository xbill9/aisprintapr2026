#!/bin/bash
set -ex

echo "Testing vLLM Docker commands locally..."

# Docker pull vLLM image
echo "Pulling vLLM Docker image: vllm/vllm-tpu:nightly"
sudo docker pull vllm/vllm-tpu:nightly

# Set vLLM environment variables
echo "Setting vLLM environment variables..."
VLLM_MODEL="google/gemma-4-31B-it"
VLLM_MAX_MODEL_LEN="16384"
VLLM_TP_SIZE="8"
VLLM_MAX_BATCHED_TOKENS="4096"
HF_HOME="/dev/shm"
HF_TOKEN="hf_token_placeholder" # Placeholder

echo "Attempting to stop and remove any existing vllm-gemma4 container..."
sudo docker stop vllm-gemma4 || true
sudo docker rm vllm-gemma4 || true

echo "Attempting to start new vLLM container..."
sudo docker run --name vllm-gemma4 --privileged --net=host -d -v /dev/shm:/dev/shm --shm-size 16gb -e HF_HOME="$HF_HOME" -e HF_TOKEN="$HF_TOKEN" vllm/vllm-tpu:nightly vllm serve "$VLLM_MODEL" --max-model-len "$VLLM_MAX_MODEL_LEN" --tensor-parallel-size "$VLLM_TP_SIZE" --disable_chunked_mm_input --max_num_batched_tokens "$VLLM_MAX_BATCHED_TOKENS" --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --verbose

echo "Docker run command executed. Check 'sudo docker ps -a' and 'sudo docker logs vllm-gemma4' for status."
