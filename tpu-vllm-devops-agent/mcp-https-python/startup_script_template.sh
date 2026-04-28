#!/bin/bash

# Install Docker
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu 
    "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | 
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add current user to docker group
sudo usermod -aG docker $USER

# Pull and run vLLM container
docker pull vllm/vllm-tpu:nightly

# Start vLLM server
docker run --privileged --runtime=tpu --network=host 
    -e HUGGING_FACE_HUB_TOKEN={hf_token} 
    -v /usr/share/tpu:/usr/share/tpu 
    -p 8000:8000 
    vllm/vllm-tpu:nightly 
    python -m vllm.entrypoints.openai.api_server 
    --model {model_name} 
    --host 0.0.0.0 
    --port 8000 
    --tensor-parallel-size 8 
    --max-model-len 16384 
    --disable-chunked-mm-input 
    --max-num_batched_tokens 4096 
    --enable-auto-tool-choice 
    --tool-call-parser gemma4 
    --reasoning-parser gemma4
