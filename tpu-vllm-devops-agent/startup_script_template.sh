#!/bin/bash
exec > /var/log/vllm-startup.log 2>&1
set -ex # Enable command tracing and exit on error

echo "Starting Queued vLLM Bootloader..."
echo "-----------------------------------"
echo "Project ID: {project_id}"
echo "Zone: {zone}"
echo "Model Name: {model_name}"
echo "HF_SECRET_ID: hf-token"
echo "-----------------------------------"

# Ensure internet connectivity
echo "Checking internet connectivity..."
set +e # Allow ping to fail without exiting immediately
for i in $(seq 1 30); do
  echo "Attempt $i/30: Pinging 8.8.8.8..."
  ping -c 1 8.8.8.8
  if [ $? -eq 0 ]; then
    echo "Internet connected."
    break
  fi
  echo "Ping failed, retrying in 5 seconds..."
  sleep 5
  if [ $i -eq 30 ]; then
    echo "ERROR: Internet connectivity failed after multiple retries. Exiting."
    exit 1
  fi
done
set -e # Re-enable exit on error

# Docker pull vLLM image
echo "Pulling vLLM Docker image: vllm/vllm-tpu:nightly"
set +e # Allow docker pull to fail without exiting immediately
for i in $(seq 1 5); do
  echo "Attempt $i/5: sudo docker pull vllm/vllm-tpu:nightly"
  sudo docker pull vllm/vllm-tpu:nightly
  if [ $? -eq 0 ]; then
    echo "Docker image pulled successfully."
    break
  fi
  echo "Docker pull failed, retrying in 20 seconds..."
  sleep 20
  if [ $i -eq 5 ]; then
    echo "ERROR: Failed to pull vLLM Docker image after multiple retries. Exiting."
    exit 1
  fi
done
set -e # Re-enable exit on error

# Set vLLM environment variables
echo "Setting vLLM environment variables..."
VLLM_MODEL="{model_name}"
VLLM_MAX_MODEL_LEN="16384"
VLLM_TP_SIZE="8"
VLLM_MAX_BATCHED_TOKENS="4096"
{limit_mm_per_prompt_env}
HF_HOME="/dev/shm"
HF_TOKEN="{hf_token}" # This will be sensitive, ensure it's quoted and not directly echoed for logs

echo "VLLM_MODEL set to: $VLLM_MODEL"
echo "VLLM_MAX_MODEL_LEN set to: $VLLM_MAX_MODEL_LEN"
echo "VLLM_TP_SIZE set to: $VLLM_TP_SIZE"
echo "VLLM_MAX_BATCHED_TOKENS set to: $VLLM_MAX_BATCHED_TOKENS"
if [ -n "{limit_mm_per_prompt_env}" ]; then
  echo "VLLM_LIMIT_MM_PER_PROMPT set." # Don't echo actual value for sensitive info
fi
echo "HF_HOME set to: $HF_HOME"
echo "HF_TOKEN set (value masked for security)."

# Main loop for vLLM container management
while true; do
  echo "--- Starting vLLM container management loop iteration ---"
  echo "Checking for existing vLLM container (name=vllm-gemma4)..."
  CONTAINER_ID=$(sudo docker ps -a --filter "name=vllm-gemma4" --format "{{.ID}}")
  CONTAINER_RUNNING=$(sudo docker ps --filter "name=vllm-gemma4" --format "{{.ID}}")
  echo "CONTAINER_ID (all states): $CONTAINER_ID"
  echo "CONTAINER_RUNNING (active only): $CONTAINER_RUNNING"

  if [ -z "$CONTAINER_RUNNING" ]; then # Container not running (or doesn't exist)
    echo "vLLM container not found or not running."
    if [ -n "$CONTAINER_ID" ]; then # Container exists but is stopped
      echo "vLLM container found (stopped). Removing existing container: $CONTAINER_ID"
      sudo docker rm vllm-gemma4 || echo "Error removing stopped container, continuing..."
    fi

    echo "Attempting to start new vLLM container..."
    # Log the full docker run command before executing it
    echo "Executing command: sudo docker run --name vllm-gemma4 --privileged --net=host -d 
      -v /dev/shm:/dev/shm --shm-size 16gb 
      -e HF_HOME="$HF_HOME" 
      -e HF_TOKEN="$HF_TOKEN" 
      vllm/vllm-tpu:nightly vllm serve "$VLLM_MODEL" 
      --max-model-len "$VLLM_MAX_MODEL_LEN" 
      --tensor-parallel-size "$VLLM_TP_SIZE" 
      --disable_chunked_mm_input 
      --max_num_batched_tokens "$VLLM_MAX_BATCHED_TOKENS" 
      {limit_mm_per_prompt_arg} 
      --enable-auto-tool-choice 
      --tool-call-parser gemma4 
      --reasoning-parser gemma4 
      --verbose"
    
    sudo docker run --name vllm-gemma4 --privileged --net=host -d -v /dev/shm:/dev/shm --shm-size 16gb -e HF_HOME="$HF_HOME" -e HF_TOKEN="$HF_TOKEN" vllm/vllm-tpu:nightly vllm serve "$VLLM_MODEL" --max-model-len "$VLLM_MAX_MODEL_LEN" --tensor-parallel-size "$VLLM_TP_SIZE" --disable_chunked_mm_input --max_num_batched_tokens "$VLLM_MAX_BATCHED_TOKENS" {limit_mm_per_prompt_arg} --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --verbose
    
    if [ $? -ne 0 ]; then
      echo "ERROR: Docker run command failed. Check parameters and image."
      sudo docker logs vllm-gemma4 || echo "Could not fetch logs for failed container."
      # Attempt a cleanup for next iteration
      sudo docker stop vllm-gemma4 && sudo docker rm vllm-gemma4 || true
      sleep 60 # Wait before next attempt
      continue # Skip health check and proceed to next loop iteration
    fi
    echo "Docker container start command issued. Waiting 5 seconds for initial setup..."
    sleep 5

    echo "Waiting for vLLM container to start and become healthy (up to 20 minutes)..."
    HEALTHY=0
    for i in $(seq 1 120); do
      HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X GET http://localhost:8000/health)
      if [ "$HEALTH_RESPONSE" -eq 200 ]; then
        echo "vLLM is running and healthy (HTTP 200)."
        HEALTHY=1
        break
      fi
      echo "vLLM not yet healthy (attempt $i/120). HTTP Status: $HEALTH_RESPONSE. Retrying in 10 seconds..."
      sleep 10
    done

    if [ "$HEALTHY" -eq 0 ]; then
      echo "ERROR: vLLM did not become healthy within the timeout."
      echo "Attempting to retrieve Docker logs for 'vllm-gemma4':"
      sudo docker logs vllm-gemma4 || echo "Could not retrieve Docker logs."
      echo "Attempting restart: stopping and removing container."
      sudo docker stop vllm-gemma4 && sudo docker rm vllm-gemma4 || echo "Error during forced restart cleanup, continuing..."
    fi
  else # Container is running
    echo "vLLM container is already running ($CONTAINER_RUNNING). Checking health..."
    HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X GET http://localhost:8000/health)
    if [ "$HEALTH_RESPONSE" -ne 200 ]; then
      echo "vLLM container is running but unhealthy or not responding (HTTP Status: $HEALTH_RESPONSE). Restarting fresh..."
      echo "Attempting to retrieve Docker logs for 'vllm-gemma4' before restart:"
      sudo docker logs vllm-gemma4 || echo "Could not retrieve Docker logs."
      sudo docker stop vllm-gemma4 && sudo docker rm vllm-gemma4 || echo "Error during forced restart cleanup, continuing..."
    else
      echo "vLLM is running and healthy (HTTP 200)."
    fi
  fi
  echo "--- End of vLLM container management loop iteration ---"
  sleep 60 # Check again every minute
done
