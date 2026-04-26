# Deployment Guide: vLLM on TPUs (Gemma 4)

This document summarizes the deployment state and configuration for the vLLM inference server running on Google Cloud TPUs.

## 📦 Model Artifacts
The model used is **Gemma 4 (31B-it)**, served directly from Hugging Face.

*   **Model ID:** `google/gemma-4-31B-it`
*   **Format:** Hugging Face Transformers
*   **Precision:** bfloat16 (Native TPU support)

## 🚀 Inference Stack (vLLM on TPU)
The inference server is deployed on **Cloud TPU v6e (Trillium)** using the `vllm-tpu` specialized container.

*   **Hardware:** 
    *   **TPU Version:** v6e (Trillium)
    *   **Topology:** `2x4` (8 chips)
*   **Software:**
    *   **Image:** `vllm/vllm-tpu:gemma4`
    *   **Max Model Length:** `16384`
    *   **Tensor Parallel Size:** `8` (1:1 mapping to chips)

## 🛠 Usage
To connect the MCP Agent to the TPU service, export the following environment variables:

```bash
export VLLM_BASE_URL="http://<TPU_VM_IP>:8000"
export MODEL_NAME="google/gemma-4-31B-it"
export GOOGLE_CLOUD_PROJECT="aisprint-491218"
```

Then run the agent:
```bash
make run
```

## 📜 Deployment Commands

### 1. Create TPU v6e Instance
```bash
gcloud alpha compute tpus tpu-vm create vllm-gemma4-tpu \
    --type v6e --topology 2x4 \
    --project $PROJECT_ID --zone $ZONE --version v2-alpha-tpuv6e
```

### 2. Launch vLLM Container (on TPU VM)
```bash
sudo docker run -t --rm --name vllm-gemma4 --privileged --net=host \
    -v /dev/shm:/dev/shm --shm-size 10gb \
    -e HF_TOKEN=$HF_TOKEN \
    vllm/vllm-tpu:gemma4 \
    vllm serve google/gemma-4-31B-it \
    --max-model-len 16384 \
    --tensor-parallel-size 8 \
    --disable_chunked_mm_input \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4
```

### 3. Verification
```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "google/gemma-4-31B-it",
        "messages": [{"role": "user", "content": "Hello Gemma 4!"}]
    }'
```
