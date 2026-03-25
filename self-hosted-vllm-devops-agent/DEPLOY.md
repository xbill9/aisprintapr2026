# Deployment Guide: Self-Hosted vLLM on Cloud Run (Gemma 2b-it)

This document summarizes the deployment state and configuration for the vLLM inference server used by the DevOps Agent.

## 📦 Model Artifacts
The model was extracted from `gemma-transformers-2b-it-v3.tar.gz` and uploaded to Google Cloud Storage.

*   **Bucket:** `gs://aisprint-491218-bucket/`
*   **Path:** `gemma-2b-it/`
*   **Format:** Hugging Face Transformers (Safetensors)

## 🚀 Inference Stack (vLLM)
The inference server is deployed on Cloud Run with GPU acceleration and GCS FUSE.

*   **Service Name:** `vllm-gemma-2b-it`
*   **Service URL:** `https://vllm-gemma-2b-it-289270257791.us-east4.run.app`
*   **Region:** `us-east4`
*   **Hardware:** 
    *   **GPU:** 1x NVIDIA L4
    *   **vCPU:** 4
    *   **Memory:** 16GiB
*   **Configuration:**
    *   **Container Port:** `8080`
    *   **Max Model Length:** `4096`
    *   **Storage:** GCS FUSE mounted at `/mnt/models`
    *   **Zonal Redundancy:** Disabled (`--no-gpu-zonal-redundancy`)

## 🛠 Usage
To connect the MCP Agent to this service, export the following environment variables:

```bash
export VLLM_BASE_URL="https://vllm-gemma-2b-it-289270257791.us-east4.run.app"
export MODEL_NAME="/mnt/models/gemma-2b-it"
export GOOGLE_CLOUD_PROJECT="aisprint-491218"
```

Then run the agent:
```bash
make run
```

## 📜 Deployment Command (Reference)
```bash
gcloud beta run deploy vllm-gemma-2b-it \
  --image=vllm/vllm-openai:latest \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --memory=16Gi \
  --cpu=4 \
  --execution-environment=gen2 \
  --add-volume=name=model-volume,type=cloud-storage,bucket=aisprint-491218-bucket,readonly=true \
  --add-volume-mount=volume=model-volume,mount-path=/mnt/models \
  --args=--model=/mnt/models/gemma-2b-it,--max-model-len=4096,--port=8080 \
  --no-allow-unauthenticated \
  --region=us-east4 \
  --no-gpu-zonal-redundancy \
  --timeout=3600
```
