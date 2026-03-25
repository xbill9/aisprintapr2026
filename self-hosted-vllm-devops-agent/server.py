import os
import json
import requests
from mcp.server.fastmcp import FastMCP
from google.cloud import logging as cloud_logging
from typing import Dict, Any, List, Optional

from google.cloud import aiplatform
from google.cloud import storage
import kagglehub

import os
import json
import requests
from mcp.server.fastmcp import FastMCP
from google.cloud import logging as cloud_logging
from typing import Dict, Any, List, Optional

from google.cloud import aiplatform
from google.cloud import storage
import kagglehub

# Initialize FastMCP server
mcp = FastMCP("Self-Hosted vLLM DevOps Agent")

# Configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "aisprint-491218")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-east4")
BUCKET_NAME = f"{PROJECT_ID}-bucket"
# The URL of the self-hosted vLLM service on Cloud Run
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-2b-it")

def discover_vllm_url(service_name: str = "vllm-gemma-2b-it") -> str:
    """Attempts to automatically discover the Cloud Run service URL."""
    if VLLM_BASE_URL:
        return VLLM_BASE_URL
    
    import subprocess
    try:
        cmd = [
            "gcloud", "run", "services", "describe", service_name,
            "--platform", "managed",
            "--region", LOCATION,
            "--format", "value(status.url)"
        ]
        url = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        if url:
            print(f"📡 Automatically discovered vLLM at: {url}")
            return url
    except Exception:
        pass
    
    return "http://localhost:8080"

# Resolve base URL at runtime
ACTIVE_VLLM_URL = discover_vllm_url()

# Initialize Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=LOCATION)

@mcp.resource("config://vllm-deployment-template")
def get_deployment_template() -> str:
    """Returns a base template for Cloud Run GPU deployment."""
    return f"""
# Cloud Run vLLM Deployment Template
# Required: Second Generation execution environment
# Required: NVIDIA L4 GPU
# Required: GCS FUSE mount

service: vllm-gemma-2b-it
image: vllm/vllm-openai:latest
resources:
  limits:
    nvidia.com/gpu: 1
    cpu: 4
    memory: 16Gi
annotations:
  run.googleapis.com/execution-environment: gen2
  run.googleapis.com/gpu-zonal-redundancy-disabled: "true"
  run.googleapis.com/cpu-throttling: "false"  # Mandatory for GPU
  run.googleapis.com/startup-cpu-boost: "true"
  run.googleapis.com/maxScale: "1"
container:
  concurrency: 4
  timeout: 3600s
startupProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 180
  periodSeconds: 60
  failureThreshold: 10
  timeoutSeconds: 60
# For gcloud deployment, use:
# gcloud run deploy vllm-gemma-2b-it --no-cpu-throttling --allow-unauthenticated --concurrency=4 \\
#   --timeout=3600 --startup-probe=timeoutSeconds=60,periodSeconds=60,failureThreshold=10,initialDelaySeconds=180,httpGet.port=8000,httpGet.path=/health \\
#   --max-instances=1 --args=--model=/mnt/models/gemma-2b-it,--max-model-len=4096,--trust-remote-code,--gpu-memory-utilization=0.9,--host=0.0.0.0
volumes:
  - name: model-volume
    cloudStorage:
      bucket: {BUCKET_NAME}
      readonly: true
"""

@mcp.tool()
def list_vertex_models() -> str:
    """
    Uses the Vertex AI SDK (part of ADK ecosystem) to list models in the project registry.
    """
    try:
        models = aiplatform.Model.list()
        if not models:
            return "No models found in Vertex AI Model Registry."
        
        model_list = [f"- {m.display_name} (ID: {m.name})" for m in models]
        return "### Vertex AI Model Registry\n" + "\n".join(model_list)
    except Exception as e:
        return f"Error listing models from Vertex AI: {str(e)}"

@mcp.tool()
def list_bucket_models(bucket_name: str = BUCKET_NAME) -> str:
    """
    Lists the contents of the GCS bucket to check for uploaded model files.
    
    Args:
        bucket_name: The GCS bucket name to check. Defaults to BUCKET_NAME.
    """
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        # List up to 100 blobs
        blobs = list(bucket.list_blobs(max_results=100))
        
        if not blobs:
            return f"The bucket '{bucket_name}' is empty."
        
        # Display up to 50 for brevity
        file_list = [f"- {b.name} ({b.size / 1024 / 1024:.2f} MB)" for b in blobs[:50]]
        summary = f"### Contents of GCS Bucket: {bucket_name}\n"
        summary += "\n".join(file_list)
        
        if len(blobs) > 50:
            summary += f"\n\n(Showing 50 of {len(blobs)} items)"
            
        return summary
    except Exception as e:
        return f"Error listing objects in bucket '{bucket_name}': {str(e)}"

@mcp.tool()
async def analyze_cloud_logging(filter_query: str, limit: int = 5) -> str:
    """
    Fetches and summarizes error logs from Google Cloud Logging using a self-hosted vLLM endpoint on Cloud Run.
    
    Args:
        filter_query: Filter for Cloud Logging (e.g., 'severity=ERROR').
        limit: Number of recent logs to fetch.
    """
    try:
        logging_client = cloud_logging.Client(project=PROJECT_ID)
        entries = list(logging_client.list_entries(filter_=filter_query, order_by=cloud_logging.DESCENDING, page_size=limit))
        
        if not entries:
            return "No matching logs found."
        
        log_texts = [f"Timestamp: {e.timestamp} | Severity: {e.severity} | Message: {e.payload if isinstance(e.payload, str) else json.dumps(e.payload)}" for e in entries]
        combined_logs = "\n---\n".join(log_texts)
        
        # Prepare prompt for Gemma
        prompt = f"Analyze the following DevOps logs and provide a high-level summary of the critical issues and potential root causes:\n\n{combined_logs}\n\nSummary:"
        
        # Query Self-Hosted vLLM (OpenAI compatible API)
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"{ACTIVE_VLLM_URL}/v1/completions",
            headers=headers,
            json={
                "model": f"/mnt/models/{MODEL_NAME.split('/')[-1]}", # Match the path in Cloud Run
                "prompt": prompt,
                "max_tokens": 512,
                "temperature": 0.2
            }
        )
        response.raise_for_status()
        result = response.json()
        
        return f"### Log Analysis (Self-Hosted vLLM)\n\n{result['choices'][0]['text']}"
        
    except Exception as e:
        return f"Error analyzing logs via self-hosted vLLM: {str(e)}"

@mcp.tool()
async def suggest_sre_remediation(error_message: str) -> str:
    """
    Proposes remediation steps for a specific SRE incident using self-hosted vLLM.
    
    Args:
        error_message: The error or incident description to remediate.
    """
    prompt = f"As an expert SRE, suggest a 3-step remediation plan for the following error:\n\nError: {error_message}\n\nRemediation Plan:"
    
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"{ACTIVE_VLLM_URL}/v1/completions",
            headers=headers,
            json={
                "model": f"/mnt/models/{MODEL_NAME.split('/')[-1]}",
                "prompt": prompt,
                "max_tokens": 512,
                "temperature": 0.2
            }
        )
        response.raise_for_status()
        result = response.json()
        return f"### Remediation Plan\n\n{result['choices'][0]['text']}"
    except Exception as e:
        return f"Error fetching remediation plan: {str(e)}"

@mcp.tool()
async def query_vllm(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """
    Directly queries the self-hosted vLLM model with a custom prompt.
    
    Args:
        prompt: The text prompt to send to the model.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 for deterministic).
    """
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"{ACTIVE_VLLM_URL}/v1/completions",
            headers=headers,
            json={
                "model": f"/mnt/models/{MODEL_NAME.split('/')[-1]}",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )
        response.raise_for_status()
        result = response.json()
        return f"### vLLM Response\n\n{result['choices'][0]['text']}"
    except Exception as e:
        return f"Error querying vLLM: {str(e)}"

@mcp.tool()
def get_vllm_deployment_config(service_name: str = "vllm-gemma-2b-it", bucket_name: str = BUCKET_NAME, model_path: str = "gemma-2b-it", allow_unauthenticated: bool = False, min_instances: int = 0, gpu_memory_utilization: float = 0.9) -> str:
    """
    Generates the gcloud command to deploy vLLM to Cloud Run with GCS FUSE and NVIDIA L4 GPU.

    Args:
        service_name: The name for the Cloud Run service.
        bucket_name: The GCS bucket containing the model weights.
        model_path: The sub-path inside the bucket (e.g., 'gemma-2b-it').
        allow_unauthenticated: Whether to allow unauthenticated access to the service.
        min_instances: The minimum number of instances to keep warm (default: 0).
        gpu_memory_utilization: The fraction of GPU memory to use for KV cache (default: 0.9).
    """
    # Note: Private Google Access must be enabled on the VPC subnet for GCS FUSE to work.
    command = [
        "gcloud beta run deploy", service_name,
        "--image=vllm/vllm-openai:latest",
        "--gpu=1",
        "--gpu-type=nvidia-l4",
        "--no-gpu-zonal-redundancy", # Fix for quota issues in us-east4
        "--no-cpu-throttling", # Required for GPU deployment
        "--concurrency=4", # Optimal for LLM throughput vs latency
        "--timeout=3600", # 1 hour timeout for long generations
        "--startup-probe=timeoutSeconds=60,periodSeconds=60,failureThreshold=10,initialDelaySeconds=180,httpGet.port=8000,httpGet.path=/health",
        "--max-instances=1", # Prevent scaling beyond quota
        f"--min-instances={min_instances}",
        "--port=8000", # vLLM default port
        "--memory=16Gi",
        "--cpu=4",
        "--execution-environment=gen2",
        f"--add-volume=name=model-volume,type=cloud-storage,bucket={bucket_name},readonly=true",
        "--add-volume-mount=volume=model-volume,mount-path=/mnt/models",
        # vLLM arguments passed as comma-separated list
        f"--args=--model=/mnt/models/{model_path},--max-model-len=4096,--trust-remote-code,--gpu-memory-utilization={gpu_memory_utilization},--host=0.0.0.0",
        "--allow-unauthenticated" if allow_unauthenticated else "--no-allow-unauthenticated",
        f"--region={LOCATION}"
    ]

    return " ".join(command)
@mcp.tool()
def get_vllm_tpu_deployment_config(cluster_name: str = "tpu-cluster", model_name: str = "google/gemma-2-9b-it") -> str:
    """
    Generates a GKE manifest and setup instructions for deploying vLLM on TPU v5e.
    
    Args:
        cluster_name: The name of the GKE cluster.
        model_name: The model identifier (e.g., 'google/gemma-2-9b-it').
    """
    manifest = f"""
### 🌀 vLLM on TPU v5e (GKE Deployment)

To deploy vLLM on TPUs, use the following GKE manifest. This configuration targets a **TPU v5e-8** (8 chips) which is ideal for Gemma 2 9B or 27B.

#### 1. Create a TPU Node Pool (if not exists)
```bash
gcloud container node-pools create tpu-v5e-8 \\
    --cluster={cluster_name} \\
    --location={LOCATION} \\
    --machine-type=ct5lp-hightpu-4t \\
    --tpu-topology=2x4 \\
    --num-nodes=1
```

#### 2. Kubernetes Manifest (vllm-tpu.yaml)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-tpu
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-tpu
  template:
    metadata:
      labels:
        app: vllm-tpu
    spec:
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu-v5-lite-podslice
        cloud.google.com/gke-tpu-topology: 2x4
      containers:
      - name: vllm-tpu
        image: vllm/vllm-tpu:latest
        resources:
          limits:
            google.com/tpu: "8"
          requests:
            google.com/tpu: "8"
        env:
        - name: VLLM_XLA_CACHE_PATH
          value: "/tmp/vllm_xla_cache"
        command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
        args:
        - "--model={model_name}"
        - "--tensor-parallel-size=8"
        - "--max-model-len=8192"
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: dshm
          mountPath: /dev/shm
      volumes:
      - name: dshm
        emptyDir:
          medium: Memory
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-tpu-service
spec:
  selector:
    app: vllm-tpu
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

#### 3. Deployment Steps
1. Save the YAML above to `vllm-tpu.yaml`.
2. Apply it: `kubectl apply -f vllm-tpu.yaml`.
3. (Optional) If using a private model, ensure a Hugging Face token is provided via secret.
"""
    return manifest

@mcp.tool()
def get_vertex_ai_model_copy_instructions(model_name: str = "gemma-2b-it") -> str:
    """
    Provides instructions and commands to transfer Gemma model artifacts from Vertex AI Model Garden to your GCS bucket.
    """
    instructions = f"""
### 🚀 Transferring {model_name} from Vertex AI Model Garden

To use vLLM with Cloud Storage FUSE without Hugging Face, follow these steps:

1. **Accept Terms:** Go to the Vertex AI Model Garden page for Gemma (https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/335) and click 'Accept' on the license agreement.
2. **Download via Signed URL:** After accepting, the console provides a 'Download' button or a signed URL.
3. **Transfer to GCS:**
   If you have the artifacts locally after downloading from the console, use:
   `gcloud storage cp -r ./model_artifacts/* gs://{BUCKET_NAME}/{model_name}/`

4. **Alternative (Direct GCS Copy):**
   Google occasionally provides a managed GCS path for verified projects. If accessible, you can use:
   `gcloud storage cp -r gs://vertex-ai-models/gemma/{model_name}/* gs://{BUCKET_NAME}/{model_name}/`

Once the artifacts are in your bucket, use `get_vllm_deployment_config` to generate your Cloud Run deployment command.
"""
    return instructions

@mcp.tool()
def get_kagglehub_download_path(model_slug: str = "google/gemma/transformers/2b-it/1") -> str:
    """
    Returns the local cache path for a Kaggle model using kagglehub.
    Note: This may trigger a download if the model is not already in the cache.
    """
    try:
        path = kagglehub.model_download(model_slug)
        return f"Model '{model_slug}' is available at: {path}"
    except Exception as e:
        return f"Error resolving kagglehub path: {str(e)}"

@mcp.tool()
def get_kaggle_model_copy_instructions(model_slug: str = "google/gemma/transformers/2b-it/1", bucket_name: str = BUCKET_NAME) -> str:
    """
    Provides instructions and commands to transfer Gemma model weights from Kaggle to your GCS bucket.
    
    Args:
        model_slug: The Kaggle model slug (e.g., 'google/gemma/transformers/2b-it/1').
        bucket_name: The target GCS bucket name.
    """
    # Extract a friendly name from the slug (e.g., '2b-it')
    parts = model_slug.split('/')
    model_name = parts[-2] if len(parts) > 2 else "gemma-model"
    
    instructions = f"""
### 📦 Transferring {model_name} from Kaggle to GCS

To use Kaggle weights with vLLM on Cloud Run, follow these steps:

#### Option A: Using `kagglehub` (Recommended)
`kagglehub` simplifies the download process and requires no local `kaggle.json` if configured with environmental variables.

1. **Download Model:**
   `python3 -c "import kagglehub; print(kagglehub.model_download('{model_slug}'))"`

2. **Upload to GCS:**
   The command above outputs the local path. Use it to copy the artifacts:
   `gcloud storage cp -r /path/to/downloaded/model/* gs://{bucket_name}/{model_name}/`

#### Option B: Using `kaggle` CLI
1. **Setup Kaggle API:**
   Ensure you have `kaggle.json` in `~/.kaggle/` (Download from Kaggle Settings > Create New Token).
   `chmod 600 ~/.kaggle/kaggle.json`

2. **Download Model Artifacts:**
   `mkdir -p ./{model_name}`
   `kaggle models instances versions download {model_slug} --path ./{model_name}`

3. **Extract Weights:**
   `unzip ./{model_name}/*.zip -d ./{model_name}/`
   `rm ./{model_name}/*.zip`

4. **Upload to GCS Bucket:**
   `gcloud storage cp -r ./{model_name}/* gs://{bucket_name}/{model_name}/`

Once uploaded, you can deploy using:
`get_vllm_deployment_config(model_path="{model_name}")`
"""
    return instructions

if __name__ == "__main__":
    mcp.run()
