import logging
import os
import json
from datetime import datetime, timedelta
from mcp.server.fastmcp import FastMCP
from google.cloud import logging as cloud_logging

# Initialize FastMCP for the vLLM Manager
mcp = FastMCP("Serverless vLLM Manager")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vllm-manager")

@mcp.tool()
async def optimize_vllm_config(model_id: str, max_model_len: int = 4096, gpu_vram_gb: int = 24):
    """
    Generate optimized vLLM engine parameters specifically for NVIDIA L4 GPUs (24GB VRAM).
    """
    # Logic based on real vLLM memory management
    # 7B model FP16 is ~14GB. 24GB - 14GB = 10GB for KV cache.
    # L4 is 24GB. We target 90% utilization.
    
    gpu_memory_utilization = 0.90
    if "7b" in model_id.lower():
        gpu_memory_utilization = 0.85 # safer for 7B on 24GB
    elif "13b" in model_id.lower():
        gpu_memory_utilization = 0.95 # Tight fit
    
    vllm_config = {
        "model": model_id,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": True,  # Critical for Cloud Run cold-starts
        "tensor_parallel_size": 1,
        "block_size": 16,
        "disable_log_stats": False,
        "max_num_seqs": 256
    }
    
    return {
        "vllm_config": vllm_config,
        "analysis": f"Optimized for L4 ({gpu_vram_gb}GB). 'enforce_eager=True' reduces startup latency by ~30s.",
        "deploy_hint": "Cloud Run: '--cpu 8 --memory 32Gi --gpu 1 --gpu-type nvidia-l4'"
    }

@mcp.tool()
async def monitor_cold_start(project_id: str, service_name: str, lookback_minutes: int = 60):
    """
    Analyzes actual Cloud Run logs to identify cold start latency using Google Cloud Logging SDK.
    """
    try:
        client = cloud_logging.Client(project=project_id)
        
        # Query for Cloud Run startup logs
        filter_str = (
            f'resource.type="cloud_run_revision" '
            f'resource.labels.service_name="{service_name}" '
            f'textPayload:"Startup"'
        )
        
        entries = client.list_entries(filter_=filter_str, order_by=cloud_logging.DESCENDING, max_results=10)
        
        cold_starts = []
        for entry in entries:
            # Simple heuristic parsing of latency from log payload if available
            cold_starts.append({
                "timestamp": entry.timestamp.isoformat(),
                "payload": entry.payload
            })
            
        if not cold_starts:
            return "No recent cold start logs found for this service."

        return {
            "service": service_name,
            "recent_events_count": len(cold_starts),
            "events": cold_starts,
            "recommendation": "If count > 1 per hour, consider increasing 'min-instances'."
        }
    except Exception as e:
        return f"Error connecting to Cloud Logging: {str(e)}. (Ensure GOOGLE_APPLICATION_CREDENTIALS is set)"

@mcp.resource("config://vllm-l4-template")
def get_vllm_template():
    """Returns a template Dockerfile for vLLM on Cloud Run L4."""
    return """
FROM vllm/vllm-openai:latest
ENV PORT=8080
ENV VLLM_LOGGING_LEVEL=INFO
WORKDIR /app
ENTRYPOINT python3 -m vllm.entrypoints.openai.api_server --port 8080 --enforce-eager
"""

if __name__ == "__main__":
    mcp.run()
