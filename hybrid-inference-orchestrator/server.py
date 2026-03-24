import logging
import json
import random
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from google.cloud import run_v2

# Initialize FastMCP for the Hybrid Orchestrator
mcp = FastMCP("Hybrid Inference Orchestrator")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logger.getLogger("hybrid-orchestrator")

@mcp.tool()
async def list_inference_services(project_id: str, location: str = "us-central1"):
    """
    List all Cloud Run services in a project that have GPUs enabled.
    """
    try:
        client = run_v2.ServicesClient()
        parent = f"projects/{project_id}/locations/{location}"
        services = client.list_services(parent=parent)
        
        gpu_services = []
        for service in services:
            # Check for GPU in container resources
            for container in service.template.containers:
                if container.resources.limits and "nvidia.com/gpu" in container.resources.limits:
                    gpu_services.append({
                        "name": service.name,
                        "uri": service.uri,
                        "gpu_count": container.resources.limits["nvidia.com/gpu"]
                    })
        
        return {
            "project": project_id,
            "gpu_services": gpu_services or "No GPU services found."
        }
    except Exception as e:
        return f"Error listing Cloud Run services: {str(e)}"

@mcp.tool()
async def get_smart_route(prompt: str, max_tokens: int = 512, budget_usd: float = 0.05):
    """
    Decide the optimal hardware (TPU vs Cloud Run GPU) based on prompt complexity.
    """
    is_complex = len(prompt.split()) > 100
    target = "Cloud TPU v5e" if is_complex else "Cloud Run GPU (L4)"
    
    return {
        "recommended_target": target,
        "routing_reason": "High complexity" if is_complex else "Simple request",
        "estimated_cost_usd": 0.02 if target == "Cloud TPU v5e" else 0.005,
        "timestamp": datetime.now().isoformat()
    }

@mcp.tool()
async def deploy_to_cloud_run(service_name: str, image_uri: str, gcs_model_path: str):
    """
    Generate a Cloud Run GPU (L4) deployment command for a serverless inference endpoint.
    """
    gcloud_command = (
        f"gcloud beta run deploy {service_name} \\\n"
        f"  --image {image_uri} \\\n"
        f"  --no-allow-unauthenticated \\\n"
        f"  --gpu 1 --gpu-type nvidia-l4 \\\n"
        f"  --cpu 8 --memory 32Gi \\\n"
        f"  --set-env-vars MODEL_PATH={gcs_model_path} \\\n"
        f"  --region us-central1"
    )
    
    return {
        "status": "Configuration Generated",
        "command": gcloud_command,
        "recommendation": "Use 'min-instances 0' for maximum cost savings."
    }

@mcp.tool()
async def analyze_latency_tradeoff(model_size_params: str = "7b"):
    """
    Compare predicted P99 latency between TPU and Cloud Run GPU.
    """
    benchmarks = {
        "2b": {"tpu_p99_ms": 45, "cr_l4_p99_ms": 65, "delta": "+20ms (CR)"},
        "7b": {"tpu_p99_ms": 120, "cr_l4_p99_ms": 185, "delta": "+65ms (CR)"},
        "32b": {"tpu_p99_ms": 450, "cr_l4_p99_ms": "OOM (Likely)", "delta": "N/A"}
    }
    
    data = benchmarks.get(model_size_params.lower(), benchmarks["7b"])
    return {
        "model_size": model_size_params,
        "latency_profile": data,
        "verdict": "Use TPU for 7B+ models if P99 < 150ms is required."
    }

@mcp.resource("metrics://live-traffic-distribution")
def get_traffic_metrics():
    """Returns simulated live traffic distribution between TPU and GPU."""
    tpu_load = random.randint(40, 95)
    gpu_load = 100 - tpu_load
    return json.dumps({
        "tpu_cluster_utilization": f"{tpu_load}%",
        "cloud_run_gpu_utilization": f"{gpu_load}%",
        "active_requests": random.randint(10, 500)
    }, indent=2)

if __name__ == "__main__":
    mcp.run()
