from mcp.server.fastmcp import FastMCP
from google.cloud import aiplatform
from google.cloud import logging as cloud_logging
import os
import json
from typing import Dict, Any, List, Optional

# Initialize FastMCP server
mcp = FastMCP("DevOps Model Garden Agent")

# Configuration for Vertex AI Model Garden
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "aisprint")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
ENDPOINT_ID = os.getenv("MODEL_GARDEN_ENDPOINT_ID", "your-endpoint-id")

# Initialize Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=LOCATION)

@mcp.tool()
async def analyze_cloud_logging(filter_query: str, limit: int = 5) -> str:
    """
    Fetches and summarizes error logs from Google Cloud Logging using a Gemma Model Garden endpoint.
    
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
        
        # Query Model Garden Endpoint
        endpoint = aiplatform.Endpoint(f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}")
        # Note: Gemma endpoints typically expect a 'instances' and 'parameters' structure
        response = endpoint.predict(instances=[{"prompt": prompt}], parameters={"max_tokens": 512, "temperature": 0.2})
        
        return f"### Log Analysis (Gemma Model Garden)\n\n{response.predictions[0]}"
        
    except Exception as e:
        return f"Error analyzing logs via Vertex AI: {str(e)}"

@mcp.tool()
async def suggest_sre_remediation(error_message: str) -> str:
    """
    Proposes remediation steps for a specific SRE incident using Gemma from Model Garden.
    
    Args:
        error_message: The error or incident description to remediate.
    """
    prompt = f"As an expert SRE, suggest a 3-step remediation plan for the following error:\n\nError: {error_message}\n\nRemediation Plan:"
    
    try:
        endpoint = aiplatform.Endpoint(f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}")
        response = endpoint.predict(instances=[{"prompt": prompt}], parameters={"max_tokens": 512, "temperature": 0.2})
        return f"### Remediation Plan\n\n{response.predictions[0]}"
    except Exception as e:
        return f"Error fetching remediation plan: {str(e)}"

import subprocess

@mcp.tool()
def deploy_gemma_to_cloud_run(service_name: str, model_path: str) -> str:
    """
    Deploys a Gemma model container to Cloud Run with NVIDIA L4 GPU support.
    
    Args:
        service_name: The name of the Cloud Run service to create.
        model_path: The GCS path to the Gemma model weights (from Model Garden).
    """
    # This tool generates the gcloud command required to deploy Gemma to Cloud Run GPU.
    # It assumes the use of a vLLM or similar inference container.
    command = [
        "gcloud beta run deploy", service_name,
        "--image=us-docker.pkg.dev/vertex-ai/prediction/vllm-cpu:latest", # Example vLLM image
        "--gpu=1",
        "--gpu-type=nvidia-l4",
        "--memory=32Gi",
        "--cpu=8",
        f"--set-env-vars=MODEL_ID={model_path}",
        "--no-allow-unauthenticated",
        f"--region={LOCATION}"
    ]
    
    cmd_str = " ".join(command)
    
    # In a real environment, we might execute this via subprocess if authorized:
    # subprocess.run(command, check=True)
    
    return f"Deployment command generated for Cloud Run GPU:\n\n```bash\n{cmd_str}\n```\n\nNote: Ensure your project has GPU quota in {LOCATION}."

if __name__ == "__main__":
    mcp.run()
