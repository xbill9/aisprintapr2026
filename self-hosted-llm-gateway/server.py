from mcp.server.fastmcp import FastMCP
import httpx
import os
import json
from typing import Dict, Any, Optional

# Initialize FastMCP server
mcp = FastMCP("Self-Hosted LLM Gateway")

# Configuration for the self-hosted vLLM or LiteLLM endpoint
BASE_URL = os.getenv("LLM_GATEWAY_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("LLM_GATEWAY_API_KEY", "EMPTY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3-8b")

current_model = DEFAULT_MODEL

@mcp.tool()
async def query_local_llm(prompt: str, model: Optional[str] = None, max_tokens: int = 512) -> str:
    """
    Routes a prompt to the self-hosted LLM endpoint (vLLM/LiteLLM).
    
    Args:
        prompt: The text prompt to send to the model.
        model: Optional model name to override the default.
        max_tokens: Maximum number of tokens to generate.
    """
    target_model = model or current_model
    
    payload = {
        "model": target_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{BASE_URL}/chat/completions", json=payload, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"Error from gateway: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Failed to connect to LLM gateway at {BASE_URL}: {str(e)}"

@mcp.tool()
def switch_model_backend(model: str) -> str:
    """
    Dynamically swaps the underlying model for the gateway.
    
    Args:
        model: The name of the new model (e.g., 'llama3-70b', 'gemma2-9b').
    """
    global current_model
    old_model = current_model
    current_model = model
    return f"Gateway switched from '{old_model}' to '{current_model}'."

@mcp.tool()
def get_gateway_metrics() -> Dict[str, Any]:
    """
    Returns metrics for the LLM gateway, including latency and GPU utilization.
    Note: In a real implementation, these would be fetched from the vLLM/Cloud Run metrics.
    """
    return {
        "active_model": current_model,
        "endpoint": BASE_URL,
        "status": "healthy",
        "mock_metrics": {
            "avg_latency_ms": 450,
            "tokens_per_second": 45.2,
            "gpu_utilization_pct": 68.5,
            "request_count": 1240
        }
    }

if __name__ == "__main__":
    mcp.run()
