# Test Client for Serverless vLLM Manager

import asyncio
from mcp.server.fastmcp import FastMCP
from server import optimize_vllm_config, monitor_cold_start

async def test_tools():
    print("--- Testing optimize_vllm_config ---")
    # Simulate a request for Gemma 7B on L4
    opt_result = await optimize_vllm_config(model_id="google/gemma-7b", max_model_len=4096)
    print(f"Config for Gemma 7B: {opt_result['vllm_config']}")
    print(f"Analysis: {opt_result['analysis']}")
    
    print("\n--- Testing monitor_cold_start ---")
    # Simulate log analysis
    cold_start_result = await monitor_cold_start(
        project_id="tpu-sprint-project", 
        service_name="vllm-gemma-l4"
    )
    print(f"Cold Start Analysis: {cold_start_result}")

if __name__ == "__main__":
    asyncio.run(test_tools())
