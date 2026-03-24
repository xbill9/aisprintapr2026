# Test Client for Hybrid Inference Orchestrator

import asyncio
from server import get_smart_route, analyze_latency_tradeoff

async def test_orchestrator():
    print("--- Testing Smart Routing (Simple Prompt) ---")
    route1 = await get_smart_route(prompt="What is 2+2?", budget_usd=0.10)
    print(f"Target: {route1['recommended_target']}")
    print(f"Reason: {route1['routing_reason']}")
    
    print("\n--- Testing Smart Routing (Complex Prompt) ---")
    complex_prompt = "Explain the architectural differences between a TPU v5e and an NVIDIA L4 GPU in the context of XLA graph compilation and HBM memory management." * 10
    route2 = await get_smart_route(prompt=complex_prompt, budget_usd=0.10)
    print(f"Target: {route2['recommended_target']}")
    print(f"Reason: {route2['routing_reason']}")

    print("\n--- Testing Latency Tradeoff (7B Model) ---")
    tradeoff = await analyze_latency_tradeoff(model_size_params="7b")
    print(f"7B Profile: {tradeoff['latency_profile']}")
    print(f"Verdict: {tradeoff['verdict']}")

if __name__ == "__main__":
    asyncio.run(test_orchestrator())
