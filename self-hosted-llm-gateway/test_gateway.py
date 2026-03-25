import asyncio
import json
from server import query_local_llm, switch_model_backend, get_gateway_metrics

async def test_gateway():
    print("--- Testing 'get_gateway_metrics' ---")
    metrics = get_gateway_metrics()
    print(json.dumps(metrics, indent=2))
    
    print("\n--- Testing 'switch_model_backend' ---")
    switch_res = switch_model_backend("gemma2-9b")
    print(switch_res)
    
    print("\n--- Testing 'get_gateway_metrics' (post-switch) ---")
    metrics = get_gateway_metrics()
    print(json.dumps(metrics, indent=2))
    
    print("\n--- Testing 'query_local_llm' (Expect connection failure unless endpoint exists) ---")
    try:
        # This will fail unless LLM_GATEWAY_URL is set and reachable
        res = await query_local_llm("Hello, who are you?", model="llama3-8b")
        print(f"Response: {res}")
    except Exception as e:
        print(f"Expected failure: {e}")

if __name__ == "__main__":
    asyncio.run(test_gateway())
