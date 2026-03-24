# Test Client for TPU Performance Analyst

import asyncio
from server import inspect_hlo, monitor_tpu_hbm

async def test_analyst():
    print("--- Testing HLO Inspection ---")
    hlo_result = await inspect_hlo(hlo_text_path="model_compilation.hlo")
    print(f"Fusion Ratio: {hlo_result['hlo_analysis']['summary']['fusion_ratio']}")
    print(f"Top Bottleneck: {hlo_result['hlo_analysis']['bottlenecks'][0]['op']}")
    print(f"Recommendation: {hlo_result['recommendation']}")
    
    print("\n--- Testing TPU HBM Monitoring ---")
    hbm_result = await monitor_tpu_hbm(tpu_name="tpu-v5e-slice-1")
    print(f"HBM Metrics: {hbm_result}")

if __name__ == "__main__":
    asyncio.run(test_analyst())
