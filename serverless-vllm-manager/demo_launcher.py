# Grand Demo: Serverless vLLM Manager

import asyncio
from server import optimize_vllm_config, monitor_cold_start, get_vllm_template

async def vllm_demo():
    print("🚀 TPU Sprint Demo: Serverless vLLM Manager")
    print("=" * 60)
    
    # Step 1: Optimize for Cloud Run L4
    print("\n[Step 1] Optimizing Gemma-7b for L4 (24GB VRAM)...")
    opt = await optimize_vllm_config(model_id="google/gemma-7b", max_model_len=4096)
    print(f"  VRAM TARGET: {opt['vllm_config']['gpu_memory_utilization'] * 100}%")
    print(f"  ENFORCE_EAGER: {opt['vllm_config']['enforce_eager']} (reduces CUDA graph latency)")
    print(f"  DEPLOY_HINT: {opt['deploy_hint']}")
    
    # Step 2: Resource retrieval
    print("\n[Step 2] Retrieving L4-optimized Dockerfile Template...")
    dockerfile = get_vllm_template()
    print(f"  ENTRYPOINT: {dockerfile.split('ENTRYPOINT')[-1].strip()}")
    
    # Step 3: Real Cold-Start Analysis
    print("\n[Step 3] Analyzing Cloud Run logs for cold-starts (last 60 mins)...")
    await asyncio.sleep(1) # Simulated network delay
    # Simulate a call where some results are found
    cs = await monitor_cold_start(project_id="tpu-sprint-demo", service_name="vllm-gemma-l4")
    print(f"  METRICS: {cs}")
    
    print("\n" + "=" * 60)
    print("✅ vLLM Optimization Complete: Ready for Serverless GPU deployment!")

if __name__ == "__main__":
    asyncio.run(vllm_demo())
