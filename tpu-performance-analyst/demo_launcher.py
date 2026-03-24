# Grand Demo: TPU Performance Analyst

import asyncio
from server import inspect_hlo, monitor_tpu_hbm, get_xla_guide

async def analyst_demo():
    print("🚀 TPU Sprint Demo: TPU Performance Analyst")
    print("=" * 60)
    
    # Step 1: Real JAX HLO Inspection
    print("\n[Step 1] Compiling JAX sample and inspecting XLA HLO...")
    hlo = await inspect_hlo()
    print(f"  HLO SUMMARY: {hlo['summary']}")
    print(f"  HLO SNIPPET: {hlo['hlo_snippet']}")
    print(f"  RECOMMENDATION: {hlo['recommendation']}")
    
    # Step 2: HBM Monitoring
    print("\n[Step 2] Monitoring HBM Memory (Cloud TPU v5e-4)...")
    await asyncio.sleep(1)
    hbm = await monitor_tpu_hbm(tpu_name="tpu-v5e-slice-1")
    print(f"  METRICS: {hbm}")
    
    # Step 3: Best Practices Retrieval
    print("\n[Step 3] Retrieving XLA Optimization Guide...")
    guide = get_xla_guide()
    print(f"  TOP TIP: {guide.split('1.')[-1].split('2.')[0].strip()}")
    print(f"  XLA FLAG TIP: {guide.split('4.')[-1].split('5.')[0].strip()}")
    
    print("\n" + "=" * 60)
    print("✅ TPU Performance Analysis Complete: Optimized XLA kernels found!")

if __name__ == "__main__":
    asyncio.run(analyst_demo())
