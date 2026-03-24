# Grand Demo: XLA Cross-Hardware Profiler

import asyncio
from server import get_xla_metadata, compare_op_fusing, get_hardware_specs

async def profiler_demo():
    print("🚀 TPU Sprint Demo: XLA Cross-Hardware Profiler")
    print("=" * 60)
    
    # Step 1: TPU Metadata
    print("\n[Step 1] Extracting XLA Metadata for TPU v5e...")
    tpu = await get_xla_metadata(model_id="Gemma-7b", hardware="tpu-v5e")
    print(f"  HARDWARE: {tpu['metadata']['hardware']} | BF16 TFLOPS: {tpu['metadata']['peak_bf16_tflops']}")
    
    # Step 2: GPU Metadata
    print("\n[Step 2] Extracting XLA Metadata for NVIDIA L4 GPU...")
    gpu = await get_xla_metadata(model_id="Gemma-7b", hardware="gpu-l4")
    print(f"  HARDWARE: {gpu['metadata']['hardware']} | BF16 TFLOPS: {gpu['metadata']['peak_bf16_tflops']}")
    
    # Step 3: Comparative Analysis
    print("\n[Step 3] Cross-Hardware Fusion Comparison (XLA HLO)...")
    await asyncio.sleep(1)
    compare = await compare_op_fusing(tpu_meta=tpu, gpu_meta=gpu)
    print(f"  ANALYSIS: {compare}")
    
    # Step 4: Hardware spec lookup
    print("\n[Step 4] Hardware Comparison Resource (TPU vs GPU)...")
    specs = get_hardware_specs()
    print(f"  MEMORY BANDWIDTH (TPU): {specs.split('Memory Bandwidth')[-1].split('|')[1].strip()}")
    print(f"  VRAM (L4 GPU): {specs.split('VRAM / HBM')[-1].split('|')[2].strip()}")
    
    print("\n" + "=" * 60)
    print("✅ Hardware Profiling Complete: Choose the best target for your model!")

if __name__ == "__main__":
    asyncio.run(profiler_demo())
