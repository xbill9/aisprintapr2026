# Test Client for XLA Cross-Hardware Profiler

import asyncio
import json
from server import get_xla_metadata, compare_op_fusing

async def test_profiler():
    print("--- Testing XLA Metadata Extraction (TPU) ---")
    tpu_meta = await get_xla_metadata(model_id="Gemma-7b", hardware="tpu-v5e")
    print(f"TPU Hardware: {tpu_meta['metadata']['hardware']}")
    
    print("\n--- Testing XLA Metadata Extraction (GPU) ---")
    gpu_meta = await get_xla_metadata(model_id="Gemma-7b", hardware="gpu-l4")
    print(f"GPU Hardware: {gpu_meta['metadata']['hardware']}")
    
    print("\n--- Testing Cross-Hardware Comparison ---")
    comparison = await compare_op_fusing(tpu_meta=tpu_meta, gpu_meta=gpu_meta)
    print(f"Comparison: {comparison}")

if __name__ == "__main__":
    asyncio.run(test_profiler())
