# XLA Cross-Hardware Profiler

This prototype provides automated comparison of XLA graph compilation across different hardware backends during the TPU Sprint.

## Overview
XLA (Accelerated Linear Algebra) compiles model graphs specifically for each target hardware. This profiler allows you to see how the *same* model logic is optimized for:
- **Google TPU v5e:** Optimized for high-throughput matrix multiplication and cross-core communication.
- **NVIDIA L4 GPU:** Optimized for CUDA kernels and specific GPU-based operation fusing.

## Comparison Points
- **Kernel Fusing:** Comparing the number and complexity of fused kernels on TPU vs GPU.
- **Data Layout:** Differences in how tensors are laid out in memory for each backend.
- **Operation Mapping:** How high-level JAX operations are mapped to hardware primitives (e.g., TPU MXUs vs GPU Tensor Cores).
