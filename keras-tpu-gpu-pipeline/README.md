# Keras 3 TPU-GPU Pipeline

This prototype demonstrates the "Train Once, Deploy Everywhere" capability of Keras 3 during the TPU Sprint.

## Overview
Using Keras 3's multi-backend support, we can build a unified pipeline that:
- **Trains on TPUs using the JAX backend:** Leveraging XLA and HBM for high-throughput training.
- **Deploys on GPUs using the PyTorch backend:** Providing seamless integration with common GPU inference stacks.

## Workflow
1. **Source Code:** A single model definition using standard Keras 3 ops.
2. **Backend Selection:** Toggle `KERAS_BACKEND=jax` for the training phase on TPU v5e.
3. **Weight Export:** Save weights in a framework-agnostic `.h5` or `.keras` format.
4. **Backend Swap:** Toggle `KERAS_BACKEND=torch` or `KERAS_BACKEND=tensorflow` for the inference phase on NVIDIA L4.
5. **Consistency Check:** Verify model outputs are within tolerance across backends.
