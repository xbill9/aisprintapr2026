# Serverless vLLM Manager

This prototype focuses on managing and optimizing vLLM deployments on serverless GPU infrastructure during the TPU Sprint.

## Overview
Efficiently running large language models in a serverless environment (Cloud Run on NVIDIA L4 GPUs) requires careful configuration to balance cost, cold start times, and throughput.

### Key Features
- **GPU Configuration:** Optimized settings for NVIDIA L4 (24GB VRAM) to handle KV cache and model weights.
- **Cold Start Mitigation:** Strategies to reduce initialization time, such as image layer optimization and model weight caching.
- **Dynamic Scaling:** Leveraging Cloud Run's concurrency-based scaling to handle variable load.

## TPU Sprint Focus
This manager helps bridge the gap from models trained on TPUs to their high-performance deployment using the vLLM engine, specifically targeting the price-performance sweet spot of L4 GPUs.
