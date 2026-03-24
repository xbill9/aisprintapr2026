# Hybrid Inference Orchestrator

This prototype demonstrates a hybrid architectural approach for Large Language Model (LLM) lifecycles during the TPU Sprint. 

## Concept
The core idea is to leverage the unique strengths of different hardware backends:
- **TPUs (Tensor Processing Units):** Used for intensive training, fine-tuning, and large-batch pre-processing where XLA optimization and high-bandwidth memory (HBM) provide a significant performance advantage.
- **Cloud Run GPU (Inference):** Used for cost-effective, auto-scaling, and low-latency inference. By deploying inference engines like vLLM or TGI on Cloud Run with NVIDIA L4 GPUs, we achieve serverless scaling for end-user requests.

The Orchestrator manages the transition from TPU training artifacts (stored in GCS) to production-ready Cloud Run services.

## Architecture
1. **Training/Fine-tuning:** Performed on TPU v5e/v5p.
2. **Model Export:** Weights are serialized and stored in Google Cloud Storage.
3. **Orchestration:** This MCP server provides tools to trigger deployments and benchmark the resulting inference endpoints.
4. **Inference:** Serverless GPU execution on Cloud Run.
