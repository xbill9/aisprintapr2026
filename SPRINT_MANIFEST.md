# TPU Sprint MCP Toolkit: Final Manifest 🚀

This toolkit provides a suite of 9 **Model Context Protocol (MCP)** servers designed to accelerate development and optimization for the **TPU Sprint**.

## 📦 Project Structure

### 1. [Hybrid Inference Orchestrator](./hybrid-inference-orchestrator)
*   **Concept:** Intelligent routing between TPU (Batch) and Cloud Run GPU (Inference).
*   **Demo:** Bursting from a high-load TPU cluster to serverless L4 GPUs.
*   **Key Tech:** `google-cloud-run` v2 SDK.

### 2. [Serverless vLLM Manager](./serverless-vllm-manager)
*   **Concept:** Optimized vLLM configurations for NVIDIA L4 GPUs on Cloud Run.
*   **Demo:** Shaving 30s+ off cold starts and analyzing real Cloud Run logs.
*   **Key Tech:** `vLLM` engine logic, `google-cloud-logging` SDK.

### 3. [Keras TPU-GPU Pipeline](./keras-tpu-gpu-pipeline)
*   **Concept:** Keras 3 multi-backend (JAX on TPU → PyTorch on GPU).
*   **Demo:** Training a classifier on TPU and converting weights for GPU serving.
*   **Key Tech:** `Keras 3` Multi-backend, `JAX`, `PyTorch`.

### 4. [TPU Performance Analyst](./tpu-performance-analyst)
*   **Concept:** Deep visibility into TPU hardware and XLA compilation.
*   **Demo:** Real JAX HLO inspection (dot products, kernel fusion) and HBM monitoring.
*   **Key Tech:** `JAX` HLO text analysis.

### 5. [XLA Cross-Hardware Profiler](./xla-cross-hardware-profiler)
*   **Concept:** Comparative profiling of TPU v5e vs. Cloud Run L4 GPU.
*   **Demo:** Side-by-side hardware spec and fusion efficiency comparison.
*   **Key Tech:** Comparative XLA heuristics.

### 6. [AutoResearch Serverless Manager](./autoresearch-serverless-manager)
*   **Concept:** Autonomous "Karpathy Loop" research on a $2/hour budget.
*   **Demo:** An AI agent (Gemini) optimizing JAX code for XLA fusion on serverless GPUs.
*   **Key Tech:** `Cloud Run Jobs`, `Cloud Workflows`, `Gemini`.

### 7. [Self-Hosted LLM Gateway](./self-hosted-llm-gateway)
*   **Concept:** Proxy for self-hosted models (Llama 3, Gemma) on Cloud Run.
*   **Demo:** Routing Gemini CLI queries to a local vLLM instance.
*   **Key Tech:** `LiteLLM`, `vLLM`, `FastAPI`.

### 8. [Self-Hosted vLLM DevOps Agent](./self-hosted-vllm-devops-agent)
*   **Concept:** SRE assistant using self-hosted vLLM on Cloud Run GPU.
*   **Demo:** Analyzing Cloud Logging errors using a private Gemma endpoint on L4 GPUs.
*   **Key Tech:** `vLLM`, `Cloud Run GPU`, `Cloud Logging`.

---

## 🛠 Management
*   **Global Makefile:** 
    *   `make install-all`: Install all dependencies.
    *   `make test-all`: Run all validation scripts.
    *   `make demo-all`: Execute all 7 Grand Demos.
*   **Unified README:** Comprehensive guide for the AI GDE community.

---

## ⚡ Sprint Entry Ideas
*   **Blog:** "Autonomous AI Ops: Letting Gemini Manage your Cloud TPUs."
*   **Video:** "Serverless vLLM on L4 GPUs: Performance for $2/hour."
*   **Repo:** "Keras 3: The Cross-Backend Pipeline for TPU and GPU."

**Manifest Saved. Ready for the Sprint!** ⚡🚀
