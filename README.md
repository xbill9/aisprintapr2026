# TPU Sprint MCP Toolkit 🚀

Welcome to the **TPU Sprint MCP Toolkit**! This repository is a collection of **nine specialized Model Context Protocol (MCP)** servers designed to accelerate development, optimization, and deployment on **Cloud TPUs** and **Serverless GPUs (Cloud Run L4)**.

Built with **FastMCP (Python)**, these tools give LLMs (like Gemini) "eyes and hands" to manage high-performance AI infrastructure.

---

## 🛠 Included Prototypes

### 1. [Hybrid Inference Orchestrator](./hybrid-inference-orchestrator)
*   **Focus:** Intelligent routing between TPU (Training/Batch) and Cloud Run GPU (Inference).
*   **Key Tools:** `deploy_to_cloud_run`, `estimate_hybrid_costs`, `toggle_traffic_policy`.
*   **Use Case:** Cost-optimizing LLM inference by scaling to zero on Cloud Run during low traffic.

### 2. [Serverless vLLM Manager](./serverless-vllm-manager)
*   **Focus:** Managing vLLM on Cloud Run with NVIDIA L4 GPUs.
*   **Key Tools:** `optimize_vllm_config`, `monitor_cold_start`.
*   **Use Case:** Shaving 30s+ off cold starts and preventing OOMs on serverless L4 GPUs.

### 3. [Keras TPU-GPU Pipeline](./keras-tpu-gpu-pipeline)
*   **Focus:** Keras 3 multi-backend workflows (JAX on TPU → PyTorch on GPU).
*   **Key Tools:** `init_training_job`, `convert_weights`.
*   **Use Case:** Training at scale on TPUs and deploying anywhere with zero code changes.

### 4. [TPU Performance Analyst](./tpu-performance-analyst)
*   **Focus:** Deep visibility into TPU hardware and XLA compilation.
*   **Key Tools:** `inspect_hlo`, `monitor_tpu_hbm`.
*   **Use Case:** Debugging XLA bottlenecks and HBM fragmentation using natural language.

### 5. [XLA Cross-Hardware Profiler](./xla-cross-hardware-profiler)
*   **Focus:** Comparative analysis of TPU v5e vs. Cloud Run L4 GPU.
*   **Key Tools:** `get_xla_metadata`, `compare_op_fusing`.
*   **Use Case:** Understanding how the same model architecture behaves across different hardware targets.

### 6. [AutoResearch Serverless Manager](./autoresearch-serverless-manager)
*   **Focus:** Automating the Karpathy "research loop" on Google Cloud Serverless.
*   **Key Tools:** `submit_research_job`, `monitor_research_workflow`, `analyze_research_costs`.
*   **Use Case:** Running autonomous ML research experiments (like XLA optimization) for ~$2/hour using serverless GPUs.

### 7. [Self-Hosted LLM Gateway](./self-hosted-llm-gateway)
*   **Focus:** Unified proxy for self-hosted open-weight models on serverless GPUs.
*   **Key Tools:** `query_local_llm`, `switch_model_backend`, `get_gateway_metrics`.
*   **Use Case:** Routing between local specialized models (Llama, Mixtral) and orchestration LLMs.

### 8. [Self-Hosted vLLM DevOps Agent](./self-hosted-vllm-devops-agent)
*   **Focus:** Private DevOps assistant leveraging vLLM on Cloud Run GPU.
*   **Key Tools:** `get_vllm_deployment_config`, `get_vertex_ai_model_copy_instructions`, `analyze_cloud_logging`.
*   **Use Case:** Running a fully private SRE loop with Gemma models without external API dependencies.

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   `pip`
*   GCP Project with Cloud Run GPU and TPU access.

### Quick Setup
Use the root `Makefile` to set up all prototypes at once:

```bash
# Install dependencies for all projects
make install-all

# Run tests for all projects
make test-all
```

### Running an MCP Server
To use a specific server with an MCP client (like Claude Desktop or Gemini), point your configuration to the `server.py` file in the desired directory:

```bash
cd serverless-vllm-manager
python server.py
```

---

## 💡 Sprint Content Ideas
These prototypes are designed to be the foundation for your TPU Sprint entries. Use them to:
*   Write a blog post on "Autonomous AI Ops for TPUs."
*   Record a YouTube tutorial on "Serverless vLLM with Cloud Run GPUs."
*   Create a GitHub sample showing "Keras 3: Train on TPU, Serve on GPU."

---

## 📄 License
This toolkit is provided under the Apache 2.0 License.

**Happy Sprinting, AI GDEs!** ⚡
