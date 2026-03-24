# TPU Sprint: MCP & ADK Project Ideas

This document outlines high-impact project ideas for the TPU Sprint, focusing on **JAX/Keras/vLLM/PyTorch on XLA/TPU** and **GPU on Cloud Run**, integrated via the **Model Context Protocol (MCP)** and **Agent Development Kit (ADK)**.

---

## 1. The "Hybrid Inference Orchestrator" (MCP Server)
*   **Frameworks:** Keras 3 / JAX / vLLM.
*   **Hardware:** Cloud TPU (Training/Batch) + GPU on Cloud Run (Serverless Inference).
*   **Concept:** An AI agent that intelligently routes workloads. It uses TPUs for heavy lifting and Cloud Run L4 GPUs for cost-effective, auto-scaling inference.
*   **MCP Tools:**
    *   `deploy_to_cloud_run`: Containerizes and deploys a model to Cloud Run GPU.
    *   `bench_latency`: Compares performance between TPU v5e and Cloud Run L4.
    *   `toggle_serverless`: Shifts traffic to Cloud Run when demand is low (scale-to-zero).

## 2. "Serverless vLLM on Cloud Run" Manager (MCP + ADK)
*   **Frameworks:** vLLM.
*   **Hardware:** GPU on Cloud Run (NVIDIA L4).
*   **Concept:** An automated manager for serverless vLLM deployments, optimizing for Cloud Run's specific constraints (cold starts, memory limits).
*   **MCP Tools:**
    *   `optimize_vllm_config`: Adjusts flags like `--gpu-memory-utilization` for L4 GPUs.
    *   `monitor_cold_start`: Analyzes logs to optimize model loading times.
    *   `scale_by_tokens`: ADK-driven scaling based on real-time throughput.

## 3. Keras 3 "One-Click" TPU-to-GPU Pipeline (ADK Agent)
*   **Frameworks:** Keras 3 (Multi-backend).
*   **Hardware:** Cloud TPU + GPU on Cloud Run.
*   **Concept:** Train once on TPU (JAX backend) and deploy anywhere (PyTorch/GPU backend on Cloud Run) with zero code changes.
*   **Workflow:**
    1.  Train on JAX/TPU.
    2.  Agent converts weights/model via MCP.
    3.  Push FastAPI + Keras 3 container to Cloud Run GPU.

## 4. The "TPU Performance Analyst" (MCP Server)
*   **Frameworks:** JAX / PyTorch on XLA.
*   **Hardware:** Cloud TPU.
*   **Concept:** Gives LLMs direct visibility into TPU hardware performance and XLA compilation.
*   **MCP Tools:**
    *   `inspect_hlo`: Summarizes XLA High-Level Optimizer graphs.
    *   `monitor_tpu_hbm`: Real-time TPU memory tracking.
    *   `profile_xla_step`: Identifies bottlenecks in the computation graph.

## 5. XLA "Cross-Hardware" Profiler (MCP Server)
*   **Frameworks:** JAX / PyTorch XLA.
*   **Hardware:** Cloud TPU vs. Cloud Run L4 GPU.
*   **Concept:** Compare how the same XLA graph behaves across different hardware backends.
*   **MCP Tools:**
    *   `get_xla_metadata`: Extracts compiled HLO from both environments.
    *   `compare_op_fusing`: Analyzes operation fusion efficiency on TPU vs. GPU.

---

## Technical Stack (Current Workspace)
*   **Language:** Python 3
*   **SDK:** `mcp` (FastMCP)
*   **Deployment Target:** Cloud Run (GPU enabled), Cloud TPU.
