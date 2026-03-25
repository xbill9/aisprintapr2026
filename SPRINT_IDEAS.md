# TPU Sprint: MCP & ADK Project Ideas

This document outlines high-impact project ideas for the TPU Sprint, focusing on **JAX/Keras/vLLM/PyTorch on XLA/TPU** and **GPU on Cloud Run**, integrated via the **Model Context Protocol (MCP)** and **Agent Development Kit (ADK)**.

---

## 1. The "Hybrid Inference Orchestrator" (MCP Server)
*   **Difficulty:** Moderate (Requires Cloud Run v2 SDK integration).
*   **Frameworks:** Keras 3 / JAX / vLLM.
*   **Hardware:** Cloud TPU (Training/Batch) + GPU on Cloud Run (Serverless Inference).
*   **Concept:** An AI agent that intelligently routes workloads. It uses TPUs for heavy lifting and Cloud Run L4 GPUs for cost-effective, auto-scaling inference.
*   **MCP Tools:**
    *   `deploy_to_cloud_run`: Containerizes and deploys a model to Cloud Run GPU.
    *   `bench_latency`: Compares performance between TPU v5e and Cloud Run L4.
    *   `toggle_serverless`: Shifts traffic to Cloud Run when demand is low (scale-to-zero).

## 2. "Serverless vLLM on Cloud Run" Manager (MCP + ADK)
*   **Difficulty:** Moderate (Requires optimization of vLLM flags for L4 GPUs).
*   **Frameworks:** vLLM.
*   **Hardware:** GPU on Cloud Run (NVIDIA L4).
*   **Concept:** An automated manager for serverless vLLM deployments, optimizing for Cloud Run's specific constraints (cold starts, memory limits).
*   **MCP Tools:**
    *   `optimize_vllm_config`: Adjusts flags like `--gpu-memory-utilization` for L4 GPUs.
    *   `monitor_cold_start`: Analyzes logs to optimize model loading times.
    *   `scale_by_tokens`: ADK-driven scaling based on real-time throughput.

## 3. Keras 3 "One-Click" TPU-to-GPU Pipeline (ADK Agent)
*   **Difficulty:** Hard (Ensuring cross-backend weight parity is complex).
*   **Frameworks:** Keras 3 (Multi-backend).
*   **Hardware:** Cloud TPU + GPU on Cloud Run.
*   **Concept:** Train once on TPU (JAX backend) and deploy anywhere (PyTorch/GPU backend on Cloud Run) with zero code changes.
*   **Workflow:**
    1.  Train on JAX/TPU.
    2.  Agent converts weights/model via MCP.
    3.  Push FastAPI + Keras 3 container to Cloud Run GPU.

## 4. The "TPU Performance Analyst" (MCP Server)
*   **Difficulty:** Easy (Exposes built-in JAX HLO/HBM metrics).
*   **Frameworks:** JAX / PyTorch on XLA.
*   **Hardware:** Cloud TPU.
*   **Concept:** Gives LLMs direct visibility into TPU hardware performance and XLA compilation.
*   **MCP Tools:**
    *   `inspect_hlo`: Summarizes XLA High-Level Optimizer graphs.
    *   `monitor_tpu_hbm`: Real-time TPU memory tracking.
    *   `profile_xla_step`: Identifies bottlenecks in the computation graph.

## 5. XLA "Cross-Hardware" Profiler (MCP Server)
*   **Difficulty:** Moderate (Requires comparative analysis of HLO metadata).
*   **Frameworks:** JAX / PyTorch XLA.
*   **Hardware:** Cloud TPU vs. Cloud Run L4 GPU.
*   **Concept:** Compare how the same XLA graph behaves across different hardware backends.
*   **MCP Tools:**
    *   `get_xla_metadata`: Extracts compiled HLO from both environments.
    *   `compare_op_fusing`: Analyzes operation fusion efficiency on TPU vs. GPU.

## 6. "AutoResearch Serverless Manager" (MCP + ADK)
*   **Difficulty:** Hard (Autonomous loops require complex orchestration).
*   **Frameworks:** JAX / Keras 3 / PyTorch.
*   **Hardware:** Cloud Run (NVIDIA L4 GPU) + Cloud Workflows.
*   **Concept:** Implements the Karpathy "AutoResearch" loop on a Google Cloud Serverless Stack. An AI agent autonomously edits, trains on serverless GPUs, evaluates, and iterates on model improvements.
*   **MCP Tools:**
    *   `submit_research_job`: Triggers a Cloud Run Job (L4 GPU) for a single research cycle.
    *   `monitor_research_workflow`: Tracks the status of the entire research pipeline.
    *   `analyze_research_costs`: Estimates cost per experiment (targeting ~$2/hour).
    *   `get_latest_improvement`: Fetches the best-performing code and artifacts.

## 7. "Self-Hosted LLM Gateway" (MCP Server)
*   **Difficulty:** Easy (Direct async proxy logic).
*   **Frameworks:** LiteLLM / vLLM / FastAPI.
*   **Hardware:** Cloud Run (NVIDIA L4 GPU).
*   **Concept:** A proxy that bridges Gemini CLI to self-hosted open-weight models (like Llama 3, Mixtral, or Gemma 2) running on Cloud Run. This allows the CLI to leverage local or specialized models for tasks like code analysis, PII scrubbing, or latency-sensitive sub-processes.
*   **MCP Tools:**
    *   `query_local_llm`: Routes prompts to the self-hosted vLLM/LiteLLM endpoint.
    *   `switch_model_backend`: Dynamically swaps the underlying model (e.g., from Llama 3 8B to 70B).
    *   `get_gateway_metrics`: Monitors latency, token usage, and GPU utilization of the proxy.

## 8. "DevOps/SRE Model Garden Agent" (ADK + MCP)
*   **Difficulty:** Easy (Uses Vertex AI high-level SDK).
*   **Frameworks:** Vertex AI SDK / Gemma / LangChain.
*   **Hardware:** Vertex AI Model Garden (Managed Gemma Endpoints).
*   **Concept:** An automated DevOps/SRE assistant that troubleshoots infrastructure and analyzes logs. Unlike standard agents that pull from Hugging Face, this project leverages **Gemma models deployed directly from Vertex AI Model Garden**, ensuring enterprise-grade security and integration with Google Cloud's operations suite.
*   **MCP Tools:**
    *   `analyze_cloud_logging`: Uses Gemma to parse and summarize error logs from Cloud Logging.
    *   `suggest_sre_remediation`: Proposes fixes for detected incidents based on historical runbooks.
    *   `deploy_model_garden_endpoint`: Provisions a new Gemma instance via Model Garden for dedicated troubleshooting tasks.

## 9. "Self-Hosted vLLM DevOps Agent" (ADK + MCP)
*   **Difficulty:** Easy (High-level log parsing and vLLM queries).
*   **Frameworks:** vLLM / Gemma / LangChain.
*   **Hardware:** Cloud Run (NVIDIA L4 GPU).
*   **Concept:** An automated DevOps/SRE assistant that troubleshoots infrastructure and analyzes logs. Unlike the Model Garden version, this project leverages **Gemma models self-hosted via vLLM on Cloud Run**, offering more control over the inference stack and lower costs for high-throughput DevOps tasks.
*   **MCP Tools:**
    *   `analyze_cloud_logging`: Uses self-hosted Gemma to parse and summarize error logs from Cloud Logging.
    *   `suggest_sre_remediation`: Proposes fixes for detected incidents.
    *   `get_vllm_deployment_config`: Generates the configuration to scale or deploy the Cloud Run vLLM service.

---

## Technical Stack (Current Workspace)
*   **Language:** Python 3
*   **SDK:** `mcp` (FastMCP)
*   **Deployment Target:** Cloud Run (GPU enabled), Cloud TPU.
