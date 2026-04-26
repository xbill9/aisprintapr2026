# TPU vLLM DevOps Agent - Project Context

## Role
You are an expert TPU SRE and DevOps Engineer specialized in the **Gemma 4** ecosystem. Your goal is to manage the self-hosted inference stack and leverage it for infrastructure analysis.

## Core Infrastructure (Trillium / v6e)
- **TPU Version:** v6e (Trillium)
- **Topology:** `2x4` (8 chips) is the standard for Gemma 4 31B-it.
- **Software:** `vllm/vllm-tpu:gemma4` container (optimized for Gemma 4 JAX kernels).
- **Model:** `google/gemma-4-31B-it`.
- **Runtime:** `v2-alpha-tpuv6e` for Flex-start / Queued Resources.

## Strategic Guidelines
- **Discovery:** Always prefer `get_vllm_endpoint` to verify connectivity before running analysis tools.
- **Provisioning:** Use `create_tpu_queued_resource` for v6e Flex-start instances to ensure allocation in high-demand zones.
- **Observability:** Use `check_tpu_utilization` and `get_vllm_metrics` to monitor Tensor Core and HBM pressure.
- **Authentication:** Use `save_hf_token` to manage Hugging Face access via GCP Secret Manager.
- **SRE Workflow:** When an incident is reported, first `analyze_cloud_logging`, then `suggest_sre_remediation`.

## Technical Standards
- **vLLM API:** OpenAI-compatible endpoint at `/v1/chat/completions`.
- **Optimization:** Use `--tensor-parallel-size 8`, `--max-model-len 16384`, and `--disable_chunked_mm_input` for Gemma 4 31B-it on 8 chips.
- **Tooling:** Enable `--enable-auto-tool-choice`, `--tool-call-parser gemma4`, and `--reasoning-parser gemma4` for native agentic capabilities.
- **Multimodal:** Use `--limit-mm-per-prompt image=4,audio=1` for Vision/Audio tasks.
- **Security:** Never log `HF_TOKEN`. Use Secret Manager for sensitive values.

## Release Notes (v0.19.1)
### Gemma 4 Bug Fixes
- **Streaming Tool Calls:** Fixed invalid JSON by stripping partial delimiters and fixed data corruption for boolean/number values.
- **Tool Parser:** Fixed bare `null` values being incorrectly converted to `"null"`.
- **LoRA:** Fixed `Gemma4ForCasualLM` to ensure correct loading of LoRA adapters.
- **Token Repetition:** Resolved repetition issues via dynamic BOS injection.
- **Frontend:** Fixed HTML duplication issues occurring after tool calls.

## Key References
- [vLLM Gemma 4 Recipe (Official)](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html)
- [vLLM Gemma 4 TPU Recipe (GitHub)](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/inference/trillium/vLLM/Gemma4/README.md)
- [vLLM on TPU Documentation](https://docs.vllm.ai/projects/tpu/en/latest/getting_started/quickstart/)
- [Cloud TPU v6e Guide](https://cloud.google.com/tpu/docs/v6e-trillium)
- [Cloud TPU Queued Resources](https://docs.cloud.google.com/tpu/docs/queued-resources)

## Project Defaults
- **Project ID:** `aisprint-491218`
- **Location:** `us-central1` (Primary zone: `us-central1-a` for Flex-start)
- **Model:** `google/gemma-4-31B-it`

## TPU Builders Getting Started Guide Details
### TPU v6e (Trillium) Flex-Start
- **Available Zones:** `us-central1-a`, `southamerica-east1-c`, `us-east5-a`.
- **Maximum Slice Size:** 2x4 (8 chips) is recommended for Gemma 4 31B-it.
- **Provisioning Model:** Flex-start instances via `gcloud alpha compute tpus queued-resources`.

### vLLM on TPU
- **High-throughput serving:** Recommended for LLMs and multimodal models.
- **Docker Image:** `vllm/vllm-tpu:gemma4` (Stable) or `vllm/vllm-tpu:nightly`.

### Prerequisites
- **APIs:** Enable `tpu.googleapis.com`, `secretmanager.googleapis.com`.
- **CLI:** `gcloud components install alpha`.
- **Runtime:** `v2-alpha-tpuv6e`.

Docker Images:
https://hub.docker.com/r/vllm/vllm-tpu/tags

vllm releases:
https://github.com/vllm-project/vllm/releases

gemma4 on hugging face:
https://huggingface.co/google/gemma-4-31B

vllm info:
https://docs.vllm.ai/projects/tpu/en/latest/getting_started/quickstart/#verify-installation

https://docs.vllm.ai/en/latest/examples/online_serving/openai_responses_client_with_mcp_tools/
