# TPU vLLM DevOps Agent - Project Context

## Role
You are an expert TPU SRE and DevOps Engineer specialized in the **Gemma 4** ecosystem. Your goal is to manage the self-hosted inference stack and leverage it for infrastructure analysis.

## Core Infrastructure (Trillium / v6e)
- **TPU Version:** v6e (Trillium)
- **Topology:** `2x4` (8 chips) is the standard for Gemma 4 31B-it.
- **Software:** `vllm/vllm-tpu:nightly` container (v0.19.2+ required for Gemma 4 stability).
- **Model:** `google/gemma-4-31B-it`.
- **Runtime:** `v2-alpha-tpuv6e` for Flex-start / Queued Resources.

## Strategic Guidelines
- **Discovery:** Always prefer `get_vllm_endpoint` to verify connectivity before running analysis tools.
- **Provisioning:** Use `orchestrate_gemma4_stack` for turnkey deployment using Queued Resources.
- **Observability:** Use `check_tpu_utilization` and `get_vllm_metrics` to monitor Tensor Core and HBM pressure.
- **Validation:** Always run `validate_gemma4_deployment` after a new stack is provisioned.

## Technical Standards
- **vLLM API:** OpenAI-compatible endpoint at `/v1/chat/completions`.
- **Optimization Flags:**
  - `--tensor-parallel-size 8`
  - `--max-model-len 16384`
  - `--disable_chunked_mm_input`
  - `--max_num_batched_tokens 4096` (required for multimodal compatibility)
  - `--limit-mm-per-prompt '{"image":4,"audio":1}'` (JSON format required in nightly)
- **Tooling:** Enable `--enable-auto-tool-choice`, `--tool-call-parser gemma4`, and `--reasoning-parser gemma4`.

## Active Deployment
- **Resource ID:** `gemma4-vllm-stack`
- **Node ID:** `gemma4-vllm-stack-node`
- **IP Address:** `35.222.239.170`
- **Port:** `8000`
- **Zone:** `us-central1-a`
- **Status:** **🟢 ONLINE** (Initialized and ready for inference).

## Release Notes (v0.19.1+)
### Gemma 4 Bug Fixes
- **Streaming Tool Calls:** Fixed invalid JSON by stripping partial delimiters.
- **Tool Parser:** Fixed bare `null` values being incorrectly converted to `"null"`.
- **LoRA:** Fixed `Gemma4ForCasualLM` to ensure correct loading of LoRA adapters.
- **Token Repetition:** Resolved repetition issues via dynamic BOS injection.

## Flex-start VMs (Deep Dive)
Our stack leverages **Flex-start VMs** (via the `v2-alpha-tpuv6e` runtime) to maximize TPU availability and minimize costs.

### Key Characteristics
- **Dynamic Workload Scheduler (DWS):** Provisions resources from a secure pool, significantly increasing the probability of securing high-demand TPU v6e chips.
- **Wait-Time Mechanism:** Requests can wait up to 2 hours for resources to become available if capacity is currently full.
- **Execution Limit:** VMs have a maximum run duration of **7 days**. The stack must be configured with a `maxRunDuration` and a termination action (stop/delete).
- **Dense Placement:** Compute Engine attempts to place TPU nodes in close physical proximity to minimize network latency and hops.
- **Cost Efficiency:** Offers discounted pricing for vCPUs, memory, and TPU accelerators compared to standard on-demand rates.

### Constraints to Note
- **No Live Migration:** Flex-start VMs do not support live migration and will be stopped during host maintenance events.
- **Quota Requirements:** Requires sufficient **preemptible quota** for the specific TPU version and region.
- **No Reservations:** These instances **cannot** consume existing TPU reservations.

## Provisioning with Flex-start
To deploy the Gemma 4 stack using Flex-start, use the Queued Resources API.

### Core Command
```bash
gcloud alpha compute tpus queued-resources create [RESOURCE_ID] \
    --zone=us-central1-a \
    --accelerator-type=v6e-8 \
    --runtime-version=v2-alpha-tpuv6e \
    --node-id=[NODE_ID] \
    --provisioning-model=flex-start \
    --max-run-duration=24h
```

### Essential Flags
- `--provisioning-model=flex-start`: **Required** to enable the Dynamic Workload Scheduler.
- `--max-run-duration`: **Required** (Max: 7 days). Defines the hard limit for the TPU's lifetime.
- `--force`: (When deleting) Ensures both the queued request and the underlying VM are cleaned up.

### Lifecycle Management
1. **Queuing:** Status starts as `WAITING_FOR_RESOURCES`.
2. **Activation:** Becomes `ACTIVE` when chips are allocated.
3. **Automatic Cleanup:** The VM is deleted automatically at the `terminationTimestamp` (start time + `max-run-duration`).
4. **Manual Cleanup:** Always use `delete --force` to ensure quota is released.

## Key References
- [Requesting TPUs using Flex-start (Official)](https://docs.cloud.google.com/tpu/docs/request-using-flex-start)
- [About Flex-start VMs (Official)](https://docs.cloud.google.com/compute/docs/instances/about-flex-start-vms)
- [vLLM Gemma 4 Recipe (Official)](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html)

- [Cloud TPU v6e Guide](https://cloud.google.com/tpu/docs/v6e-trillium)
- [Cloud TPU Queued Resources](https://docs.cloud.google.com/tpu/docs/queued-resources)

https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM/Gemma4

## Step-by-Step Setup

### 1. Install LiteLLM Proxy
You need the [proxy] version of LiteLLM to handle the translation:
```bash
pip install 'litellm[proxy]'
```

### 2. Create a configuration file (`litellm_config.yaml`)
Create this file to map the Gemini model names used by the CLI to your TPU endpoint:
```yaml
model_list:
  - model_name: "gemma4-tpu"
    litellm_params:
      model: "openai/google/gemma-4-31B-it" # Tell LiteLLM it's an OpenAI-style endpoint
      api_base: "http://35.222.239.170:8000/v1" # Your TPU IP
      api_key: "none" # vLLM doesn't require a key by default
    router_settings:
      model_group_alias:
        # Map common Gemini model names to your TPU-hosted Gemma 4
        "gemini-2.0-flash": "gemma4-tpu"
        "gemini-2.0-flash-lite": "gemma4-tpu"
        "gemini-1.5-flash": "gemma4-tpu"
        "gemini-1.5-pro": "gemma4-tpu"
```
*Note: The IP `35.222.239.170` matches the Active Deployment in this workspace.*

### 3. Start the LiteLLM Proxy
Run this in a separate terminal (or in the background):
```bash
litellm --config litellm_config.yaml --port 4000
```

### 4. Configure Gemini CLI to use the Proxy
Set these environment variables in your shell (e.g., in `~/.bashrc` or `~/.zshrc`) to make it permanent:
```bash
# Point the CLI to your local LiteLLM proxy
export GOOGLE_GEMINI_BASE_URL="http://localhost:4000"

# Set the default model globally
export GEMINI_MODEL="google/gemma-4-31B-it"

# The CLI requires a key even if the proxy ignores it
export GEMINI_API_KEY="local-proxy-token"
```

**Why this works:**
* **API Translation:** When you run `gemini "Hello"`, the CLI sends a request to `localhost:4000` in Google format. LiteLLM translates this to the OpenAI format and forwards it to your TPU.
* **Tool Calling Compatibility:** Because we deployed your Gemma 4 stack with `--tool-call-parser gemma4`, the model's reasoning and tool outputs will be perfectly understood by the Gemini CLI when it tries to run shell commands or edit files.

Now, every time you run `gemini`, it will be powered by your private TPU v6e cluster.

