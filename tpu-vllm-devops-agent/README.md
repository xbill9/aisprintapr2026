# TPU vLLM DevOps Agent (MCP Server)

This project provides an automated DevOps/SRE assistant that leverages **Gemma 4 models self-hosted via vLLM on Cloud TPUs**. It bridges Google Cloud Logging with a private inference endpoint to analyze infrastructure issues and suggest remediations.

## 🚀 Deployment Requirements

To deploy and run this project, you need to address two main components: the **Inference Stack** (vLLM on TPU v6e) and the **MCP Server** itself.

### 1. Infrastructure Requirements (The Inference Stack)
The MCP server expects a running vLLM instance. Your TPU deployment for the model needs:
*   **Hardware:** Cloud TPU v6e (Trillium) with topology `2x4` (8 chips).
*   **Software:** `vllm/vllm-tpu:gemma4` specialized container.
*   **Model:** `google/gemma-4-31B-it` (Hugging Face ID).
*   **Networking:** Private Google Access must be enabled for internal connectivity, or direct internet access for Hugging Face downloads.

### 2. Software & API Dependencies
The agent relies on several Google Cloud services and Python libraries:
*   **Libraries:** `mcp`, `fastmcp`, `google-cloud-logging`, `google-cloud-secret-manager`, and `requests`.
*   **Permissions:** The service account running the agent needs:
    *   `logging.logEntries.list` (to read logs).
    *   `tpu.nodes.get` and `tpu.nodes.list` (for discovery).
    *   `secretmanager.versions.access` (if using Hugging Face tokens).
    *   Access to the vLLM endpoint.

### 3. Environment Variables
You can configure the following variables for the MCP server:
*   `GOOGLE_CLOUD_PROJECT`: Your GCP Project ID (defaults to `aisprint-491218`).
*   `GOOGLE_CLOUD_LOCATION`: The region for TPUs (defaults to `us-central1`).
*   `VLLM_BASE_URL`: The URL of your TPU VM vLLM service. **If omitted, the agent will attempt to auto-discover it using `gcloud` (TPU VM).**
*   `MODEL_NAME`: The model identifier used by vLLM (defaults to `google/gemma-4-31B-it`).

## 🛠 Usage & Setup

### Step 1: Deploy vLLM to TPU
Run the `get_vllm_deployment_config` tool within the MCP server to generate the exact `gcloud` command for TPU VM creation, or use the provided `Makefile`:
```bash
make deploy
```

### Step 2: Start vLLM on the TPU VM
SSH into the TPU VM and run the container (see `DEPLOY.md` or `make run-container` for details). Ensure you have set your `HF_TOKEN`.

### Step 3: Run the MCP Server
Install dependencies and run the server locally:
```bash
make install
# Optional: export VLLM_BASE_URL="http://<TPU_IP>:8000"
make run
```

## 🛠 Available Tools

The following tools are available via the MCP server:

### Infrastructure Management
*   **`deploy_vllm`**: Creates a new TPU v6e VM instance.
*   **`destroy_vllm`**: Deletes the TPU VM instance.
*   **`status_vllm`**: Checks the health and status of the TPU instance.
*   **`get_vllm_endpoint`**: Returns the current active vLLM endpoint URL (discovers it if needed).
*   **`get_vllm_deployment_config`**: Generates commands for TPU v6e deployment.
*   **`get_vllm_tpu_deployment_config`**: Generates GKE manifests for TPU v6e.
*   **`create_tpu_queued_resource`**: Provisions a TPU v6e via Queued Resources (Flex-start).
*   **`is_tpu_ready`**: Checks if the queued resource is ACTIVE.
*   **`launch_vllm_container`**: Remotely starts the vLLM Docker container on the TPU VM.
*   **`is_vllm_ready`**: Verifies if the vLLM server is responding via SSH.

### Observability & Metrics
*   **`check_tpu_utilization`**: Checks real-time Tensor Core and HBM usage on the TPU.
*   **`get_vllm_metrics`**: Scrapes vLLM Prometheus metrics for traffic analysis.
*   **`get_vllm_container_logs`**: Fetches latest logs from the vLLM container.
*   **`verify_model_health`**: Performs an end-to-end smoke test on the model endpoint.

### AI & Operations
*   **`analyze_cloud_logging`**: Summarizes error logs using self-hosted vLLM on TPU.
*   **`suggest_sre_remediation`**: Provides 3-step plans for SRE incidents using Gemma 4.
*   **`query_vllm_chat`**: Sends direct chat messages to the self-hosted Gemma 4 model.
*   **`save_hf_token`**: Securely saves a Hugging Face API token to GCP Secret Manager.

## 📦 Resources
The server exposes the following MCP resources:
*   **`config://vllm-deployment-template`**: A script snippet for TPU v6e vLLM deployment.

## 🌟 Grand Demo
A standalone demo script is included to showcase the agent's capabilities with Gemma 4:
```bash
python demo_launcher.py
```

## 🛠 Makefile Helpers
The included `Makefile` provides several shortcuts:
*   `make install`: Installs Python dependencies.
*   `make run`: Starts the MCP server.
*   `make deploy`: Creates the TPU v6e instance.
*   `make run-container`: Displays the command to start vLLM on the TPU VM.
*   `make destroy`: Removes the TPU VM.
*   `make status`: Checks the status of the TPU VM.
*   `make query PROMPT="your prompt"`: Queries the vLLM model directly via `curl`.
*   `make test`: Runs the test suite.
