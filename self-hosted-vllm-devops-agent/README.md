# Self-Hosted vLLM DevOps Agent (MCP Server)

This project provides an automated DevOps/SRE assistant that leverages **Gemma models self-hosted via vLLM on Cloud Run GPU**. It bridges Google Cloud Logging with a private inference endpoint to analyze infrastructure issues and suggest remediations.

## 🚀 Deployment Requirements

To deploy and run this project, you need to address two main components: the **Inference Stack** (vLLM on Cloud Run) and the **MCP Server** itself.

### 1. Infrastructure Requirements (The Inference Stack)
The MCP server expects a running vLLM instance. Your Cloud Run deployment for the model needs:
*   **Hardware:** NVIDIA L4 GPU (1 unit).
*   **Compute:** Minimum 4 vCPUs and 16GiB RAM.
*   **Execution Environment:** `gen2` (required for GPU and GCS FUSE).
*   **Storage:** A GCS Bucket containing the Gemma model weights (e.g., `gs://PROJECT_ID-bucket/gemma-2b-it/`).
*   **Networking:** Private Google Access must be enabled on the VPC subnet if using GCS FUSE.

### 2. Software & API Dependencies
The agent relies on several Google Cloud services and Python libraries:
*   **Libraries:** `mcp`, `fastmcp`, `google-cloud-logging`, `google-cloud-aiplatform`, and `requests`.
*   **Permissions:** The service account running the agent needs:
    *   `logging.logEntries.list` (to read logs).
    *   `aiplatform.models.list` (to list Vertex AI models).
    *   Access to the vLLM endpoint (either public with auth or via VPC).

### 3. Environment Variables
You must configure the following variables for the MCP server:
*   `GOOGLE_CLOUD_PROJECT`: Your GCP Project ID (defaults to `aisprint`).
*   `GOOGLE_CLOUD_LOCATION`: The region for Vertex AI (defaults to `us-east4`).
*   `VLLM_BASE_URL`: The URL of your Cloud Run vLLM service (e.g., `https://vllm-service-xyz.a.run.app`).
*   `MODEL_NAME`: The model identifier used by vLLM (defaults to `google/gemma-2b-it`).

## 🛠 Usage & Setup

### Step 1: Prepare Model Weights
Use the built-in tool `get_vertex_ai_model_copy_instructions` to move Gemma weights from Vertex AI Model Garden to your GCS bucket.

### Step 2: Deploy vLLM to Cloud Run
Run the `get_vllm_deployment_config` tool within the MCP server to generate the exact `gcloud` command for deployment.

### Step 3: Run the MCP Server
Install dependencies and run the server:
```bash
make install
export VLLM_BASE_URL="your-vllm-url"
make run
```

## 🧪 Testing
Run the included test suite to verify the tool registration:
```bash
make test
```
