# DevOps/SRE Model Garden Agent (MCP) 🛠️

An automated DevOps/SRE assistant that troubleshoots infrastructure and analyzes logs using **Gemma models** deployed directly from **Vertex AI Model Garden**.

## 🚀 Concept
This project replaces reliance on external Hugging Face downloads with **enterprise-grade, managed Gemma endpoints** on Google Cloud. It provides a secure and integrated way for AI agents to:
1.  **Analyze Logs:** Directly summarize and troubleshoot Google Cloud Logging entries.
2.  **Suggest Fixes:** Generate SRE-focused remediation plans based on incident descriptions.
3.  **Manage Resources:** Provision or manage model endpoints natively via the Vertex AI SDK.

## 🛠 MCP Tools

### `analyze_cloud_logging`
*   **Action:** Fetches recent logs from Cloud Logging and sends them to a Gemma Model Garden endpoint for summary.
*   **Integration:** Uses `google-cloud-logging` and `google-cloud-aiplatform`.

### `suggest_sre_remediation`
*   **Action:** Uses Gemma to propose structured remediation steps for specific error messages.

### `deploy_gemma_to_cloud_run`
*   **Action:** Generates the deployment command for Gemma on Cloud Run with NVIDIA L4 GPU support.
*   **Infrastructure:** Cloud Run (Serverless GPU) with a vLLM container.

## ⚡ Cloud Run GPU Deployment
You can deploy Gemma directly to Cloud Run for serverless inference.

1.  **Build and Push Image:**
    ```bash
    gcloud builds submit --tag gcr.io/$PROJECT_ID/gemma-vllm .
    ```

2.  **Deploy to Cloud Run (NVIDIA L4):**
    ```bash
    gcloud beta run deploy gemma-service \
      --image gcr.io/$PROJECT_ID/gemma-vllm \
      --gpu 1 \
      --gpu-type nvidia-l4 \
      --memory 32Gi \
      --cpu 8 \
      --region us-central1
    ```

## ⚡ Setup & Requirements

1.  **Enable APIs:**
    Ensure Vertex AI and Cloud Logging APIs are enabled in your project.

2.  **Deploy Gemma in Model Garden:**
    Go to [Vertex AI Model Garden](https://console.cloud.google.com/vertex-ai/model-garden), select a Gemma model (e.g., Gemma 2), and deploy it to an endpoint.

3.  **Configure Environment:**
    ```bash
    export GOOGLE_CLOUD_PROJECT="your-project-id"
    export GOOGLE_CLOUD_LOCATION="us-central1"
    export MODEL_GARDEN_ENDPOINT_ID="your-endpoint-id"
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the MCP server:**
    ```bash
    python server.py
    ```

## 📂 Structure
*   `server.py`: FastMCP server integrating Vertex AI and Cloud Logging.
*   `requirements.txt`: Python dependencies.
