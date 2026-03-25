# Self-Hosted LLM Gateway (MCP) 🚀

This MCP server acts as a proxy between the Gemini CLI (or any MCP client) and self-hosted open-weight models (like Llama 3, Mixtral, or Gemma 2) running on serverless infrastructure like **Cloud Run (NVIDIA L4 GPU)**.

## 🚀 Concept
- **Open-Weight Flexibility:** Use specialized models for local tasks while leveraging Gemini for orchestration.
- **Serverless Scaling:** Route requests to Cloud Run endpoints that scale-to-zero when not in use.
- **Unified Interface:** Standardize tool calls across different self-hosted LLM backends (vLLM, LiteLLM, Ollama).

## 🛠 MCP Tools

### `query_local_llm`
*   **Action:** Routes a prompt to the OpenAI-compatible `/chat/completions` endpoint of the gateway.
*   **Parameters:** `prompt`, `model` (optional), `max_tokens`.

### `switch_model_backend`
*   **Action:** Dynamically updates the target model name for future requests.

### `get_gateway_metrics`
*   **Action:** Retrieves status, latency, and GPU utilization metrics from the backend.

## 📂 Structure
*   `server.py`: FastMCP server for LLM orchestration.
*   `requirements.txt`: Python dependencies (`mcp`, `httpx`).
*   `test_gateway.py`: Validation script.

## ⚡ How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure (Optional):**
    Set environment variables for your self-hosted backend:
    ```bash
    export LLM_GATEWAY_URL="https://your-cloud-run-url.a.run.app/v1"
    export LLM_GATEWAY_API_KEY="your-secret-key"
    ```

3.  **Run the MCP server:**
    ```bash
    python server.py
    ```

4.  **Run tests:**
    ```bash
    python test_gateway.py
    ```
