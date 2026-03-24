# AutoResearch Serverless Manager 🧠

This prototype implements the **Karpathy "AutoResearch" loop** on a **Google Cloud Serverless Stack** (Cloud Run + NVIDIA L4 GPUs), as featured in [Karl Weinmeister's guide](https://medium.com/google-cloud/run-karpathys-autoresearch-on-a-google-serverless-stack-for-2-hour-210fc8e2a829).

## 🚀 Concept
An AI agent (Gemini CLI) autonomously:
1.  **Edits** model code (`train.py`) to propose improvements.
2.  **Trains** the model using a serverless **Cloud Run Job** with an **NVIDIA L4 GPU**.
3.  **Evaluates** performance and either commits or reverts the changes.
4.  **Chains** research cycles using **Cloud Workflows** for multi-hour experimentation at ~$2/hour.

---

## 🛠 MCP Tools

### `submit_research_job`
*   **Action:** Triggers a Cloud Run Job (L4 GPU) for a single research cycle.
*   **Hardware:** Serverless L4 GPU (24GB VRAM).

### `monitor_research_workflow`
*   **Action:** Tracks the status of the entire research pipeline (Workflows).

### `analyze_research_costs`
*   **Action:** Estimates the cost per experiment, targeting the **$2/hour** budget.

### `get_latest_improvement`
*   **Action:** Fetches the best-performing code and artifacts from Google Cloud Storage.

---

## ⚡ Integration with TPU Sprint
While this specific architecture leverages **L4 GPUs** for serverless efficiency, the **research topics** are ideally suited for the TPU Sprint:
*   **XLA Fusion Optimization:** AutoResearch can experiment with `XLA_FLAGS` to find optimal kernel fusion.
*   **Keras 3 Backend Benchmarking:** AutoResearch can automatically test the same model across JAX/Torch backends.
*   **Sharding Search:** AutoResearch can find the optimal JAX sharding strategy for a given model architecture.

---

## 📂 Structure
*   `server.py`: FastMCP server for research orchestration.
*   `requirements.txt`: GCP SDK dependencies.
*   `test_autoresearch.py`: Validation script for the research tools.
