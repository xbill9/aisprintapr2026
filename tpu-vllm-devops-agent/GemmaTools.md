# Gemma 4 DevOps Agent Tools

This document provides a summary of the MCP tools available in the `server.py` file for the Queued TPU vLLM Agent.

## Deployment & Configuration

-   **`get_vllm_deployment_config`**: Generates the gcloud command for a single-host TPU v6e vLLM deployment.
-   **`get_vllm_tpu_deployment_config`**: Generates a GKE manifest for a TPU v6e vLLM deployment.
-   **`orchestrate_gemma4_stack`**: A high-level tool that orchestrates a seamless, turnkey deployment of the Gemma 4 stack. It handles saving the HF token, validating quota, and initiating the Queued Resource creation.
-   **`deploy_queued_vllm`**: Deploys vLLM using Queued Resources for Flex-start allocation.
-   **`create_tpu_queued_resource`**: Creates a TPU Queued Resource (Flex-start) with a specified configuration.
-   **`destroy_queued_resource`**: Safely deletes a Queued Resource and its associated node.

## Monitoring & Status

-   **`get_system_status`**: Provides a high-level status dashboard of the Queued Resource states in `us-central1-a`, including TPU quota and vLLM health.
-   **`list_queued_resources`**: Lists all Queued Resources in a specific zone.
-   **`describe_queued_resource`**: Provides detailed information about a specific Queued Resource.
-   **`get_reservation_status`**: Checks the lifecycle state and expiry time of a Queued Resource.
-   **`check_tpu_availability`**: A simple check to see if a Queued Resource has reached the `ACTIVE` state.
-   **`get_vllm_endpoint`**: A discovery tool to verify connectivity and return the active vLLM service URL.
-   **`validate_gemma4_deployment`**: Performs a comprehensive sanity check on the Gemma 4 deployment, including connectivity, configuration flags, and a logic test.

## Performance & Benchmarking

-   **`estimate_deployment_cost`**: Estimates the cost of a TPU deployment.
-   **`check_tpu_utilization`**: Monitors Tensor Core and HBM pressure on the TPU VM.
-   **`get_vllm_metrics`**: Fetches real-time Prometheus metrics from the active vLLM service.
-   **`get_vllm_model_stats`**: Aggregates model-specific statistics from the vLLM server.
-   **`run_vllm_benchmark`**: Runs vLLM's internal benchmark tool inside the container on the TPU VM.
-   **`run_vllm_internal_benchmark`**: Runs vLLM's internal benchmark tool inside the container on the TPU VM.
-   **`run_external_load_test`**: Performs an external load test against the active vLLM endpoint.

## Interaction & Diagnostics

-   **`query_queued_gemma4`**: Queries the model hosted on the active Queued Resource.
-   **`query_vllm_with_metrics`**: Queries the model and provides streaming-based performance metrics.
-   **`verify_model_health`**: Performs a deep health check by querying the model with a simple prompt.
-   **`fetch_tpu_vm_logs`**: Fetches specific logs from a TPU VM (`vllm`, `startup`, or `system`).
-   **`grep_tpu_logs`**: Searches for a pattern in both startup and container logs on the TPU VM.
-   **`fetch_queued_node_logs`**: Fetches logs by identifying the node linked to a Queued Resource.
-   **`analyze_cloud_logging`**: Searches Cloud Logging for TPU-related errors and lifecycle events.

## Security

-   **`save_hf_token`**: Saves the Hugging Face token to GCP Secret Manager.
-   **`get_secret`**: Retrieves a secret from Secret Manager.
