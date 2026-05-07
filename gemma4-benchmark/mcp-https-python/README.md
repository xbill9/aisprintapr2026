# TPU vLLM Agent (MCP Server)

This is a Model Context Protocol (MCP) server designed to manage and query a self-hosted vLLM inference stack on Cloud TPUs.

## Features
- **TPU Management**: Create, list, describe, and delete TPU Queued Resources.
- **vLLM Interaction**: Query the self-hosted Gemma 4 model with streaming and performance stats.
- **Observability**: Fetch Docker logs, system logs, and Google Cloud Logging entries.
- **Cost Estimation**: Estimate TPU deployment costs.

## Deployment

The server is optimized for deployment on Google Cloud Run using the `streamable-http` transport.

### Automated Deployment
```bash
make deploy
```

### Manual Configuration
- **Host**: `0.0.0.0`
- **Port**: Configured via `PORT` environment variable (default: `8080`)
- **Transport**: `streamable-http`

## Environment Variables
- `GOOGLE_CLOUD_PROJECT`: GCP Project ID.
- `GOOGLE_CLOUD_ZONE`: GCP Zone for TPU resources.
- `GOOGLE_CLOUD_REGION`: GCP Region for TPU and Cloud Run.
- `MODEL_NAME`: The model being served by vLLM.

## Tech Stack
- Python 3.13
- FastMCP
- Google Cloud SDK (gcloud)
- OpenAI SDK (for vLLM interaction)
