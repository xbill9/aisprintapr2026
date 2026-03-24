# Base image with vLLM pre-installed (example using official image)
FROM vllm/vllm-openai:latest

# Additional tools for management
RUN pip install --no-cache-dir mcp google-cloud-logging

# Configuration scripts
COPY server.py /app/manager_server.py

# entrypoint can be vllm itself, with the manager running as a sidecar or pre-hook
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
