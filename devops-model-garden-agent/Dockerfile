# Use an official vLLM image as the base
FROM vllm/vllm-openai:latest

# Environment variables to point to the Gemma model from Model Garden (GCS path)
# Note: You must grant the Cloud Run service account access to the GCS bucket.
ENV MODEL_ID="google/gemma-2-9b-it" 
ENV PORT=8080

# Cloud Run expects the container to listen on $PORT
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model", "$MODEL_ID", "--port", "8080", "--max-model-len", "8192"]
