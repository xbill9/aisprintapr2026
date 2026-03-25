# Configuration - Update these or override via environment variables
PROJECT_ID ?= aisprint-491218
REGION ?= us-east4
BUCKET_NAME ?= $(PROJECT_ID)-bucket
SERVICE_NAME ?= vllm-gemma-2b-it
MODEL_PATH ?= gemma-2b-it

install:
	pip install -r requirements.txt

run:
	python server.py

test:
	python test_agent.py

clean:
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Recommended deployment settings for vLLM on Cloud Run with NVIDIA L4 GPU
# Requires 'gcloud beta' component and 'gen2' execution environment
deploy-vllm:
	@echo "🚀 Deploying vLLM to Cloud Run ($(SERVICE_NAME)) in $(REGION)..."
	gcloud beta run deploy $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--image=vllm/vllm-openai:latest \
		--gpu=1 \
		--gpu-type=nvidia-l4 \
		--no-gpu-zonal-redundancy \
		--no-cpu-throttling \
		--concurrency=4 \
		--timeout=3600 \
		--startup-probe=timeoutSeconds=60,periodSeconds=60,failureThreshold=10,initialDelaySeconds=180,httpGet.port=8000,httpGet.path=/health \
		--max-instances=1 \
		--min-instances=0 \
		--port=8000 \
		--memory=16Gi \
		--cpu=4 \
		--execution-environment=gen2 \
		--add-volume=name=model-volume,type=cloud-storage,bucket=$(BUCKET_NAME),readonly=true \
		--add-volume-mount=volume=model-volume,mount-path=/mnt/models \
		--args=--model=/mnt/models/$(MODEL_PATH),--max-model-len=4096,--trust-remote-code,--gpu-memory-utilization=0.9,--host=0.0.0.0 \
		--no-allow-unauthenticated \
		--region=$(REGION)

# Deploy the vLLM inference stack
deploy: deploy-vllm

# Destroy the vLLM inference stack
destroy-vllm:
	@echo "🗑️  Destroying Cloud Run service $(SERVICE_NAME) in $(REGION)..."
	gcloud run services delete $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--region=$(REGION) \
		--quiet

destroy: destroy-vllm

.PHONY: install run test clean deploy-vllm deploy destroy-vllm destroy
