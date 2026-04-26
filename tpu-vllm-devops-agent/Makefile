# Configuration - Update these or override via environment variables
PROJECT_ID ?= aisprint-491218
REGION ?= us-central1
ZONE ?= $(REGION)-a
BUCKET_NAME ?= $(PROJECT_ID)-bucket
SERVICE_NAME ?= vllm-gemma-4-tpu
MODEL_PATH ?= google/gemma-4-31B-it

install:
	pip install -r requirements.txt

run:
	python server.py

test:
	python test_agent.py

lint:
	ruff check .
	ruff format --check .
	mypy .

clean:
	rm -rf __pycache__
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.log" -exec rm -rf {} +

# Recommended deployment settings for vLLM on Cloud TPU v6e (Trillium)
deploy-tpu:
	@echo "🚀 Creating TPU v6e instance ($(SERVICE_NAME)) in $(ZONE)..."
	gcloud alpha compute tpus tpu-vm create $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE) \
		--type=v6e \
		--topology=2x4 \
		--version=v2-alpha-tpuv6e

# Note: Running the container requires SSHing into the TPU VM.
# This target provides the command for reference.
run-container:
	@echo "🛠 Run this command on the TPU VM to start vLLM:"
	@echo "sudo docker run -t --rm --name vllm-gemma4 --privileged --net=host \\"
	@echo "    -v /dev/shm:/dev/shm --shm-size 10gb \\"
	@echo "    -e HF_TOKEN=\$$HF_TOKEN \\"
	@echo "    vllm/vllm-tpu:gemma4 \\"
	@echo "    vllm serve $(MODEL_PATH) \\"
	@echo "    --max-model-len 16384 \\"
	@echo "    --tensor-parallel-size 8 \\"
	@echo "    --disable_chunked_mm_input \\"
	@echo "    --enable-auto-tool-choice \\"
	@echo "    --tool-call-parser gemma4"

# Deploy the vLLM inference stack
deploy: deploy-tpu

# Destroy the vLLM inference stack
destroy-tpu:
	@echo "🗑️  Destroying TPU VM $(SERVICE_NAME) in $(ZONE)..."
	gcloud compute tpus tpu-vm delete $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE) \
		--quiet

destroy: destroy-tpu

# Check status of the TPU VM
status:
	@echo "🔍 Checking status of TPU VM $(SERVICE_NAME) in $(ZONE)..."
	gcloud compute tpus tpu-vm describe $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE)

# Get the IP address of the deployed TPU VM
endpoint:
	@gcloud compute tpus tpu-vm describe $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE) \
		--format='value(networkEndpoints[0].ipAddress)'

# Run the performance benchmarking suite
benchmark:
	@echo "📊 Discovering endpoint for $(SERVICE_NAME)..."
	@IP=$$(gcloud compute tpus tpu-vm describe $(SERVICE_NAME) --project=$(PROJECT_ID) --zone=$(ZONE) --format='value(networkEndpoints[0].accessConfig.externalIp)'); \
	if [ -z "$$IP" ]; then IP=$$(gcloud compute tpus tpu-vm describe $(SERVICE_NAME) --project=$(PROJECT_ID) --zone=$(ZONE) --format='value(networkEndpoints[0].ipAddress)'); fi; \
	if [ -z "$$IP" ]; then echo "❌ Error: Could not find IP for $(SERVICE_NAME)"; exit 1; fi; \
	echo "🚀 Starting benchmark against http://$$IP:8000"; \
	python benchmarking_suite.py --url http://$$IP:8000 --model $(MODEL_PATH)

PROMPT ?= "What is Site Reliability Engineering?"

# Query the vLLM model via the TPU VM endpoint (assuming it's running)
query:
	@echo "🔍 Querying $(SERVICE_NAME) with prompt: \"$(PROMPT)\"..."
	@IP=$$(gcloud compute tpus tpu-vm describe $(SERVICE_NAME) --project=$(PROJECT_ID) --zone=$(ZONE) --format='value(networkEndpoints[0].accessConfig.externalIp)'); \
	if [ -z "$$IP" ]; then IP=$$(gcloud compute tpus tpu-vm describe $(SERVICE_NAME) --project=$(PROJECT_ID) --zone=$(ZONE) --format='value(networkEndpoints[0].ipAddress)'); fi; \
	curl -s -X POST "http://$$IP:8000/v1/completions" \
		-H "Content-Type: application/json" \
		-d '{"model": "$(MODEL_PATH)", "prompt": "$(PROMPT)", "max_tokens": 128, "temperature": 0.2}' | python3 -m json.tool

.PHONY: install run test clean deploy-tpu deploy destroy-tpu destroy status endpoint query
