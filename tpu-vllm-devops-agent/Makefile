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

.PHONY: install run test clean deploy-tpu deploy destroy-tpu destroy status endpoint query help

help:
	@echo "Available MCP tools in server.py:"
	@echo ""
	@echo "  get_vllm_deployment_config       - Generates the gcloud command for a single-host TPU v6e vLLM deployment."
	@echo "  get_vllm_tpu_deployment_config   - Generates a GKE manifest for a TPU v6e vLLM deployment."
	@echo "  list_queued_resources            - Lists all Queued Resources in a specific zone."
	@echo "  describe_queued_resource         - Provides detailed information about a specific Queued Resource."
	@echo "  get_reservation_status           - Checks the lifecycle state and expiry time of a Queued Resource."
	@echo "  check_tpu_availability         - Simple check to see if a Queued Resource has reached ACTIVE state."
	@echo "  estimate_deployment_cost         - Estimates the cost of a TPU deployment."
	@echo "  get_system_status                - Status dashboard prioritizing Queued Resource states in us-central1-a."
	@echo "  orchestrate_gemma4_stack         - Seamless turnkey deployment of the Gemma 4 stack."
	@echo "  get_vllm_endpoint                - Discovery tool to verify connectivity and return the active vLLM service URL."
	@echo "  deploy_queued_vllm               - Deploys vLLM strictly using Queued Resources for Flex-start allocation."
	@echo "  create_tpu_queued_resource     - Creates a TPU Queued Resource (Flex-start) with the specified configuration."
	@echo "  check_tpu_utilization            - Monitors Tensor Core and HBM pressure on the TPU VM."
	@echo "  get_vllm_metrics                 - Fetches real-time Prometheus metrics from the active vLLM service."
	@echo "  validate_gemma4_deployment       - Performs a comprehensive sanity check on the Gemma 4 deployment."
	@echo "  query_queued_gemma4              - Queries the model hosted on the active Queued Resource."
	@echo "  run_vllm_benchmark               - Runs vLLM's internal benchmark tool inside the container on the TPU VM."
	@echo "  run_vllm_internal_benchmark    - Runs vLLM's internal benchmark tool inside the container on the TPU VM."
	@echo "  run_external_load_test         - Performs an external load test against the active vLLM endpoint."
	@echo "  get_gemma4_full_report           - Generates a deep technical report of the Gemma 4 deployment."
	@echo "  fetch_tpu_vm_logs                - Fetches specific logs from a TPU VM."
	@echo "  grep_tpu_logs                    - Searches for a pattern in both startup and container logs on the TPU VM."
	@echo "  destroy_queued_resource          - Safely deletes a Queued Resource and its associated node."
	@echo "  fetch_queued_node_logs           - Fetches logs by identifying the node linked to a Queued Resource."
	@echo "  save_hf_token                    - Saves HF token to GCP Secret Manager for the Queued Resource deployer."
	@echo "  analyze_cloud_logging            - Searches Cloud Logging for TPU-related errors and lifecycle events."
	@echo "  verify_model_health              - Performs a deep health check by querying the model with a simple prompt."
	@echo "  query_vllm_with_metrics          - Queries the model and provides streaming-based performance metrics."
	@echo "  get_vllm_model_stats             - Aggregates model-specific statistics from the vLLM server."

