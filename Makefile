SUBDIRS = hybrid-inference-orchestrator \
          serverless-vllm-manager \
          keras-tpu-gpu-pipeline \
          tpu-performance-analyst \
          xla-cross-hardware-profiler \
          autoresearch-serverless-manager \
          self-hosted-vllm-devops-agent \
          devops-model-garden-agent \
          self-hosted-llm-gateway \
          tpu-vllm-strategy-advisor

install-all:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir install; \
	done

test-all:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir test; \
	done

demo-all:
	@echo "🔥 Running all TPU Sprint Grand Demos 🔥"
	@for dir in $(SUBDIRS); do \
		echo "\n--- Running Demo for: $$dir ---"; \
		PYTHONPATH=$$dir python3 $$dir/demo_launcher.py; \
	done

run-all:
	@echo "Warning: Running all MCP servers at once might conflict on stdio."
	@echo "Available MCP servers in subdirectories:"
	@for dir in $(SUBDIRS); do \
		echo "  - $$dir (run with 'make -C $$dir run')"; \
	done

clean-all:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
