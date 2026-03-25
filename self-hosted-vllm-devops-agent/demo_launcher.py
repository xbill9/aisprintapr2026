# Grand Demo: Self-Hosted vLLM DevOps Agent

import asyncio
from server import analyze_cloud_logging, suggest_sre_remediation, get_vllm_deployment_config, get_vertex_ai_model_copy_instructions, list_vertex_models, get_deployment_template

async def devops_demo():
    print("🚀 TPU Sprint Demo: Self-Hosted vLLM DevOps Agent")
    print("=" * 60)
    
    # Step 1: Log Analysis
    print("\n[Step 1] Analyzing Cloud Logging errors (severity=ERROR)...")
    # Simulate a call where some logs are found
    # (In a real demo, we'd mock the cloud_logging.Client or use a real project)
    analysis = await analyze_cloud_logging(filter_query="severity=ERROR", limit=2)
    print(f"  ANALYSIS: {analysis[:200]}...") # Truncate for display
    
    # Step 2: SRE Remediation
    print("\n[Step 2] Proposing remediation for 'MemoryLimitExceeded'...")
    remediation = await suggest_sre_remediation(error_message="Pod 'vllm-gemma' terminated with Reason: OOMKilled")
    print(f"  REMEDIATION: {remediation}")
    
    # Step 3: Deployment Config & Vertex AI instructions
    print("\n[Step 3] Vertex AI Model Garden Instructions...")
    instructions = get_vertex_ai_model_copy_instructions(model_name="gemma-2b-it")
    print(instructions)
    
    print("\n[Step 4] Generating Cloud Run GPU Deployment Config (with GCS FUSE)...")
    config = get_vllm_deployment_config(service_name="vllm-sre-agent", bucket_name="my-gemma-bucket", model_path="gemma-2b-it")
    print(f"  COMMAND: {config}")

    # Step 5: MCP Resources & Vertex SDK (ADK)
    print("\n[Step 5] Listing available Vertex AI Models (using ADK/SDK)...")
    models = list_vertex_models()
    print(f"  {models[:200]}...") # Truncate for display
    
    print("\n[Step 6] Reading MCP Resource (vLLM Deployment Template)...")
    template = get_deployment_template()
    print(f"  TEMPLATE (first 100 chars): {template.strip()[:100]}...")
    
    print("\n" + "=" * 60)
    print("✅ DevOps Agent Demo Complete: Self-hosted SRE intelligence ready!")

if __name__ == "__main__":
    asyncio.run(devops_demo())
