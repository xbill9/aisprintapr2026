# Grand Demo: DevOps/SRE Model Garden Agent

import asyncio
import os
from server import analyze_cloud_logging, suggest_sre_remediation, deploy_gemma_to_cloud_run

async def devops_demo():
    print("🛠️ TPU Sprint Demo: DevOps/SRE Model Garden Agent")
    print("=" * 60)
    
    # Step 1: Mock fetch logs and analyze via Model Garden Gemma
    print("\n[Step 1] Analyzing critical Cloud Logging events (Vertex AI)...")
    analysis = await analyze_cloud_logging(filter_query="severity=ERROR", limit=3)
    print(f"  ANALYSIS:\n{analysis}")
    
    # Step 2: Suggest remediation plan
    print("\n[Step 2] Generating SRE remediation steps via Gemma...")
    remediation = await suggest_sre_remediation(error_message="HTTP 504 Gateway Timeout in us-east4-a")
    print(f"  PLAN:\n{remediation}")
    
    # Step 3: Deployment instruction for serverless Gemma
    print("\n[Step 3] Preparing Cloud Run L4 GPU deployment configuration...")
    deploy_cmd = deploy_gemma_to_cloud_run(service_name="gemma-sre-service", model_path="gs://vertex-ai-models/gemma/gemma-2b-it")
    print(f"  {deploy_cmd}")
    
    print("\n" + "=" * 60)
    print("✅ DevOps SRE Loop Complete: Integrated with Vertex AI Model Garden!")

if __name__ == "__main__":
    asyncio.run(devops_demo())
