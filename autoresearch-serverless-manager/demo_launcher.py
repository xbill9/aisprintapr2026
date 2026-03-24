# Grand Demo: Autonomous AutoResearch Loop Simulation

import asyncio
import random
import os
from server import submit_research_job, analyze_research_costs, get_latest_improvement

async def grand_demo():
    print("🚀 TPU Sprint Grand Demo: AutoResearch Serverless Manager")
    print("=" * 60)
    
    # Step 1: Initialize the Research Job
    print("\n[Step 1] Initializing Autonomous Research Job...")
    job = await submit_research_job(
        job_name="xla-optimization-study",
        image_uri="gcr.io/tpu-sprint/autoresearch-agent:v1",
        research_topic="JAX/XLA Kernel Fusion on L4 GPU"
    )
    print(f"Target: {job['hardware']} | Experiment: {job['experiment_id']}")
    print(f"Command:\n{job['command']}")
    
    # Step 2: Simulate the Agent's Loop (Gemini in action)
    print("\n[Step 2] Agent (Gemini) Analyzing 'train.py' and 'program.md'...")
    await asyncio.sleep(1) # Simulation delay
    print("AI THOUGHT: 'The loop in model_fn can be fused better if I use bfloat16 and enable XLA fusion flags.'")
    
    # Step 3: Simulate Multiple Research Cycles
    print("\n[Step 3] Running Research Cycles ($2/hr budget)...")
    for cycle in range(1, 4):
        print(f"  Cycle {cycle}: Running experiment in Cloud Run Job...")
        await asyncio.sleep(1)
        score = 45.2 + (cycle * random.uniform(2.5, 5.0))
        print(f"  Result: METRIC_SCORE = {score:.2f} (IMPROVEMENT: +{random.uniform(5, 10):.1f}%)")
    
    # Step 4: Cost Analysis
    print("\n[Step 4] Final Cost Analysis for 3 Cycles...")
    costs = await analyze_research_costs(runtime_hours=0.25) # 15 mins total
    print(f"  Total Estimated Cost: ${costs['total_estimated_cost_usd']}")
    print(f"  Budget Efficiency Target: {costs['target_met']}")
    
    # Step 5: Retrieve Best Result
    print("\n[Step 5] Retrieving Best Improvement Artifacts...")
    best = await get_latest_improvement(experiment_id=job['experiment_id'])
    print(f"  Optimized Code Snippet Found:\n{best['improvement_code']}")
    print(f"  Final Artifact Path: {best['artifact_path']}")

    print("\n" + "=" * 60)
    print("✅ Grand Demo Complete: Autonomous research successful on Serverless GPUs!")

if __name__ == "__main__":
    asyncio.run(grand_demo())
