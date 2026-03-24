# Test Client for AutoResearch Serverless Manager

import asyncio
from server import submit_research_job, analyze_research_costs, get_latest_improvement

async def test_autoresearch():
    print("--- Testing submit_research_job ---")
    job = await submit_research_job(
        job_name="xla-fusion-study", 
        image_uri="gcr.io/tpu-sprint/autoresearch-agent:v1"
    )
    print(f"Status: {job['status']}")
    print(f"Experiment ID: {job['experiment_id']}")
    print(f"Command:\n{job['command']}")
    
    print("\n--- Testing analyze_research_costs ---")
    costs = await analyze_research_costs(runtime_hours=12.0)
    print(f"Total Cost for 12hrs: ${costs['total_estimated_cost_usd']}")
    print(f"Budget Target Met: {costs['target_met']}")

    print("\n--- Testing get_latest_improvement ---")
    improvement = await get_latest_improvement(experiment_id=job['experiment_id'])
    print(f"Summary: {improvement['improvement_summary']}")
    print(f"Code Snippet:\n{improvement['improvement_code']}")

if __name__ == "__main__":
    asyncio.run(test_autoresearch())
