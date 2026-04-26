import asyncio
import httpx
import json
import logging
import os
import subprocess
import sys
import time
from typing import Optional, List, Dict, Any

from google.cloud import secretmanager
from google.cloud import logging as cloud_logging
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI

# Setup logging
logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vllm-devops-agent")

# Initialize FastMCP server
mcp = FastMCP("Queued TPU vLLM Agent (Gemma 4)")

# Configuration - STRICTLY us-central1-a & Queued Resources focus
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "aisprint-491218")
ZONE = "us-central1-a"
REGION = "us-central1"
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-31B-it")
HF_SECRET_ID = "hf-token"

async def run_command(cmd: list[str], timeout: int = 60) -> tuple[int, str, str]:
    """Runs a shell command asynchronously."""
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return process.returncode or 0, stdout.decode().strip(), stderr.decode().strip()
    except asyncio.TimeoutError:
        try: process.kill()
        except: pass
        return -1, "", f"Timeout after {timeout}s"
    except Exception as e:
        return -1, "", str(e)

async def get_secret(secret_id: str = HF_SECRET_ID) -> Optional[str]:
    """Retrieves a secret from Secret Manager."""
    def _get():
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
        try:
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except: return None
    return await asyncio.to_thread(_get)

async def discover_vllm_url() -> Optional[str]:
    """
    Discovery driven strictly by Queued Resources.
    Finds ACTIVE Queued Resources and extracts the IP from their associated nodes.
    """
    # 1. List all Queued Resources in the zone
    list_cmd = [
        "gcloud", "alpha", "compute", "tpus", "queued-resources", "list",
        f"--project={PROJECT_ID}", f"--zone={ZONE}", "--format=json"
    ]
    rc, stdout, stderr = await run_command(list_cmd)
    if rc != 0 or not stdout: return None
    
    try:
        resources = json.loads(stdout)
        for res in resources:
            # Check if the resource is ACTIVE
            if res.get("state", {}).get("state") == "ACTIVE":
                resource_id = res.get("name", "").split("/")[-1]
                # Describe the Queued Resource to find node IP
                desc_cmd = [
                    "gcloud", "alpha", "compute", "tpus", "queued-resources", "describe", resource_id,
                    f"--project={PROJECT_ID}", f"--zone={ZONE}", "--format=json"
                ]
                rc_d, stdout_d, _ = await run_command(desc_cmd)
                if rc_d == 0 and stdout_d:
                    data = json.loads(stdout_d)
                    # For v6e, we check the node list in the description
                    nodes = data.get("tpu", {}).get("nodeSpec", [])
                    # We also need the actual VM status which is linked
                    for node_spec in nodes:
                        node_id = node_spec.get("nodeId")
                        if node_id:
                            # Final step: Get the IP of the manifest node
                            ip_cmd = ["gcloud", "compute", "tpus", "tpu-vm", "describe", node_id, 
                                     f"--project={PROJECT_ID}", f"--zone={ZONE}", 
                                     "--format=value(networkEndpoints[0].accessConfig.externalIp)"]
                            rc_ip, ip, _ = await run_command(ip_cmd)
                            if not ip:
                                ip_cmd[-1] = "value(networkEndpoints[0].ipAddress)"
                                _, ip, _ = await run_command(ip_cmd)
                            
                            if ip: 
                                url = f"http://{ip.strip()}:8000"
                                logger.info(f"📡 Found ACTIVE Queued Resource {resource_id} node at {url}")
                                return url
    except Exception as e:
        logger.error(f"Discovery error: {e}")
    return None

async def get_vllm_client() -> AsyncOpenAI:
    url = await discover_vllm_url()
    if not url: raise Exception(f"No ACTIVE Queued Resource found in {ZONE}.")
    return AsyncOpenAI(base_url=f"{url}/v1", api_key="not-needed")

@mcp.tool()
async def get_system_status() -> str:
    """Status dashboard prioritizing Queued Resource states in us-central1-a."""
    # 1. Check Quota
    quota_cmd = ["gcloud", "compute", "regions", "describe", REGION, f"--project={PROJECT_ID}", "--format=json(quotas)"]
    _, q_out, _ = await run_command(quota_cmd)
    available_v6e = 0
    try:
        quotas = json.loads(q_out).get("quotas", [])
        v6e_q = next((q for q in quotas if "TPU-V6E" in q["metric"]), {})
        available_v6e = v6e_q.get("limit", 0) - v6e_q.get("usage", 0)
    except: pass

    # 2. List Queued Resources (Source of truth)
    list_cmd = [
        "gcloud", "alpha", "compute", "tpus", "queued-resources", "list",
        f"--zone={ZONE}", f"--project={PROJECT_ID}", "--format=table(name, state.state, node_id, accelerator_type)"
    ]
    _, r_out, _ = await run_command(list_cmd)

    # 3. Endpoint Health
    health = "🔴 Offline"
    url = await discover_vllm_url()
    if url:
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(f"{url}/health", timeout=2)
                if res.status_code == 200: health = f"🟢 Online ({url})"
        except: pass

    return (
        f"### 🌀 Queued Resource Status ({ZONE})\n"
        f"- **TPU v6e Quota:** {available_v6e} chips available\n"
        f"- **vLLM Health:** {health}\n\n"
        f"#### Managed Queued Resources:\n```\n{r_out}\n```"
    )

@mcp.tool()
async def deploy_queued_vllm(resource_id: str = "vllm-gemma4-qr") -> str:
    """Deploys vLLM strictly using Queued Resources for Flex-start allocation."""
    token = await get_secret()
    if not token: return "❌ Deployment Aborted: 'hf-token' secret missing."

    node_id = f"{resource_id}-node"
    
    # Watchdog Startup Script
    startup_script = (
        "#!/bin/bash\n"
        "exec > /var/log/vllm-startup.log 2>&1\n"
        "set -x\n"
        "echo 'Starting Queued vLLM Bootloader...'\n"
        "for i in {1..30}; do ping -c 1 8.8.8.8 && break || sleep 5; done\n"
        "for i in {1..5}; do sudo docker pull vllm/vllm-tpu:nightly && break || sleep 20; done\n"
        "while true; do\n"
        "  sudo docker run --rm --name vllm-gemma4 --privileged --net=host -v /dev/shm:/dev/shm --shm-size 10gb "
        f"-e HF_TOKEN={token} vllm/vllm-tpu:nightly vllm serve {MODEL_NAME} "
        "--max-model-len 16384 --tensor-parallel-size 8 --disable_chunked_mm_input "
        "--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4\n"
        "  sleep 30\n"
        "done\n"
    )

    cmd = [
        "gcloud", "alpha", "compute", "tpus", "queued-resources", "create", resource_id,
        f"--zone={ZONE}", "--type=v6e", "--topology=2x4", "--runtime-version=v2-alpha-tpuv6e",
        f"--node-id={node_id}", "--provisioning-model=flex-start", "--max-run-duration=4h",
        f"--metadata=startup-script={startup_script}", f"--project={PROJECT_ID}"
    ]
    
    logger.info(f"Creating Queued Resource {resource_id}...")
    rc, out, err = await run_command(cmd)
    if rc != 0: return f"❌ Queued Resource Creation Failed: {err}"
    
    return (
        f"🚀 **Queued Resource Created: {resource_id}**\n"
        f"- **Node ID:** `{node_id}`\n"
        f"- **Zone:** `{ZONE}`\n"
        f"- **Action:** Monitor state with `get_system_status`. It must reach `ACTIVE` before serving starts."
    )

@mcp.tool()
async def query_queued_gemma4(prompt: str) -> str:
    """Queries the model hosted on the active Queued Resource."""
    try:
        client = await get_vllm_client()
        res = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.2
        )
        return f"### Response (via Queued Resource)\n\n{res.choices[0].message.content}"
    except Exception as e:
        return f"❌ Query Failed: {str(e)}"

@mcp.tool()
async def run_vllm_benchmark(
    resource_id: str,
    num_prompts: int = 100,
    input_len: int = 1024,
    output_len: int = 128
) -> str:
    """
    Runs vLLM's internal benchmark tool inside the container on the TPU VM.
    
    Args:
        resource_id: The ID of the Queued Resource.
        num_prompts: Number of prompts to send.
        input_len: Length of each random input prompt.
        output_len: Length of each random output generation.
    """
    # 1. Identify the node linked to the resource
    desc_cmd = [
        "gcloud", "alpha", "compute", "tpus", "queued-resources", "describe", resource_id,
        f"--project={PROJECT_ID}", f"--zone={ZONE}", "--format=value(tpu.nodeSpec[0].nodeId)"
    ]
    rc, node_id, _ = await run_command(desc_cmd)
    if rc != 0 or not node_id: return f"❌ Could not find node for resource {resource_id}"
    
    node_id = node_id.strip()

    # 2. Build the benchmark command to run inside the container
    bench_cmd = (
        f"sudo docker exec vllm-gemma4 vllm bench serve "
        f"--backend vllm "
        f"--model {MODEL_NAME} "
        f"--dataset-name random "
        f"--num-prompts {num_prompts} "
        f"--random-input-len {input_len} "
        f"--random-output-len {output_len}"
    )

    # 3. Execute via SSH
    ssh_cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "ssh", node_id,
        f"--zone={ZONE}", f"--project={PROJECT_ID}", "--command", bench_cmd
    ]
    
    logger.info(f"Running vLLM benchmark on node {node_id}...")
    rc_bench, out, err = await run_command(ssh_cmd, timeout=600) # Long timeout for benchmark
    
    if rc_bench == 0:
        return f"### 📊 Benchmark Results for {resource_id}\n\n{out}"
    else:
        return f"❌ Benchmark Failed: {err}\n\nOutput: {out}"

@mcp.tool()
async def destroy_queued_resource(resource_id: str) -> str:
    """Safely deletes a Queued Resource and its associated node."""
    cmd = [
        "gcloud", "alpha", "compute", "tpus", "queued-resources", "delete", resource_id,
        f"--zone={ZONE}", f"--project={PROJECT_ID}", "--quiet", "--force"
    ]
    rc, out, err = await run_command(cmd)
    return f"✅ Deletion of Queued Resource {resource_id} initiated." if rc == 0 else f"❌ Cleanup failed: {err}"

@mcp.tool()
async def fetch_queued_node_logs(resource_id: str, tail: int = 50) -> str:
    """Fetches logs by identifying the node linked to a Queued Resource."""
    # First, get the node_id from the resource
    desc_cmd = ["gcloud", "alpha", "compute", "tpus", "queued-resources", "describe", resource_id, f"--project={PROJECT_ID}", f"--zone={ZONE}", "--format=value(tpu.nodeSpec[0].nodeId)"]
    rc, node_id, _ = await run_command(desc_cmd)
    if rc != 0 or not node_id: return f"❌ Could not find node for resource {resource_id}"

    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "ssh", node_id.strip(),
        f"--zone={ZONE}", f"--project={PROJECT_ID}", "--command",
        f"sudo tail -n {tail} /var/log/vllm-startup.log && sudo docker logs --tail {tail} vllm-gemma4"
    ]
    rc_log, out, _ = await run_command(cmd)
    return f"### 📝 Logs for {node_id}\n```\n{out}\n```" if rc_log == 0 else "❌ SSH log fetch failed."

@mcp.tool()
async def save_hf_token(token: str) -> str:
    """Saves HF token to GCP Secret Manager for the Queued Resource deployer."""
    def _save():
        client = secretmanager.SecretManagerServiceClient()
        parent = f"projects/{PROJECT_ID}"
        try: client.create_secret(request={"parent": parent, "secret_id": HF_SECRET_ID, "secret": {"replication": {"automatic": {}}}})
        except: pass
        client.add_secret_version(request={"parent": f"{parent}/secrets/{HF_SECRET_ID}", "payload": {"data": token.encode("UTF-8")}})
    try:
        await asyncio.to_thread(_save)
        return "✅ Token saved. Ready for Queued Resource deployment."
    except Exception as e:
        return f"❌ Failed: {str(e)}"

if __name__ == "__main__":
    mcp.run()
