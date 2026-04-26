import asyncio
import httpx
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from typing import Optional, List, Dict, Any

from google.cloud import secretmanager
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
async def get_vllm_deployment_config(service_name: str = "gemma4-vllm", model_name: str = MODEL_NAME) -> str:
    """Generates the gcloud command for a single-host TPU v6e vLLM deployment."""
    token = await get_secret() or "${HF_TOKEN}"
    # Quote parameters for safe shell usage inside the script
    q_token = shlex.quote(token)
    q_model = shlex.quote(model_name)
    
    startup_script = (
        "#!/bin/bash\n"
        "sudo docker run --rm --name vllm-gemma4 --privileged --net=host -v /dev/shm:/dev/shm --shm-size 10gb "
        f"-e HF_TOKEN={q_token} vllm/vllm-tpu:nightly vllm serve {q_model} "
        "--max-model-len 16384 --tensor-parallel-size 8 --disable_chunked_mm_input --max_num_batched_tokens 4096 "
        "--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 "
        "--limit-mm-per-prompt '{\"image\":4,\"audio\":1}'"
    )
    
    quoted_script = shlex.quote(startup_script)
    cmd = (
        f"gcloud alpha compute tpus tpu-vm create {service_name} \\\n"
        f"  --zone={ZONE} \\\n"
        f"  --accelerator-type=v6e-8 \\\n"
        f"  --version=v2-alpha-tpuv6e \\\n"
        f"  --metadata=startup-script={quoted_script}"
    )
    return f"### 🚀 TPU v6e (Trillium) Deployment Command\n```bash\n{cmd}\n```"

@mcp.tool()
async def get_vllm_tpu_deployment_config(service_name: str = "gemma4-vllm", model_name: str = MODEL_NAME) -> str:
    """Generates a GKE manifest for a TPU v6e vLLM deployment."""
    token = await get_secret() or "${HF_TOKEN}"
    mm_limit = '{\\"image\\":4,\\"audio\\":1}'
    
    manifest = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {service_name}
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: vllm-tpu
        image: vllm/vllm-tpu:nightly
        command: ["vllm", "serve", "{model_name}"]
        args:
        - "--max-model-len=16384"
        - "--tensor-parallel-size=8"
        - "--disable_chunked_mm_input"
        - "--max_num_batched_tokens=4096"
        - "--enable-auto-tool-choice"
        - "--tool-call-parser=gemma4"
        - "--reasoning-parser=gemma4"
        - "--limit-mm-per-prompt={mm_limit}"
        env:
        - name: HF_TOKEN
          value: "{token}"
        resources:
          requests:
            google.com/tpu: 8
          limits:
            google.com/tpu: 8
"""
    return f"### ☸️ GKE TPU v6e (Trillium) Manifest\n```yaml\n{manifest}\n```"

@mcp.tool()
async def list_queued_resources(zone: str = ZONE) -> str:
    """Lists all Queued Resources in a specific zone."""
    cmd = [
        "gcloud", "alpha", "compute", "tpus", "queued-resources", "list",
        f"--zone={zone}", f"--project={PROJECT_ID}", "--format=table(name, state.state, node_id, accelerator_type, create_time)"
    ]
    rc, out, err = await run_command(cmd)
    return f"### 📋 Queued Resources in {zone}\n```\n{out}\n```" if rc == 0 else f"❌ List failed: {err}"

@mcp.tool()
async def describe_queued_resource(resource_id: str, zone: str = ZONE) -> str:
    """Provides detailed information about a specific Queued Resource."""
    cmd = [
        "gcloud", "alpha", "compute", "tpus", "queued-resources", "describe", resource_id,
        f"--zone={zone}", f"--project={PROJECT_ID}", "--format=json"
    ]
    rc, out, err = await run_command(cmd)
    if rc != 0: return f"❌ Describe failed: {err}"
    
    try:
        data = json.loads(out)
        state = data.get("state", {}).get("state", "UNKNOWN")
        node_id = data.get("tpu", {}).get("nodeSpec", [{}])[0].get("nodeId", "N/A")
        return (
            f"### 🔍 Detail: {resource_id}\n"
            f"- **State:** `{state}`\n"
            f"- **Node ID:** `{node_id}`\n"
            f"- **Full Data:**\n```json\n{json.dumps(data, indent=2)}\n```"
        )
    except:
        return f"### 🔍 Detail: {resource_id}\n```\n{out}\n```"

@mcp.tool()
async def get_reservation_status(resource_id: str) -> str:
    """
    Checks the lifecycle state and expiry time of a Queued Resource.
    
    Args:
        resource_id: The ID of the Queued Resource.
    """
    cmd = [
        "gcloud", "alpha", "compute", "tpus", "queued-resources", "describe", resource_id,
        f"--zone={ZONE}", f"--project={PROJECT_ID}", "--format=json"
    ]
    rc, out, err = await run_command(cmd)
    if rc != 0: return f"❌ Describe failed: {err}"
    
    try:
        data = json.loads(out)
        state = data.get("state", {}).get("state", "UNKNOWN")
        
        # Extract termination timestamp for Flex-start/Queued Resources
        # It's usually in tpu.nodeSpec[0].node.schedulingConfig.terminationTimestamp
        nodes = data.get("tpu", {}).get("nodeSpec", [])
        expiry_str = "N/A"
        time_remaining = "Unknown"
        
        if nodes and "node" in nodes[0]:
            expiry_str = nodes[0]["node"].get("schedulingConfig", {}).get("terminationTimestamp", "N/A")
            
        if expiry_str != "N/A":
            try:
                from datetime import datetime, timezone
                # ISO format: 2026-04-26T09:09:06.486553384Z
                # Python's fromisoformat might need the 'Z' replaced or handled
                expiry_dt = datetime.fromisoformat(expiry_str.replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                diff = expiry_dt - now
                
                if diff.total_seconds() > 0:
                    hours, remainder = divmod(int(diff.total_seconds()), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    time_remaining = f"{hours}h {minutes}m {seconds}s"
                else:
                    time_remaining = "EXPIRED"
            except Exception as e:
                time_remaining = f"Error calculating: {str(e)}"

        return (
            f"### 🎫 Reservation Status: {resource_id}\n"
            f"- **State:** `{state}`\n"
            f"- **Expiry Timestamp:** `{expiry_str}`\n"
            f"- **Time Remaining:** `⏳ {time_remaining}`\n\n"
            f"**Action:** {'Use `destroy_queued_resource` if done to save costs.' if state == 'ACTIVE' else 'Wait for state to reach ACTIVE.'}"
        )
    except Exception as e:
        return f"❌ Parsing error: {str(e)}\n\nRaw: {out}"

@mcp.tool()
async def check_tpu_availability(resource_id: str) -> str:
    """Simple check to see if a Queued Resource has reached ACTIVE state."""
    cmd = [
        "gcloud", "alpha", "compute", "tpus", "queued-resources", "describe", resource_id,
        f"--zone={ZONE}", f"--project={PROJECT_ID}", "--format=value(state.state)"
    ]
    rc, state, err = await run_command(cmd)
    if rc != 0: return f"❌ Check failed: {err}"
    
    state = state.strip()
    is_active = state == "ACTIVE"
    return (
        f"### 🧊 TPU Availability: {resource_id}\n"
        f"- **State:** `{state}`\n"
        f"- **Available:** {'✅ Yes' if is_active else '⏳ No'}\n"
        f"- **Action:** {'Ready to use!' if is_active else 'Wait for provisioning or check logs.'}"
    )

@mcp.tool()
async def estimate_deployment_cost(
    hours: float = 1.0,
    tpu_type: str = "v6e",
    topology: str = "2x4",
    is_flex: bool = True
) -> str:
    """
    Estimates the cost of a TPU deployment.
    
    Args:
        hours: Duration of the deployment in hours.
        tpu_type: TPU version (v6e, v5e, v5p).
        topology: TPU topology (e.g., 2x4, 1x1, 4x4).
        is_flex: Whether using Flex-start (discounted) pricing.
    """
    # Pricing rates (Approximate USD per chip-hour as of 2026)
    # v6e: Standard ~$0.30, Flex ~$0.15
    # v5e: Standard ~$0.24, Flex ~$0.12
    rates = {
        "v6e": {"standard": 0.30, "flex": 0.15},
        "v5e": {"standard": 0.24, "flex": 0.12},
        "v5p": {"standard": 1.20, "flex": 0.60}, # v5p is high-perf
    }
    
    # Calculate chip count from topology (e.g., "2x4" -> 8)
    try:
        parts = [int(p) for p in topology.lower().split('x')]
        chips = 1
        for p in parts: chips *= p
    except:
        chips = 8 # Default for Gemma 4
        
    rate_type = "flex" if is_flex else "standard"
    hourly_rate = rates.get(tpu_type, rates["v6e"])[rate_type]
    
    total_hourly = chips * hourly_rate
    total_cost = total_hourly * hours
    
    return (
        f"### 💸 Estimated TPU Cost Report\n"
        f"- **Config:** `{tpu_type}` topology `{topology}` ({chips} chips)\n"
        f"- **Model:** `{'Flex-start (Spot)' if is_flex else 'On-demand'}`\n"
        f"- **Duration:** `{hours} hours`\n"
        f"---\n"
        f"- **Rate per Chip/Hr:** `${hourly_rate:.2f}`\n"
        f"- **Total Hourly Rate:** `${total_hourly:.2f}/hr`\n"
        f"- **Estimated Total:** `${total_cost:.2f}`\n\n"
        f"⚠️ *Note: Prices are estimates based on us-central1 rates. Actual billing may vary.*"
    )

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

    # 4. Next Step Recommendation
    next_step = "🚀 Call `orchestrate_gemma4_stack` to start a new deployment."
    if "ACTIVE" in r_out:
        if "🟢 Online" in health:
            next_step = "✅ System is Ready! Use `query_queued_gemma4` or `validate_gemma4_deployment`."
        else:
            next_step = "⏳ TPU is ACTIVE but vLLM is still booting. Use `fetch_queued_node_logs` to check progress."
    elif "ACCEPTED" in r_out or "PROVISIONING" in r_out:
        next_step = "⏳ TPU is being provisioned. This usually takes 2-5 minutes. Check again shortly."

    return (
        f"### 🌀 Queued Resource Status ({ZONE})\n"
        f"- **TPU v6e Quota:** {available_v6e} chips available\n"
        f"- **vLLM Health:** {health}\n\n"
        f"#### Managed Queued Resources:\n```\n{r_out}\n```\n"
        f"**👉 Next Step:** {next_step}"
    )

@mcp.tool()
async def orchestrate_gemma4_stack(resource_id: str = "vllm-gemma4-qr", hf_token: Optional[str] = None) -> str:
    """
    Seamless turnkey deployment:
    1. Saves HF Token (if provided).
    2. Validates Quota.
    3. Initiates Queued Resource creation with optimized Gemma 4 vLLM stack.
    """
    # 1. Handle Token
    if hf_token:
        await save_hf_token(hf_token)
    else:
        token = await get_secret()
        if not token:
            return "❌ Seamless Deployment Aborted: No `hf_token` provided and none found in Secret Manager."

    # 2. Check Quota
    quota_res = await get_system_status()
    if "0 chips available" in quota_res and "ACTIVE" not in quota_res:
        return f"❌ Quota Check Failed:\n{quota_res}\n\nPlease check your GCP console for TPU-V6E quota."

    # 3. Deploy
    deploy_res = await deploy_queued_vllm(resource_id)
    if "❌" in deploy_res:
        return deploy_res

    return (
        f"## 🌊 Seamless Gemma 4 Deployment Initiated\n"
        f"{deploy_res}\n\n"
        f"**Deployment Roadmap:**\n"
        f"1. **Provisioning:** TPU VM is being allocated (2-5 mins).\n"
        f"2. **Booting:** Docker image `vllm/vllm-tpu:nightly` is being pulled.\n"
        f"3. **Serving:** vLLM initializes Gemma 4 weights and starts the API.\n\n"
        f"**Monitoring:** Use `get_system_status` to track progress."
    )

@mcp.tool()
async def get_vllm_endpoint() -> str:
    """Discovery tool to verify connectivity and return the active vLLM service URL."""
    url = await discover_vllm_url()
    if url:
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(f"{url}/health", timeout=2)
                if res.status_code == 200:
                    return f"🟢 vLLM is Online at: {url}"
        except:
            return f"🟡 vLLM found at {url} but health check failed."
    return "❌ No ACTIVE Queued Resource with a reachable vLLM service found."

@mcp.tool()
async def deploy_queued_vllm(resource_id: str = "vllm-gemma4-qr") -> str:
    """Deploys vLLM strictly using Queued Resources for Flex-start allocation."""
    token = await get_secret()
    if not token: return "❌ Deployment Aborted: 'hf-token' secret missing."

    node_id = f"{resource_id}-node"
    
    # Quote parameters for safe shell usage inside the script
    q_token = shlex.quote(token)
    q_model = shlex.quote(MODEL_NAME)
    
    # Watchdog Startup Script optimized for Gemma 4
    startup_script = (
        "#!/bin/bash\n"
        "exec > /var/log/vllm-startup.log 2>&1\n"
        "set -x\n"
        "echo 'Starting Queued vLLM Bootloader...'\n"
        "mkdir -p /root/.cache/huggingface\n"
        "for i in {1..30}; do ping -c 1 8.8.8.8 && break || sleep 5; done\n"
        "for i in {1..5}; do sudo docker pull vllm/vllm-tpu:nightly && break || sleep 20; done\n"
        "while true; do\n"
        "  sudo docker run --rm --name vllm-gemma4 --privileged --net=host "
        "-v /dev/shm:/dev/shm --shm-size 16gb "
        "-v /root/.cache/huggingface:/root/.cache/huggingface "
        f"-e HF_TOKEN={q_token} vllm/vllm-tpu:nightly vllm serve {q_model} "
        "--max-model-len 16384 --tensor-parallel-size 8 --disable_chunked_mm_input --max_num_batched_tokens 4096 "
        "--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 "
        "--limit-mm-per-prompt '{\"image\":4,\"audio\":1}'\n"
        "  sleep 30\n"
        "done\n"
    )

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as tf:
        tf.write(startup_script)
        temp_script_path = tf.name

    cmd = [
        "gcloud", "alpha", "compute", "tpus", "queued-resources", "create", resource_id,
        f"--zone={ZONE}", "--type=v6e", "--topology=2x4", "--runtime-version=v2-alpha-tpuv6e",
        f"--node-id={node_id}", "--provisioning-model=flex-start", "--max-run-duration=1h", "--valid-until-duration=1h",
        f"--metadata-from-file=startup-script={temp_script_path}", f"--project={PROJECT_ID}"
    ]
    
    try:
        logger.info(f"Creating Queued Resource {resource_id}...")
        rc, out, err = await run_command(cmd)
        if rc != 0: return f"❌ Queued Resource Creation Failed: {err}"
    finally:
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)
    
    return (
        f"🚀 **Queued Resource Created: {resource_id}**\n"
        f"- **Node ID:** `{node_id}`\n"
        f"- **Zone:** `{ZONE}`\n"
        f"- **Action:** Monitor state with `get_system_status`. It must reach `ACTIVE` before serving starts."
    )

@mcp.tool()
async def create_tpu_queued_resource(
    resource_id: str,
    node_id: str,
    zone: str = ZONE,
    tpu_type: str = "v6e",
    topology: str = "2x4",
    runtime_version: str = "v2-alpha-tpuv6e",
    max_run_duration: str = "1h"
) -> str:
    """
    Creates a TPU Queued Resource (Flex-start) with the specified configuration.
    This is useful for general provisioning tasks.
    """
    cmd = [
        "gcloud", "alpha", "compute", "tpus", "queued-resources", "create", resource_id,
        f"--zone={zone}", f"--type={tpu_type}", f"--topology={topology}",
        f"--runtime-version={runtime_version}", f"--node-id={node_id}",
        "--provisioning-model=flex-start", f"--max-run-duration={max_run_duration}",
        f"--valid-until-duration={max_run_duration}", "--labels=purpose=flex-start",
        f"--project={PROJECT_ID}"
    ]
    
    logger.info(f"Creating Queued Resource {resource_id}...")
    rc, out, err = await run_command(cmd)
    if rc != 0: return f"❌ Queued Resource Creation Failed: {err}"
    
    return (
        f"🚀 **Queued Resource Created: {resource_id}**\n"
        f"- **Node ID:** `{node_id}`\n"
        f"- **Zone:** `{zone}`\n"
        f"- **Type:** `{tpu_type}`\n"
        f"- **Topology:** `{topology}`\n"
        f"- **Action:** Monitor state with `get_system_status`."
    )

@mcp.tool()
async def check_tpu_utilization(resource_id: str) -> str:
    """Monitors Tensor Core and HBM pressure on the TPU VM."""
    desc_cmd = ["gcloud", "alpha", "compute", "tpus", "queued-resources", "describe", resource_id, f"--project={PROJECT_ID}", f"--zone={ZONE}", "--format=value(tpu.nodeSpec[0].nodeId)"]
    rc, node_id, _ = await run_command(desc_cmd)
    if rc != 0 or not node_id: return f"❌ Could not find node for resource {resource_id}"

    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "ssh", node_id.strip(),
        f"--zone={ZONE}", f"--project={PROJECT_ID}", "--command",
        r"sudo docker logs --tail 100 vllm-gemma4 | grep -i 'memory\|utilization\|usage'"
    ]
    rc_util, out, _ = await run_command(cmd)
    return f"### 📊 TPU Utilization for {node_id}\n```\n{out}\n```" if rc_util == 0 else "❌ Failed to fetch utilization metrics."

@mcp.tool()
async def get_vllm_metrics() -> str:
    """Fetches real-time Prometheus metrics from the active vLLM service."""
    url = await discover_vllm_url()
    if not url: return "❌ No active vLLM service found."
    
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{url}/metrics", timeout=5)
            # Filter for high-signal performance metrics
            signal_keys = ["num_requests_running", "num_requests_swapped", "gpu_cache_usage", "num_requests_waiting"]
            metrics = []
            for line in res.text.splitlines():
                if any(key in line for key in signal_keys) and "help" not in line.lower() and "type" not in line.lower():
                    metrics.append(line)
            
            if not metrics:
                # Fallback to general vllm metrics if specific signal keys aren't found
                metrics = [line for line in res.text.splitlines() if "vllm:" in line and "help" not in line.lower()][:20]
                
            return "### 📈 vLLM Performance Metrics\n```\n" + "\n".join(metrics) + "\n```"
    except Exception as e:
        return f"❌ Failed to fetch metrics: {str(e)}"

@mcp.tool()
async def validate_gemma4_deployment(resource_id: str) -> str:
    """Performs a comprehensive sanity check on the Gemma 4 deployment."""
    endpoint = await get_vllm_endpoint()
    if "🟢" not in endpoint: return f"❌ Validation failed: {endpoint}"
    
    test_prompt = "Say 'Gemma 4 is active' if you can hear me."
    query_res = await query_queued_gemma4(test_prompt)
    
    logs = await fetch_queued_node_logs(resource_id, tail=200)
    config_ok = "--tool-call-parser gemma4" in logs
    
    return (
        f"## ✅ Gemma 4 Deployment Validation\n"
        f"- **Connectivity:** {endpoint}\n"
        f"- **Config Flags:** {'Verified' if config_ok else '⚠️ Warning: parser flags missing in logs'}\n"
        f"- **Logic Test:** {query_res}"
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
    # We use shlex.quote to ensure MODEL_NAME and other parameters are safe for the remote shell
    q_model = shlex.quote(MODEL_NAME)
    bench_cmd = (
        f"sudo docker exec vllm-gemma4 vllm bench serve "
        f"--backend vllm "
        f"--model {q_model} "
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
async def run_load_test_benchmark(

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
    # We use shlex.quote to ensure MODEL_NAME and other parameters are safe for the remote shell
    q_model = shlex.quote(MODEL_NAME)
    bench_cmd = (
        f"sudo docker exec vllm-gemma4 vllm bench serve "
        f"--backend vllm "
        f"--model {q_model} "
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
async def run_load_test_benchmark(
    num_requests: int = 10,
    concurrency: int = 2,
    max_tokens: int = 128
) -> str:
    """
    Performs an external load test against the active vLLM endpoint.
    Reports success rate, average/P95 latency, and throughput (req/s).
    """
    url = await discover_vllm_url()
    if not url: return "❌ No active vLLM service found."
    
    endpoint = f"{url}/v1/completions"
    prompt = "Benchmarking Gemma 4 on TPU v6e. Tell me a short story about Trillium."
    
    results = []
    start_test = time.time()

    async def _send_req():
        async with httpx.AsyncClient() as client:
            s = time.time()
            try:
                res = await client.post(
                    endpoint, 
                    json={"model": MODEL_NAME, "prompt": prompt, "max_tokens": max_tokens},
                    timeout=60
                )
                if res.status_code == 200:
                    return time.time() - s
            except: pass
            return None

    # Run batches with concurrency
    for i in range(0, num_requests, concurrency):
        batch = [ _send_req() for _ in range(min(concurrency, num_requests - i)) ]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)

    total_test_time = time.time() - start_test
    latencies = [l for l in results if l is not None]
    
    if not latencies:
        return "❌ Load test failed: All requests failed. Check connectivity."

    avg_lat = sum(latencies) / len(latencies)
    sorted_lat = sorted(latencies)
    p95_lat = sorted_lat[int(len(latencies) * 0.95)]
    throughput = len(latencies) / total_test_time

    return (
        f"## 🏎️ External Load Test Results\n"
        f"- **Endpoint:** `{url}`\n"
        f"- **Model:** `{MODEL_NAME}`\n"
        f"- **Successes:** `{len(latencies)}/{num_requests}`\n"
        f"- **Total Time:** `{total_test_time:.2f}s`\n"
        f"- **Throughput:** `{throughput:.2f} req/s`\n"
        f"- **Avg Latency:** `{avg_lat:.2f}s`\n"
        f"- **P95 Latency:** `{p95_lat:.2f}s`"
    )

@mcp.tool()
async def get_gemma4_full_report(resource_id: str = "gemma4-vllm-stack") -> str:
    """
    Generates a deep technical report of the Gemma 4 deployment:
    1. Infrastructure (Queued Resource specs)
    2. Compute (TPU Node network/state)
    3. Software (vLLM version, model config, metrics)
    """
    # 1. QR Data
    qr_cmd = ["gcloud", "alpha", "compute", "tpus", "queued-resources", "describe", resource_id, f"--project={PROJECT_ID}", f"--zone={ZONE}", "--format=json"]
    _, qr_out, _ = await run_command(qr_cmd)
    
    # 2. Node Data
    node_id = f"{resource_id}-node"
    node_cmd = ["gcloud", "compute", "tpus", "tpu-vm", "describe", node_id, f"--project={PROJECT_ID}", f"--zone={ZONE}", "--format=json"]
    _, node_out, _ = await run_command(node_cmd)
    
    # 3. vLLM Data
    metrics = await get_vllm_metrics()
    stats = await get_vllm_model_stats()
    
    # Parse and Format
    try:
        qr = json.loads(qr_out)
        node = json.loads(node_out)
        
        report = [
            f"# 📜 Detailed Deployment Report: {resource_id}",
            f"\n## 🏗️ Infrastructure (GCP)",
            f"- **State:** `{qr.get('state', {}).get('state')}`",
            f"- **TPU Type:** `{qr.get('tpu', {}).get('nodeSpec', [{}])[0].get('node', {}).get('acceleratorConfig', {}).get('type')}`",
            f"- **Topology:** `{qr.get('tpu', {}).get('nodeSpec', [{}])[0].get('node', {}).get('acceleratorConfig', {}).get('topology')}`",
            f"- **Created:** `{qr.get('createTime')}`",
            
            f"\n## 🖥️ Compute (TPU VM)",
            f"- **Node ID:** `{node_id}`",
            f"- **Internal IP:** `{node.get('networkEndpoints', [{}])[0].get('ipAddress')}`",
            f"- **External IP:** `{node.get('networkEndpoints', [{}])[0].get('accessConfig', {}).get('externalIp')}`",
            f"- **Health:** `{node.get('health')}`",
            
            f"\n## 🧠 Model & Inference (vLLM)",
            stats,
            f"\n### 📊 Live Metrics",
            metrics,
            
            f"\n## ⚙️ Optimization Standards",
            "- **Precision:** `BF16` (Weights) / `FP8` (KV Cache on v6e)",
            "- **Parallelism:** `TP=8`",
            "- **Max Seq Len:** `16384`",
            "- **Engine:** `Flax/JAX (OpenXLA)`"
        ]
        return "\n".join(report)
    except Exception as e:
        return f"❌ Failed to generate full report: {str(e)}\n\nRaw Data Available:\nQR: {qr_out[:200]}...\nNode: {node_out[:200]}..."

@mcp.tool()
async def fetch_tpu_vm_logs(resource_id: Optional[str] = None, node_id: Optional[str] = None, log_type: str = "vllm", tail: int = 100) -> str:
    """
    Fetches specific logs from a TPU VM.
    log_type options: 'vllm' (docker), 'startup' (/var/log/vllm-startup.log), 'system' (journalctl -u tpu-runtime)
    """
    if not node_id and resource_id:
        desc_cmd = ["gcloud", "alpha", "compute", "tpus", "queued-resources", "describe", resource_id, f"--project={PROJECT_ID}", f"--zone={ZONE}", "--format=value(tpu.nodeSpec[0].nodeId)"]
        rc, res_node, _ = await run_command(desc_cmd)
        if rc == 0 and res_node:
            node_id = res_node.strip()
    
    if not node_id:
        return "❌ Error: Either `node_id` or `resource_id` must be provided."

    log_map = {
        "vllm": "sudo docker logs vllm-gemma4",
        "startup": "sudo tail -n {tail} /var/log/vllm-startup.log",
        "system": "sudo journalctl -u tpu-runtime --no-pager | tail -n {tail}"
    }
    
    cmd_template = log_map.get(log_type, log_map["vllm"])
    if "{tail}" in cmd_template:
        command = cmd_template.format(tail=tail)
    else:
        command = f"{cmd_template} --tail {tail}"

    ssh_cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "ssh", node_id,
        f"--zone={ZONE}", f"--project={PROJECT_ID}", "--command", command
    ]
    
    rc_log, out, err = await run_command(ssh_cmd)
    return f"### 📝 {log_type.upper()} Logs for {node_id}\n```\n{out}\n```" if rc_log == 0 else f"❌ Failed to fetch logs: {err}"

@mcp.tool()
async def grep_tpu_logs(pattern: str, resource_id: Optional[str] = None, node_id: Optional[str] = None) -> str:
    """Searches for a pattern in both startup and container logs on the TPU VM."""
    if not node_id and resource_id:
        desc_cmd = ["gcloud", "alpha", "compute", "tpus", "queued-resources", "describe", resource_id, f"--project={PROJECT_ID}", f"--zone={ZONE}", "--format=value(tpu.nodeSpec[0].nodeId)"]
        rc, res_node, _ = await run_command(desc_cmd)
        if rc == 0 and res_node:
            node_id = res_node.strip()
            
    if not node_id:
        return "❌ Error: Either `node_id` or `resource_id` must be provided."

    command = f"grep -i '{pattern}' /var/log/vllm-startup.log; sudo docker logs vllm-gemma4 2>&1 | grep -i '{pattern}'"
    
    ssh_cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "ssh", node_id,
        f"--zone={ZONE}", f"--project={PROJECT_ID}", "--command", command
    ]
    
    rc_log, out, _ = await run_command(ssh_cmd)
    return f"### 🔍 Search Results for '{pattern}' on {node_id}\n```\n{out}\n```" if out else f"ℹ️ No matches found for '{pattern}'."

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

@mcp.tool()
async def analyze_cloud_logging(minutes: int = 60) -> str:
    """Searches Cloud Logging for TPU-related errors and lifecycle events."""
    log_filter = (
        f'resource.type="tpu_worker" AND '
        f'resource.labels.project_id="{PROJECT_ID}" AND '
        f'severity>=ERROR'
    )
    
    cmd = [
        "gcloud", "logging", "read", log_filter,
        f"--project={PROJECT_ID}", "--limit=20", "--format=json",
        f"--freshness={minutes}m"
    ]
    
    rc, out, err = await run_command(cmd)
    if rc != 0: return f"❌ Logging analysis failed: {err}"
    
    try:
        logs = json.loads(out)
        if not logs: return f"✅ No ERROR logs found in the last {minutes} minutes."
        
        summary = []
        for log in logs:
            ts = log.get('timestamp', 'N/A')
            msg = log.get('textPayload', log.get('jsonPayload', {}).get('message', 'No message'))
            summary.append(f"- [{ts}] {msg}")
        
        return f"### 🔍 Cloud Logging Analysis (Last {minutes}m)\n" + "\n".join(summary)
    except:
        return f"### 🔍 Raw Logs\n```\n{out}\n```"

@mcp.tool()
async def suggest_sre_remediation(status_or_error: str) -> str:
    """Provides actionable remediation steps for common TPU failure modes."""
    suggestions = []
    status_upper = status_or_error.upper()
    
    if "WAITING_FOR_RESOURCES" in status_upper:
        suggestions.append("- **Issue:** Resource exhaustion in zone (Flex-start queue).")
        suggestions.append(f"- **Fix:** Check quota with `get_system_status`. Consider `us-east5-a` or `southamerica-east1-c`.")
    
    if any(x in status_upper for x in ["COMMUNICATION_ERROR", "HEALTH_CHECK_FAIL", "UNREACHABLE"]):
        suggestions.append("- **Issue:** TPU Node/Network mismatch or container crash.")
        suggestions.append("- **Fix:** Use `fetch_queued_node_logs` to check for OOM or Docker pull failures.")
        suggestions.append("- **Action:** Restart with `destroy_queued_resource` followed by `deploy_queued_vllm`.")

    if "FAILED" in status_upper or "PROVISIONING" in status_upper:
        suggestions.append("- **Issue:** Lifecycle transition failure.")
        suggestions.append("- **Fix:** Run `describe_queued_resource` to see the `state.details` message from the TPU controller.")
        
    if not suggestions:
        return "❓ No specific remediation found. Please provide logs from `analyze_cloud_logging` or `fetch_queued_node_logs`."
        
    return "### 🛠️ SRE Remediation Suggestions\n" + "\n".join(suggestions)

@mcp.tool()
async def verify_model_health() -> str:
    """Performs a deep health check by querying the model with a simple prompt."""
    try:
        client = await get_vllm_client()
        start = time.time()
        res = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "health_check"}],
            max_tokens=5
        )
        latency = time.time() - start
        content = res.choices[0].message.content.strip()
        return f"✅ **Health Check PASSED**\n- **Response:** `{content}`\n- **Latency:** `{latency:.2f}s`"
    except Exception as e:
        return f"❌ **Health Check FAILED**: {str(e)}"

@mcp.tool()
async def query_vllm_with_metrics(prompt: str) -> str:
    """Queries the model and provides streaming-based performance metrics."""
    try:
        client = await get_vllm_client()
        start = time.time()
        ttft = 0.0
        full_content = []
        
        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            stream=True
        )
        
        async for chunk in stream:
            if not ttft:
                ttft = time.time() - start
            if chunk.choices[0].delta.content:
                full_content.append(chunk.choices[0].delta.content)
        
        total_time = time.time() - start
        response_text = "".join(full_content)
        
        return (
            f"### 🤖 Response\n{response_text}\n\n"
            f"### 📊 Performance Metrics\n"
            f"- **TTFT (Time to First Token):** `{ttft:.3f}s`\n"
            f"- **Total Latency:** `{total_time:.3f}s`"
        )
    except Exception as e:
        return f"❌ **Query Failed**: {str(e)}"

@mcp.tool()
async def get_vllm_model_stats() -> str:
    """Aggregates model-specific statistics from the vLLM server."""
    url = await discover_vllm_url()
    if not url: return "❌ No active vLLM service found."
    
    try:
        async with httpx.AsyncClient() as client:
            # 1. API Health
            health = await client.get(f"{url}/health")
            # 2. Model Version/Info
            models = await client.get(f"{url}/v1/models")
            
            return (
                f"### 📈 Model Statistics\n"
                f"- **Model ID:** `{MODEL_NAME}`\n"
                f"- **API Status:** `{'🟢 Online' if health.status_code == 200 else '🔴 Error'}`\n"
                f"- **Load Info:** {json.dumps(models.json(), indent=2)}"
            )
    except Exception as e:
        return f"❌ Failed to fetch stats: {str(e)}"

if __name__ == "__main__":
    mcp.run()
