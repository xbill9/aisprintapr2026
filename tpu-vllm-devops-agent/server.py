import asyncio
import json
import logging
import os
import shlex
import sys
import time
from typing import Optional

import httpx
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

# --- Configuration ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "aisprint-491218")
ZONE = os.getenv("GOOGLE_CLOUD_ZONE", "southamerica-east1-c")
REGION = os.getenv("GOOGLE_CLOUD_REGION", "southamerica-east1")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-31B-it")
HF_SECRET_ID = "hf-token"

# --- Helper Functions ---


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
        try:
            process.kill()
        except Exception:
            pass
        return -1, "", f"Timeout after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


async def _get_node_id(resource_id: str) -> Optional[str]:
    """Retrieves the node ID for a given Queued Resource."""
    cmd = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "queued-resources",
        "describe",
        resource_id,
        f"--project={PROJECT_ID}",
        f"--zone={ZONE}",
        "--format=value(tpu.nodeSpec[0].nodeId)",
    ]
    rc, node_id, _ = await run_command(cmd)
    return node_id.strip() if rc == 0 and node_id else None


async def _get_node_ip(node_id: str) -> Optional[str]:
    """Gets the external or internal IP of a TPU node."""
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "describe",
        node_id,
        f"--project={PROJECT_ID}",
        f"--zone={ZONE}",
        "--format=value(networkEndpoints[0].accessConfig.externalIp)",
    ]
    rc, ip, _ = await run_command(cmd)
    if rc == 0 and ip:
        return ip.strip()

    # Fallback to internal IP if external is not found
    cmd[-1] = "value(networkEndpoints[0].ipAddress)"
    rc, ip, _ = await run_command(cmd)
    return ip.strip() if rc == 0 and ip else None


async def get_secret(secret_id: str = HF_SECRET_ID) -> Optional[str]:
    """Retrieves a secret from Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    try:
        response = await asyncio.to_thread(client.access_secret_version, request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception:
        return None


async def _get_formatted_startup_script(model_name: str, hf_token: str) -> str:
    """Formats the startup script with necessary values."""
    template_path = os.path.join(os.path.dirname(__file__), "startup_script_template.sh")
    try:
        with open(template_path, "r") as f:
            template = f.read()
        return template.format(
            project_id=PROJECT_ID,
            zone=ZONE,
            model_name=model_name,
            hf_token=hf_token,
        )
    except Exception as e:
        logger.error(f"Error formatting startup script: {e}")
        return f"""#!/bin/bash
echo 'Error loading template: {e}'"""


async def discover_vllm_url() -> Optional[str]:
    """Finds the URL of an ACTIVE Queued Resource vLLM service."""
    list_cmd = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "queued-resources",
        "list",
        f"--project={PROJECT_ID}",
        f"--zone={ZONE}",
        "--format=json",
    ]
    rc, stdout, _ = await run_command(list_cmd)
    if rc != 0 or not stdout:
        return None

    try:
        resources = json.loads(stdout)
        for res in resources:
            if res.get("state", {}).get("state") == "ACTIVE":
                resource_id = res.get("name", "").split("/")[-1]
                node_id = await _get_node_id(resource_id)
                if node_id:
                    ip = await _get_node_ip(node_id)
                    if ip:
                        url = f"http://{ip}:8000"
                        logger.info(f"📡 Found ACTIVE Queued Resource {resource_id} at {url}")
                        return url
    except Exception as e:
        logger.error(f"Discovery error: {e}")
    return None


async def get_vllm_client() -> AsyncOpenAI:
    """Initializes and returns an AsyncOpenAI client for the vLLM service."""
    url = await discover_vllm_url()
    if not url:
        raise Exception(f"No ACTIVE Queued Resource found in {ZONE}.")
    return AsyncOpenAI(base_url=f"{url}/v1", api_key="not-needed")


@mcp.tool()
async def get_vllm_deployment_config(
    service_name: str = "vllm-gemma4",
    model_name: str = MODEL_NAME,
    tpu_type: str = "v6e-8",
    runtime_version: str = "v2-alpha-tpuv6e",
) -> str:
    """Generates the exact `gcloud` command for manual TPU v6e deployment."""
    gcloud_command = (
        f"gcloud alpha compute tpus tpu-vm create {service_name} "
        f"--zone={ZONE} "
        f"--project={PROJECT_ID} "
        f"--accelerator-type={tpu_type} "
        f"--version={runtime_version} "
        f"--metadata=startup-script='#!/bin/bash\\n"
        f"sudo apt-get update && sudo apt-get install -y docker.io\\n"
        f"sudo docker run --name vllm-gemma4 --privileged --net=host -d -v /dev/shm:/dev/shm --shm-size 10gb "
        f"vllm/vllm-tpu:nightly vllm serve {model_name} "
        f"--max-model-len 16384 --tensor-parallel-size 8 --disable_chunked_mm_input "
        f"--max-num_batched_tokens 4096 --enable-auto-tool-choice --tool-call-parser gemma4 "
        f"--reasoning-parser gemma4' "
        f"--labels=purpose=manual-deployment"
    )
    return gcloud_command


@mcp.tool()
async def verify_model_health() -> str:
    """Runs a deep logic check with latency reporting."""
    try:
        client = await get_vllm_client()
        start_time = time.monotonic()
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello, is the model working?"}],
            model=MODEL_NAME,
            max_tokens=10,
        )
        end_time = time.monotonic()
        latency = end_time - start_time
        response_content = chat_completion.choices[0].message.content

        if response_content:
            return (
                f"✅ Model health check PASSED.\\n"
                f"Response: '{response_content[:50]}...\\n'"
                f"Latency: {latency:.2f} seconds."
            )
        else:
            return "❌ Model health check FAILED: Empty response."
    except Exception as e:
        return f"❌ Model health check FAILED: {e}"


@mcp.tool()
async def save_hf_token(token: str) -> str:
    """Securely saves a Hugging Face API token to GCP Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    secret_parent = f"projects/{PROJECT_ID}/secrets/{HF_SECRET_ID}"
    
    try:
        # Check if the secret already exists
        await asyncio.to_thread(client.get_secret, request={"name": secret_parent})
    except Exception:
        # If not, create it
        await asyncio.to_thread(
            client.create_secret,
            request={
                "parent": f"projects/{PROJECT_ID}",
                "secret_id": HF_SECRET_ID,
                "secret": {"replication": {"automatic": {}}},
            },
        )

    # Add the new version
    response = await asyncio.to_thread(
        client.add_secret_version,
        request={"parent": secret_parent, "payload": {"data": token.encode("UTF-8")}},
    )
    return f"✅ Token saved. Version: {response.name}"


# --- MCP Tools ---


@mcp.tool()
async def destroy_queued_resource(resource_id: str) -> str:
    """Safely deletes a Queued Resource and its node."""
    cmd = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "queued-resources",
        "delete",
        resource_id,
        f"--zone={ZONE}",
        f"--project={PROJECT_ID}",
        "--async",
        "--quiet",
    ]
    rc, stdout, stderr = await run_command(cmd)
    if rc != 0:
        return f"❌ Failed to delete resource {resource_id}: {stderr}"
    return f"🗑️ Deletion of {resource_id} initiated: {stdout}"


@mcp.tool()
async def manage_queued_resource(resource_id: str = "vllm-gemma4-qr") -> str:
    """Ensures the primary Queued Resource exists and cleans up redundant ones."""
    list_cmd = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "queued-resources",
        "list",
        f"--zone={ZONE}",
        f"--project={PROJECT_ID}",
        "--format=json",
    ]
    rc, stdout, stderr = await run_command(list_cmd)
    if rc != 0:
        return f"❌ Failed to list resources: {stderr}"

    try:
        resources = json.loads(stdout)
    except Exception:
        resources = []

    redundant_deleted = []
    primary_res = None

    for res in resources:
        name = res.get("name", "").split("/")[-1]
        state = res.get("state", {}).get("state", "UNKNOWN")

        if name == resource_id:
            if state in ["FAILED", "SUSPENDED"]:
                logger.info(f"Primary resource {name} is {state}. Deleting to recreate.")
                await destroy_queued_resource(name)
                redundant_deleted.append(f"{name} (Failed)")
            else:
                primary_res = res
        else:
            logger.info(f"Deleting redundant resource: {name}")
            await destroy_queued_resource(name)
            redundant_deleted.append(name)

    if not primary_res:
        token = await get_secret()
        if not token:
            return "❌ Aborted: 'hf-token' secret missing."

        # The startup_script variable was assigned but never used. Removing it to resolve linting error.
        # If startup script functionality is needed, it should be integrated into the create_cmd.

        create_cmd = [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "queued-resources",
            "create",
            resource_id,
            f"--zone={ZONE}",
            "--runtime-version=v2-alpha-tpuv6e",
            f"--node-id={resource_id}-node",
            "--provisioning-model=flex-start",
            "--max-run-duration=4h",
            "--valid-until-duration=4h",
            f"--project={PROJECT_ID}",
            "--labels=purpose=flex-start",
            "--accelerator-type=v6e-8",
        ]

        logger.info(f"Executing gcloud command: {' '.join(shlex.quote(c) for c in create_cmd)}")
        logger.debug(
            f"Attempting to create primary resource with command: {' '.join(shlex.quote(c) for c in create_cmd)}"
        )
        rc_c, _, err_c = await run_command(create_cmd)

        if rc_c != 0:
            return f"❌ Creation failed: {err_c}. Cleaned up: {redundant_deleted}"
        return f"🚀 Primary resource {resource_id} creation initiated. Cleaned up: {redundant_deleted}"

    state = primary_res.get("state", {}).get("state", "UNKNOWN")
    return f"✅ Primary resource {resource_id} is {state}. Cleaned up: {redundant_deleted}"


@mcp.tool()
async def manage_vllm_docker(resource_id: str = "vllm-gemma4-qr", action: str = "start") -> str:
    """Manages the vLLM Docker container on the TPU VM."""
    node_id = await _get_node_id(resource_id)
    if not node_id:
        return f"❌ Could not find node for resource {resource_id}. Ensure it is ACTIVE."

    commands = {
        "start": f"sudo docker start vllm-gemma4 || sudo docker run --name vllm-gemma4 --privileged --net=host -d -v /dev/shm:/dev/shm --shm-size 10gb vllm/vllm-tpu:nightly vllm serve {MODEL_NAME} --max-model-len 16384 --tensor-parallel-size 8 --disable_chunked_mm_input --max_num_batched_tokens 4096 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4",
        "stop": "sudo docker stop vllm-gemma4",
        "restart": "sudo docker restart vllm-gemma4",
        "status": "sudo docker ps -a --filter name=vllm-gemma4",
    }

    ssh_cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        node_id,
        f"--zone={ZONE}",
        f"--project={PROJECT_ID}",
        "--command",
        commands.get(action, commands["status"]),
    ]

    rc, out, err = await run_command(ssh_cmd)
    if rc != 0:
        return f"""⚠️ Docker {action} failed, but reservation {resource_id} remains safe.
Error: {err}"""
    return f"""✅ Docker {action} command executed on {node_id}.
{out}"""


@mcp.tool()
async def list_queued_resources(zone: str = ZONE) -> str:
    """Lists all Queued Resources in a specific zone."""
    cmd = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "queued-resources",
        "list",
        f"--zone={zone}",
        f"--project={PROJECT_ID}",
        "--format=table(name, state.state, node_id, accelerator_type, create_time)",
    ]
    rc, out, err = await run_command(cmd)
    if rc == 0:
        return f"""### 📋 Queued Resources in {zone}
```
{out}
```"""
    else:
        return f"❌ List failed: {err}"


@mcp.tool()
async def describe_queued_resource(resource_id: str = "vllm-gemma4-qr", zone: str = ZONE) -> str:
    """Provides detailed information about a specific Queued Resource."""
    cmd = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "queued-resources",
        "describe",
        resource_id,
        f"--zone={zone}",
        f"--project={PROJECT_ID}",
        "--format=json",
    ]
    rc, out, err = await run_command(cmd)
    if rc != 0:
        return f"❌ Describe failed: {err}"
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
    except Exception:
        return f"""### 🔍 Detail: {resource_id}
```
{out}
```"""


@mcp.tool()
async def get_reservation_status(resource_id: str = "vllm-gemma4-qr") -> str:
    """Checks the lifecycle state and expiry time of a Queued Resource."""
    # This function can be simplified if `describe_queued_resource` is sufficient
    return await describe_queued_resource(resource_id)


@mcp.tool()
async def check_tpu_availability(resource_id: str) -> str:
    """Simple check to see if a Queued Resource has reached ACTIVE state."""
    cmd = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "queued-resources",
        "describe",
        resource_id,
        f"--zone={ZONE}",
        f"--project={PROJECT_ID}",
        "--format=value(state.state)",
    ]
    rc, state, err = await run_command(cmd)
    if rc != 0:
        return f"❌ Check failed: {err}"
    is_active = state.strip() == "ACTIVE"
    return (
        f"### 🧊 TPU Availability: {resource_id}\n"
        f"- **State:** `{state.strip()}`\n"
        f"- **Available:** {'✅ Yes' if is_active else '⏳ No'}"
    )


@mcp.tool()
async def estimate_deployment_cost(
    hours: float = 1.0, tpu_type: str = "v6e", topology: str = "2x4", is_flex: bool = True
) -> str:
    """Estimates the cost of a TPU deployment."""
    rates = {"v6e": 0.15, "v5e": 0.12, "v5p": 0.60}  # Flex-start rates
    rate = rates.get(tpu_type, rates["v6e"]) * (1 if is_flex else 2)

    try:
        chips = eval(topology.replace("x", "*"))
    except Exception as e:
        logger.warning(f"Failed to parse topology string '{topology}': {e}. Using default chips=8.")
        chips = 8

    total_cost = chips * rate * hours
    return (
        f"### 💸 Estimated Cost: `${total_cost:.2f}` for `{hours}h` on `{chips}` chip `{tpu_type}` "
        f"({'Flex-start' if is_flex else 'On-demand'})."
    )


@mcp.tool()
async def get_system_status() -> str:
    """Provides a high-level dashboard of system status."""
    resources_str = await list_queued_resources()
    health = "🔴 Offline"
    url = await discover_vllm_url()
    if url:
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(f"{url}/health", timeout=2)
                if res.status_code == 200:
                    health = f"🟢 Online ({url})"
        except Exception:
            pass

    next_step = "Call `manage_queued_resource` to provision infrastructure."
    if "ACTIVE" in resources_str:
        next_step = (
            "Use `query_queued_gemma4` to interact with the model."
            if "🟢" in health
            else "Use `manage_vllm_docker` to start the service."
        )

    return f"### 🌀 System Status ({ZONE})\n- **vLLM Health:** {health}\n{resources_str}\n**👉 Next Step:** {next_step}"


@mcp.tool()
async def get_vllm_endpoint() -> str:
    """Returns the active vLLM service URL if available."""
    url = await discover_vllm_url()
    if url:
        return f"🟢 vLLM is Online at: {url}"
    return "❌ No ACTIVE Queued Resource with a reachable vLLM service found."


@mcp.tool()
async def query_queued_gemma4(prompt: str) -> str:
    """Queries the self-hosted Gemma 4 model on the active Queued Resource."""
    logger.info(f"Querying model with prompt: '{prompt[:50]}...'")
    try:
        client = await get_vllm_client()
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
        )
        response = chat_completion.choices[0].message.content or "No response from model."
        logger.info(f"Model response: '{response[:100]}...'")
        return response or "No response from model."
    except Exception as e:
        logger.error(f"Error querying model: {e}")
        return f"❌ An error occurred while querying the model: {e}"


@mcp.tool()
async def query_queued_gemma4_with_stats(prompt: str) -> str:
    """
    Queries the self-hosted Gemma 4 model and returns detailed performance statistics.

    This tool provides:
    - The full model response.
    - Time to First Token (TTFT).
    - Total generation time.
    - Tokens per second.
    """
    logger.info(f"Querying model with stats with prompt: '{prompt[:50]}...'")
    try:
        client = await get_vllm_client()

        start_time = time.monotonic()
        ttft = None
        response_content = ""
        total_tokens = 0

        stream = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
            stream=True,
        )

        async for chunk in stream:
            if ttft is None:
                ttft = time.monotonic() - start_time

            content = chunk.choices[0].delta.content
            if content:
                response_content += content
                total_tokens += 1  # Rough token count

        end_time = time.monotonic()
        total_time = end_time - start_time

        if not response_content:
            return "❌ Model returned an empty response."

        tokens_per_second = total_tokens / (total_time - ttft) if ttft and total_time > ttft else 0

        stats_report = (
            f"### 📊 Performance Stats\n"
            f"- **Time to First Token (TTFT):** `{ttft:.3f}s`\n"
            f"- **Total Generation Time:** `{total_time:.3f}s`\n"
            f"- **Tokens per Second:** `{tokens_per_second:.2f} tokens/s`\n"
            f"- **Total Tokens (approx.):** `{total_tokens}`\n"
            f"\n### 💬 Model Response\n"
            f"{response_content}"
        )

        logger.info(f"Model response with stats: TTFT={ttft:.3f}s, TotalTime={total_time:.3f}s")
        return stats_report

    except Exception as e:
        logger.error(f"Error querying model with stats: {e}")
        return f"❌ An error occurred while querying the model with stats: {e}"


@mcp.tool()
async def run_vllm_benchmark(
    resource_id: str = "vllm-gemma4-qr",
    backend: str = "vllm",
    model: str = "google/gemma-4-31B-it",
    dataset_name: str = "random",
    num_prompts: int = 100,
    random_input_len: int = 1024,
    random_output_len: int = 128,
) -> str:
    """Runs vLLM's internal benchmark tool inside the container on the TPU VM."""
    node_id = await _get_node_id(resource_id)
    if not node_id:
        return f"❌ Could not find node for resource {resource_id}. Ensure it is ACTIVE."

    benchmark_cmd = (
        f"python -m vllm.bench.benchmark_throughput "
        f"--backend {backend} "
        f"--model {model} "
        f"--dataset-name {dataset_name} "
        f"--num-prompts {num_prompts} "
        f"--input-len {random_input_len} "
        f"--output-len {random_output_len}"
    )

    ssh_cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        node_id,
        f"--zone={ZONE}",
        f"--project={PROJECT_ID}",
        "--command",
        f"sudo docker exec vllm-gemma4 {benchmark_cmd}",
    ]

    rc, out, err = await run_command(ssh_cmd, timeout=300)  # Increased timeout for benchmark
    if rc != 0:
        return f"""⚠️ Benchmark failed on {node_id}.
Error: {err}
Output: {out}"""
    return f"""✅ Benchmark completed on {node_id}:
{out}"""


@mcp.tool()
async def get_vllm_docker_logs(resource_id: str = "vllm-gemma4-qr", tail: Optional[int] = None) -> str:
    """Retrieves logs from the vLLM Docker container on the TPU VM."""
    node_id = await _get_node_id(resource_id)
    if not node_id:
        return f"❌ Could not find node for resource {resource_id}. Ensure it is ACTIVE."

    log_cmd = "sudo docker logs vllm-gemma4"
    if tail:
        log_cmd += f" --tail {tail}"

    ssh_cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        node_id,
        f"--zone={ZONE}",
        f"--project={PROJECT_ID}",
        "--command",
        log_cmd,
    ]

    rc, out, err = await run_command(ssh_cmd)
    if rc != 0:
        return f"""⚠️ Failed to get Docker logs from {node_id}.
Error: {err}"""
    return f"""✅ Docker logs from {node_id}:
{out}"""


@mcp.tool()
async def get_tpu_system_logs(
    resource_id: str = "vllm-gemma4-qr", service: str = "docker", tail: Optional[int] = None
) -> str:
    """Retrieves systemd logs for a specific service from the TPU VM."""
    node_id = await _get_node_id(resource_id)
    if not node_id:
        return f"❌ Could not find node for resource {resource_id}. Ensure it is ACTIVE."

    log_cmd = f"journalctl -u {service} -n {tail or 100}"

    ssh_cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        node_id,
        f"--zone={ZONE}",
        f"--project={PROJECT_ID}",
        "--command",
        log_cmd,
    ]

    rc, out, err = await run_command(ssh_cmd)
    if rc != 0:
        return f"""⚠️ Failed to get system logs from {node_id}.
Error: {err}"""
    return f"""✅ System logs for '{service}' from {node_id}:
{out}"""


@mcp.tool()
async def get_cloud_logging_logs(log_filter: str = 'resource.type="tpu_worker"', limit: int = 20) -> str:
    """Fetches logs from Google Cloud Logging."""
    cmd = ["gcloud", "logging", "read", log_filter, f"--project={PROJECT_ID}", f"--limit={limit}", "--format=json"]
    rc, out, err = await run_command(cmd)
    if rc != 0:
        return f"❌ Failed to fetch Cloud Logs: {err}"

    try:
        logs = json.loads(out)
        formatted_logs = "\n".join(
            [
                f"[{log_entry.get('timestamp')}] {log_entry.get('resource', {}).get('labels', {}).get('node_id', 'N/A')} - "
                f"{log_entry.get('textPayload', log_entry.get('jsonPayload', {}))}"
                for log_entry in logs
            ]
        )
        return f"### ☁️ Cloud Logs (filter: `{log_filter}`)\n```\n{formatted_logs}\n```"
    except Exception:
        return f"### ☁️ Cloud Logs (raw)\n```\n{out}\n```"


@mcp.tool()
async def get_model_details() -> str:
    """
    Retrieves detailed information about the running model, vLLM engine, and versions.

    Provides a verbose report including:
    - Model ID and details from the vLLM engine.
    - vLLM version and build information.
    - Health status.
    - Key performance metrics.
    """
    url = await discover_vllm_url()
    if not url:
        return "❌ No ACTIVE Queued Resource with a reachable vLLM service found."

    report = f"### 🧩 Model & vLLM Engine Details ({url})\n\n"

    async with httpx.AsyncClient(timeout=10) as client:
        # 1. Get Model Details from /v1/models
        try:
            models_res = await client.get(f"{url}/v1/models")
            if models_res.status_code == 200:
                models_data = models_res.json()
                report += "**Model Information (`/v1/models`):**\n"
                report += f"```json\n{json.dumps(models_data, indent=2)}\n```\n"
            else:
                report += f"⚠️ Could not fetch model details. Status: {models_res.status_code}\n"
        except Exception as e:
            report += f"❌ Error fetching model details: {e}\n"

        # 2. Get vLLM Version from /version
        try:
            version_res = await client.get(f"{url}/version")
            if version_res.status_code == 200:
                version_data = version_res.json()
                report += "**vLLM Version (`/version`):**\n"
                report += f"- Version: `{version_data.get('version', 'N/A')}`\n\n"
            else:
                report += f"⚠️ Could not fetch vLLM version. Status: {version_res.status_code}\n\n"
        except Exception as e:
            report += f"❌ Error fetching vLLM version: {e}\n\n"

        # 3. Get Health Status from /health
        try:
            health_res = await client.get(f"{url}/health")
            if health_res.status_code == 200:
                report += "**Health Status (`/health`):**\n- Status: `Healthy` ✅\n\n"
            else:
                report += (
                    f"**Health Status (`/health`):**\n- Status: `Unhealthy` ❌ (Code: {health_res.status_code})\n\n"
                )
        except Exception as e:
            report += f"❌ Error fetching health status: {e}\n\n"

        # 4. Get Metrics from /metrics
        try:
            metrics_res = await client.get(f"{url}/metrics")
            if metrics_res.status_code == 200:
                report += "**Key vLLM Metrics (`/metrics`):**\n"
                metrics_lines = metrics_res.text.splitlines()
                key_metrics = [
                    line
                    for line in metrics_lines
                    if "vllm_requests_running" in line
                    or "vllm_requests_swapped" in line
                    or "vllm_requests_waiting" in line
                    or "vllm_tpu_cache_usage_perc" in line
                    or "process_resident_memory_bytes" in line
                ]
                if key_metrics:
                    report += "```\n" + "\n".join(key_metrics) + "\n```\n"
                else:
                    report += "Metrics endpoint available, but no key metrics found in snippet.\n"
            else:
                report += "⚠️ Metrics endpoint not available or failed.\n"
        except Exception as e:
            report += f"❌ Error fetching metrics: {e}\n"

    return report


if __name__ == "__main__":
    mcp.run()
