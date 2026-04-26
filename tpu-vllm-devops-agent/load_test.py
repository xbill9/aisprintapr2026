import concurrent.futures
import time

import requests

# Configuration
URL = "http://34.46.31.222:8000/v1/completions"
MODEL = "google/gemma-4-31B-it"
PROMPT = "Explain the architecture of TPU v6e (Trillium) and why it is optimized for JAX."
NUM_REQUESTS = 20
CONCURRENCY = 4


def send_request(request_id):
    print(f"  [#{request_id}] Sending request...")
    start = time.time()
    try:
        response = requests.post(
            URL, json={"model": MODEL, "prompt": PROMPT, "max_tokens": 128, "temperature": 0.2}, timeout=60
        )
        response.raise_for_status()
        latency = time.time() - start
        print(f"  [#{request_id}] Completed in {latency:.2f}s")
        return latency
    except Exception as e:
        print(f"  [#{request_id}] Failed: {e}")
        return None


print(f"🚀 Starting load test on {URL}")
print(f"   Model: {MODEL}")
print(f"   Concurrent Workers: {CONCURRENCY}")
print(f"   Total Requests: {NUM_REQUESTS}\n")

start_test = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
    results = list(executor.map(send_request, range(1, NUM_REQUESTS + 1)))

# Filter out failures
latencies = [latency for latency in results if latency is not None]
total_test_time = time.time() - start_test

if latencies:
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    throughput = len(latencies) / total_test_time

    print("\n✅ Load Test Results:")
    print("   -------------------")
    print(f"   Total Successes:  {len(latencies)}/{NUM_REQUESTS}")
    print(f"   Total Time:       {total_test_time:.2f}s")
    print(f"   Average Latency:  {avg_latency:.2f}s")
    print(f"   P95 Latency:      {p95_latency:.2f}s")
    print(f"   Throughput:       {throughput:.2f} req/s")
else:
    print("\n❌ All requests failed. Check if the endpoint is reachable.")
