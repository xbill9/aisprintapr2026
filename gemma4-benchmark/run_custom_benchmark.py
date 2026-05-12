import argparse
import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List
import httpx
import pandas as pd
import os

class ContextBenchmark:
    def __init__(self, base_url: str, model: str):
        self.base_url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self.model = model

    async def run_single_test(self, client: httpx.AsyncClient, prompt_len: int, concurrency: int = 1) -> Dict[str, Any]:
        # Generating a prompt of approx prompt_len tokens
        prompt = "test " * prompt_len

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": True,
        }

        async def send_request():
            start_time = time.perf_counter()
            ttft = -1.0
            try:
                async with client.stream("POST", self.base_url, json=payload, timeout=600) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        return {"success": False, "error": f"Status {response.status_code}: {error_text.decode()[:100]}"}

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            if ttft < 0:
                                ttft = time.perf_counter() - start_time
                            if line.strip() == "data: [DONE]":
                                break
                    
                    end_time = time.perf_counter()
                    return {
                        "success": True,
                        "ttft": ttft,
                        "total_latency": end_time - start_time
                    }
            except Exception as e:
                return {"success": False, "error": str(e)}

        start_batch = time.perf_counter()
        tasks = [send_request() for _ in range(concurrency)]
        results = await asyncio.gather(*tasks)
        end_batch = time.perf_counter()
        
        batch_duration = end_batch - start_batch
        successes = [r for r in results if r["success"]]
        
        if not successes:
            errors = [r["error"] for r in results if not r["success"]]
            return {
                "success": False,
                "target_len": prompt_len,
                "error": f"All {concurrency} requests failed. First error: {errors[0] if errors else 'Unknown'}"
            }

        avg_ttft = sum(r["ttft"] for r in successes) / len(successes)
        total_tokens = prompt_len * len(successes)
        
        return {
            "success": True,
            "target_len": prompt_len,
            "concurrency": concurrency,
            "actual_tokens": prompt_len,
            "avg_ttft": avg_ttft,
            "batch_duration": batch_duration,
            "prefill_tps": total_tokens / batch_duration if batch_duration > 0 else 0
        }

async def main():
    parser = argparse.ArgumentParser(description="Gemma 4 Comprehensive Benchmark")
    parser.add_argument("--url", type=str, required=True, help="vLLM Endpoint URL")
    parser.add_argument("--model", type=str, default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--output", type=str, default="comprehensive_benchmark_results.csv")
    args = parser.parse_args()

    concurrencies = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65500, 131000]

    benchmark = ContextBenchmark(args.url, args.model)
    results = []
    
    print(f"🚀 Starting Comprehensive Sweep on {args.url}")
    print(f"🤖 Model: {args.model}")
    
    async with httpx.AsyncClient() as client:
        for concurrency in concurrencies:
            print(f"\n👥 Testing Concurrency: {concurrency}")
            for length in lengths:
                # To prevent unnecessary long waits for configurations that will definitely fail,
                # we can skip if the previous smaller length already failed with OOM.
                # However, for a benchmark, it's better to try and record the failure.
                
                print(f"  ⏳ Testing Len: {length}...", end="", flush=True)
                res = await benchmark.run_single_test(client, length, concurrency)
                
                if res["success"]:
                    print(f" ✅ Avg TTFT: {res['avg_ttft']:.3f}s | Prefill: {res['prefill_tps']:.2f} tok/s")
                    results.append({
                        "timestamp": datetime.now().isoformat(),
                        "model": args.model,
                        "prompt_tokens": length,
                        "concurrency": concurrency,
                        "avg_ttft": res["avg_ttft"],
                        "batch_duration": res["batch_duration"],
                        "prefill_tps": res["prefill_tps"],
                        "status": "success"
                    })
                else:
                    print(f" ❌ Failed: {res['error']}")
                    results.append({
                        "timestamp": datetime.now().isoformat(),
                        "model": args.model,
                        "prompt_tokens": length,
                        "concurrency": concurrency,
                        "avg_ttft": None,
                        "batch_duration": None,
                        "prefill_tps": 0,
                        "status": "failed",
                        "error": res["error"]
                    })
                    # If it fails with a status code like 413 or 422 or certain connection errors, 
                    # we might want to stop the length sweep for this concurrency.
                    if "Status 429" in res["error"] or "Status 503" in res["error"]:
                        print("  🛑 High load detected, skipping remaining lengths for this concurrency.")
                        break

            # Save intermediate results
            pd.DataFrame(results).to_csv(args.output, index=False)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        json_file = args.output.replace(".csv", ".json")
        df.to_json(json_file, orient="records", indent=2)
        print(f"\n📊 Final results saved to {args.output} and {json_file}")
    else:
        print("\n❌ No results gathered.")

if __name__ == "__main__":
    asyncio.run(main())
