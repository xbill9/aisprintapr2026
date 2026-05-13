import argparse
import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List

import httpx
import pandas as pd


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

    async def run_sweep(self, lengths: List[int], concurrencies: List[int], output_file: str):
        print(f"🚀 Starting Multi-Dimensional Sweep on {self.base_url}")
        print(f"🤖 Model: {self.model}")
        print(f"👥 Concurrencies: {concurrencies}")
        print(f"📊 Testing {len(lengths)} length points...")

        results = []
        async with httpx.AsyncClient() as client:
            for concurrency in concurrencies:
                print(f"\n👥 Testing Concurrency: {concurrency}")
                for length in lengths:
                    res = await self.run_single_test(client, length, concurrency)
                    if res["success"]:
                        print(f"  ✅ Len: {res['target_len']} | Avg TTFT: {res['avg_ttft']:.3f}s | Batch Prefill: {res['prefill_tps']:.2f} tok/s")
                        results.append(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "model": self.model,
                                "prompt_tokens": res["target_len"],
                                "concurrency": res["concurrency"],
                                "avg_ttft": res["avg_ttft"],
                                "batch_duration": res["batch_duration"],
                                "prefill_tps": res["prefill_tps"]
                            }
                        )
                    else:
                        print(f"  ❌ Target {length} failed: {res['error']}")

        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            json_file = output_file.replace(".csv", ".json")
            df.to_json(json_file, orient="records", indent=2)
            
            print(f"\n📊 Results saved to {output_file} and {json_file}")

            print("\n### 📈 Benchmark Summary Table (Markdown)")
            # Show a pivot-like view or just the full table if manageable
            markdown_table = df[["prompt_tokens", "concurrency", "avg_ttft", "prefill_tps"]].to_markdown(index=False)
            print(markdown_table)
            return markdown_table
        else:
            print("\n❌ No successful results.")
            return None


async def main():
    parser = argparse.ArgumentParser(description="Gemma 4 Context Length Benchmark")
    parser.add_argument("--url", type=str, required=True, help="vLLM Endpoint URL")
    parser.add_argument("--model", type=str, default="google/gemma-4-26B-A4B-it-assistant")
    parser.add_argument("--max-context", type=int, default=16384, help="Max context length to test")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps")
    parser.add_argument("--concurrency", type=str, default="1", help="Comma-separated list of concurrent requests")
    parser.add_argument("--output", type=str, default="context_benchmark_results.csv")
    args = parser.parse_args()

    concurrencies = [int(c.strip()) for c in args.concurrency.split(",")]

    # Generate lengths: linear steps
    step_size = max(1, args.max_context // args.steps)
    lengths = list(range(step_size, args.max_context + 1, step_size))

    # Ensure baseline is included
    if 128 not in lengths and 128 < args.max_context:
        lengths.insert(0, 128)

    if lengths and lengths[-1] != args.max_context:
        lengths.append(args.max_context)

    benchmark = ContextBenchmark(args.url, args.model)
    await benchmark.run_sweep(lengths, concurrencies, args.output)


if __name__ == "__main__":
    asyncio.run(main())
