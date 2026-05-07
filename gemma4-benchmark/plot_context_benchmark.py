
import pandas as pd
import termplotlib as tpl
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Plot Context Benchmark Results")
    parser.add_argument("--input", type=str, default="context_benchmark_results.csv", help="Input CSV file")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: {args.input} not found.")
        sys.exit(1)

    if "prompt_tokens" not in df.columns or "avg_ttft" not in df.columns:
        # Fallback for old single-request results if they exist
        if "ttft" in df.columns:
             df["avg_ttft"] = df["ttft"]
        else:
            print(f"Error: {args.input} does not contain required columns (prompt_tokens, avg_ttft/ttft).")
            sys.exit(1)

    prompt_tokens = df["prompt_tokens"].to_numpy()
    avg_ttft = df["avg_ttft"].to_numpy()
    prefill_tps = df["prefill_tps"].to_numpy() if "prefill_tps" in df.columns else None

    print("\n--- Context Length vs. Avg TTFT (s) ---")
    fig1 = tpl.figure()
    fig1.plot(prompt_tokens, avg_ttft, label="Avg TTFT (s)")
    fig1.show()

    if prefill_tps is not None:
        print("\n--- Context Length vs. Prefill Throughput (tok/s) ---")
        fig2 = tpl.figure()
        fig2.plot(prompt_tokens, prefill_tps, label="Prefill TPS")
        fig2.show()

if __name__ == "__main__":
    main()
