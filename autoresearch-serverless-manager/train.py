import jax
import jax.numpy as jnp
import time
import os

# Baseline JAX training loop (MNIST-scale or larger)
def model_fn(params, x):
    # Intentional bottleneck: Non-fused operations or inefficient gather
    for _ in range(5):
        x = jnp.dot(x, params)
        x = jax.nn.relu(x)
    return x

def main():
    print("--- AutoResearch Baseline Training ---")
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1024, 1024))
    params = jax.random.normal(key, (1024, 1024))

    # Measure iteration speed
    start_time = time.time()
    
    # Run 100 iterations
    for _ in range(100):
        _ = jax.jit(model_fn)(params, x).block_until_ready()
    
    duration = time.time() - start_time
    tokens_per_sec = 100 / duration
    
    print(f"RESULT: {tokens_per_sec:.2f} iterations/sec")
    print(f"METRIC_SCORE: {tokens_per_sec}") # AutoResearch parses this tag

if __name__ == "__main__":
    main()
