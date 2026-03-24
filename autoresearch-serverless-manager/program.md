# Researcher Program: Optimize JAX/XLA Performance

## Goal
Improve the `METRIC_SCORE` (iterations/sec) of `train.py` on an **NVIDIA L4 GPU (24GB VRAM)** on **Cloud Run Jobs**.

## Target Optimizations
1.  **XLA Fusion:** Use `jax.jit` efficiently or set `XLA_FLAGS` to enable aggressive fusion.
2.  **Kernel Fusing:** Rewrite the `model_fn` loop to allow XLA to fuse operations more effectively.
3.  **BF16 Precision:** Convert operations to `bfloat16` to leverage L4 Tensor Cores.

## Research Loop
1.  **Analyze** the current `train.py` performance.
2.  **Edit** `train.py` (e.g., adding `XLA_FLAGS`, changing precision, or refactoring the JIT loop).
3.  **Run** `python train.py`.
4.  **Parse** the `METRIC_SCORE`.
5.  **If** the new score is higher:
    *   **Commit** the change to the research branch.
    *   **Log** the improvement rationale.
6.  **Else**:
    *   **Revert** to the previous best version.

## Constraints
*   **Time Limit:** 5 minutes per experiment.
*   **Budget:** Do not exceed 20 experiment cycles (~$2 total).
