# TPU Performance Analyst

This prototype provides deep-dive tools for inspecting and optimizing TPU performance during the TPU Sprint.

## Overview
Understanding the lower-level execution of machine learning models on TPUs is key to achieving peak performance. This analyst tool provides visibility into:
- **XLA HLO (High Level Optimizer):** Visualizing the intermediate representation of the model after graph compilation but before hardware-specific mapping.
- **TPU HBM (High Bandwidth Memory):** Monitoring memory usage, fragmentation, and potential bottlenecks in data transfer to/from the TPU cores.

## Key Concepts
- **Operation Fusing:** XLA's ability to combine multiple operations into a single kernel to reduce memory overhead.
- **Sharding Strategy:** How tensors are distributed across multiple TPU cores (important for large models).
- **HBM Latency:** Analyzing data loading patterns that might cause TPU stalls.
