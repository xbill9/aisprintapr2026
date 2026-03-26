## Sample Project Ideas

### vLLM
1. Guide to running basic Offline Batch Inference & Online API Server on TPU.
2. Guide to benchmarking (`benchmark_throughput.py` in the repo) to test TPU performance vs GPU.
3. Create a "Cheat Sheet" for Model Size vs TPU Chip and Cost (e.g. Llama-70B -> v5e-8, Llama-8B -> v5e-4).
4. Video/Blog on how to add new Pallas kernels to the infrastructure and tune.
5. Guide to autoscaling vLLM on GKE with TPU nodes (Autopilot or Standard).

### PyTorch on XLA/TPU
1. Intro to Pallas, and how it works.
2. How to write a Pallas kernel.
3. How to profile with xprof, including how to detect issues like performance or memory bottlenecks, and how to determine that a TPU is being fully utilized.

### Keras
1. Getting started on Keras Remote, a new way to deploy workloads with Keras: [Keras Remote](https://github.com/keras-team/remote)

### TPU
1. Comparison between TPUs and GPUs and what workloads are best suited for each type of chip.
2. An explanation of the different TPU architecture versions and why you might use one vs another (v4 - v7, including lite versions).
