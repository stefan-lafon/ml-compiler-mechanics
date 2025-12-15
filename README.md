# ML Compiler Mechanics: From Graph to Silicon

## Mission
Modern machine learning frameworks excel at abstraction, allowing engineers to define complex architectures without worrying about memory alignment or kernel fusion. However, deploying efficient models on specialized hardware—like Tensor SOCs—requires piercing this veil of abstraction.

This repository is a collection of interactive case studies designed to bridge the gap between high-level model design and low-level system performance. It explores the "hardware-software contract," demonstrating how compiler decisions (like XLA fusion) and arithmetic precision (quantization) directly impact latency, throughput, and memory bandwidth.

## Why This Matters
For ML practitioners and systems engineers, understanding these mechanics is no longer optional. As models grow larger and compute budgets tighter, the ability to reason about **arithmetic intensity**, **memory hierarchies**, and **compiler IR** becomes the primary lever for optimization.

## Modules

### 1. The Operator Fusion Advantage (XLA/JAX)
* **Focus:** Graph-Level Optimization & Memory Bandwidth.
* **Concept:** Analysis of how the XLA compiler fuses element-wise operations (like GELU or Swish) into single kernels to reduce HBM access penalties.
* **Key Artifact:** Inspection of HLO (High-Level Optimizer) IR to visualize the "Fusion" instruction in practice.

### 2. Quantization & Precision (TFLite)
* **Focus:** Arithmetic Efficiency & Model Footprint.
* **Concept:** A comparative study of FP32 vs. INT8 execution. This module explores how reducing precision impacts model size and enables the usage of high-throughput integer pipelines on NPU architectures.
* **Key Artifact:** Latency and size benchmarks on standard vision backbones.

### 3. Future Roadmap (Planned)
* **Memory Layouts:** Investigating the impact of NHWC vs. NCHW formats on vectorization efficiency.
* **Sparsity:** Techniques for exploiting zero-weight structures to skip compute cycles.
* **Custom Kernels:** Introduction to writing custom ops when the compiler cannot infer the optimal path.