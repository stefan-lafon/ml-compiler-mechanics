# 01_fusion.ipynb: JAX Operator Fusion & Bandwidth Analysis

This lab demonstrates how XLA’s "Operator Fusion" solves this bottleneck by merging multiple steps into a single kernel, saving those expensive trips to memory.

### What We're Simulating
I’ve set up a benchmark using `ipywidgets` to pit **Eager Execution** (standard Python dispatch) against **JIT Compilation** (XLA Fused). We’ll run three specific workloads to see where the breaks are:

1.  **Light (Memory Bound):** A simple scaling op (`x * 0.5 + 1.0`). There's barely any math here, so performance is purely limited by how fast we can read/write data.
2.  **Medium (Standard DL):** A mix of activations like `tanh` and `sin`. This mimics a typical neural network layer.
3.  **Heavy (Compute Bound):** A "heavy shader" loop. We force the GPU to do complex math for every byte it reads to see if we can hide the memory latency.

---

### The Roofline Analysis (Visualizing the Bottleneck)

When you hit **Run Profiler**, the notebook generates a Roofline plot. This is the best way to visualize your hardware's limits.

![Roofline Analysis Chart](assets/01_fusion_chart.png)
*(Note: Run the profiler in the notebook to generate the curve for your specific GPU)*

**How to read the graph:**
* **The Red Line (Eager):** You'll see this lag behind. Without fusion, JAX has to write the result of every intermediate step (like the `sin` before the `tanh`) back to VRAM, then read it again. It's inefficient.
* **The Green Line (JIT):** This should jump significantly higher, especially on the "Light" and "Medium" tests. XLA fuses everything into one loop: it reads `x` once, does all the math in the registers, and writes `y` once.
* **The Convergence:** As we switch to the "Heavy" workload, you'll notice the gap between Red and Green shrinks. This is because the GPU is finally limited by **Compute** speed, not Memory speed, so the cost of those extra memory round-trips matters less.

### TL;DR
* **Fusion is Free Speed:** For memory-bound operations (which is most of Deep Learning), JIT compilation can double or triple your effective bandwidth.
* **The Memory Wall:** We scale tensors up to hundreds of megabytes in this test to ensure we're hitting the physical limits of the GPU memory bus, not just Python overhead.

### Quick Start
1.  Open the notebook in Colab or a local GPU environment.
2.  Pick your workloads with the checkboxes.
3.  Hit **Run Profiler** and watch the throughput (GB/s) plot build in real-time.