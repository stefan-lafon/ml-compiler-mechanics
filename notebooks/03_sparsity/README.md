# 03_pruning_and_sparsity.ipynb: The Sparsity Paradox

This notebook reveals an uncomfortable truth in machine learning: **Setting weights to zero does not automatically make your model faster.**

We often hear that "pruning" reduces model size and latency. But if you just random-access mask weights (Unstructured Pruning), your GPU still has to load those zeros and multiply by them.

This lab proves that unless you physically change the shape of your tensors (Structured Pruning), you won't see a speedup on standard hardware.

### The Experiment
We benchmark three scenarios to see where the performance gains actually live:

1.  **Dense Baseline (VGG11):** The standard, heavy model.
2.  **Unstructured Pruning (50% Sparse):** We zero out the smallest 50% of the weights. The model *mathematically* has fewer parameters, but *physically* occupies the same amount of memory in the GPU kernel.
3.  **Structured Pruning (Geometry Change):** We physically shrink the layer dimensions (simulating the removal of entire filters/channels).

### The Results (Visualized)
We compare the inference latency (milliseconds) across all three.

![Sparsity Analysis Chart](assets/sparsity.png)
*(Note: Run the notebook to generate this latency graph for your specific hardware)*

**The "Paradox":**
* **Red Bar (Unstructured):** You will likely see **zero speedup** compared to the baseline. In fact, due to overhead, it might even be slightly slower. The GPU is still doing the math, just with zeros.
* **Green Bar (Structured):** This is where the speed is. By reducing the width of the matrix, we reduce the actual FLOPs and memory reads required.

### TL;DR
* **Masking != Speed:** Just creating a "sparse mask" is useful for compression (storage), but it does nothing for latency unless you have specialized hardware (like Sparse Tensor Cores).
* **Shape Matters:** To get faster inference on standard GPUs/CPUs, you need **Structured Pruning** (removing whole columns/filters), not just individual weights.

### Quick Start
1.  **Use a GPU:** While this runs on CPU, the "Sparsity Paradox" is most obvious on a GPU, where the parallel compute capability hides the cost of the dense math anyway.
2.  **Run All Cells:** The notebook creates a copy of VGG11, prunes it, and runs the benchmarks live.