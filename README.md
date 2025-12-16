# ML Compiler Mechanics: From Graph to Silicon

## Mission
Modern machine learning frameworks excel at abstraction, allowing engineers to define complex architectures without worrying about memory alignment or kernel fusion. However, deploying efficient models on specialized hardware—like Tensor SOCs—requires piercing this veil of abstraction.

This repository is a collection of interactive case studies designed to bridge the gap between high-level model design and low-level system performance. It explores the "hardware-software contract," demonstrating how compiler decisions (like XLA fusion) and arithmetic precision (quantization) directly impact latency, throughput, and memory bandwidth.

## Why This Matters
For ML practitioners and systems engineers, understanding these mechanics is no longer optional. As models grow larger and compute budgets tighter, the ability to reason about **arithmetic intensity**, **memory hierarchies**, and **compiler IR** becomes the primary lever for optimization.

## Modules

<table>
  <tr>
    <td width="60%" valign="top">
      <h3>1. The Operator Fusion Advantage (XLA/JAX)</h3>
      <ul>
        <li><b>Focus:</b> Graph-Level Optimization & Memory Bandwidth.</li>
        <li><b>Concept:</b> Analysis of how the XLA compiler fuses element-wise operations (like GELU or Swish) into single kernels to reduce HBM access penalties.</li>
        <li><b>Key Artifact:</b> Roofline analysis comparing Eager Execution vs. XLA JIT.</li>
      </ul>
    </td>
    <td width="40%" valign="top">
      <br>
      <img src="images/fusion_medium.png" alt="Medium Fusion Roofline Plot" width="100%">
      <b>Eager Execution (Red)</b> collapses on medium-complexity workloads, while <b>JIT (Green)</b> saturates the hardware bandwidth.
    </td>
  </tr>

  <tr>
    <td width="60%" valign="top">
      <h3>2. Quantization & Precision (TFLite)</h3>
      <ul>
        <li><b>Focus:</b> Arithmetic Efficiency & Model Footprint.</li>
        <li><b>Concept:</b> A comparative study of FP32 vs. INT8 execution. This module explores how reducing precision impacts model size and enables the usage of high-throughput integer pipelines on NPU architectures.</li>
        <li><b>Key Artifact:</b> Latency and size benchmarks on standard vision backbones.</li>
      </ul>
    </td>
    <td width="40%" valign="top">
      <br>
      <img src="images/quantization_confidence.png" alt="Quantization Confidence Drift" width="100%">
      <b>4x size reduction</b> (13MB → 3.6MB) with negligible accuracy loss and minimal confidence drift (<5%) on standard test images.
    </td>
  </tr>

  <tr>
    <td width="60%" valign="top">
      <h3>3. Pruning & The Sparsity Paradox</h3>
      <ul>
        <li><b>Focus:</b> Execution Efficiency vs. Theoretical FLOPs.</li>
        <li><b>Concept:</b> A counter-intuitive benchmark revealing that setting weights to zero (Unstructured Pruning) often yields <b>zero speedup</b> on standard hardware. Real gains require modifying tensor geometry (Structured Pruning).</li>
        <li><b>Key Artifact:</b> "The Sparsity Paradox" plot comparing dense, masked, and geometrically pruned inference times.</li>
      </ul>
    </td>
    <td width="40%" valign="top">
      <br>
      <img src="images/sparsity.png" alt="Sparsity Paradox Benchmark" width="100%">
      <b>Unstructured Pruning (Red)</b> fails to accelerate inference despite 50% sparsity. <b>Structured Pruning (Green)</b> delivers linear speedups by physically shrinking the matrix dimensions.
    </td>
  </tr>

  <tr>
    <td width="60%" valign="top">
      <h3>4. Future Roadmap (Planned)</h3>
      <ul>
        <li><b>Sensitivity Analysis:</b> Optimal bit-width allocation (INT4 vs INT8) per layer.</li>
        <li><b>QAT from Scratch:</b> Implementing the "Straight Through Estimator" to train quantized models.</li>
        <li><b>Custom Kernels:</b> Writing OpenAI Triton kernels to bypass compiler limitations.</li>
      </ul>
    </td>
    <td width="40%" valign="center" align="center">
      <i>[Coming Soon]</i>
    </td>
  </tr>
</table>