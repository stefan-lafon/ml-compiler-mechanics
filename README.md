# ML Compiler Mechanics: From Graph to Silicon

## Mission
Modern machine learning frameworks excel at abstraction, allowing engineers to define complex architectures without worrying about memory alignment or kernel fusion. However, deploying efficient models on specialized hardware—like Tensor SOCs—requires piercing this veil of abstraction.

This repository is a collection of interactive case studies designed to bridge the gap between high-level model design and low-level system performance. It explores the "hardware-software contract," demonstrating how compiler decisions (like XLA fusion) and arithmetic precision (quantization) directly impact latency, throughput, and memory bandwidth.

## Why This Matters
For ML practitioners and systems engineers, understanding these mechanics is no longer optional. As models grow larger and compute budgets tighter, the ability to reason about **arithmetic intensity**, **memory hierarchies**, and **compiler IR** becomes the primary lever for optimization.

## Modules

<table>
  <thead>
    <tr>
      <th width="20%">Key Artifact</th>
      <th width="80%">Module Description</th>
    </tr>
  </thead>
  
  <tr>
    <td align="center">
      <img src="notebooks/01_fusion/assets/fusion_medium.png" width="180px" alt="Roofline Plot">
    </td>
    <td valign="top">
      <h3><a href="notebooks/01_fusion/01_fusion.ipynb">1. The Operator Fusion Advantage (XLA/JAX)</a></h3>
      <p><b>Focus:</b> Graph-Level Optimization & Memory Bandwidth.</p>
      <p>Analysis of how the XLA compiler fuses element-wise operations (like GELU) into single kernels to reduce HBM access penalties. <i>(Comparing Eager Execution vs. XLA JIT)</i>.</p>
    </td>
  </tr>

  <tr>
    <td align="center">
      <img src="notebooks/02_quantization/assets/quantization_confidence.png" width="180px" alt="Quantization Drift">
    </td>
    <td valign="top">
      <h3><a href="notebooks/02_quantization/02_quantization_and_precision.ipynb">2. Quantization & Precision (TFLite)</a></h3>
      <p><b>Focus:</b> Arithmetic Efficiency & Model Footprint.</p>
      <p>A comparative study of FP32 vs. INT8 execution. Showcases how reducing precision impacts model size (4x reduction) and enables high-throughput integer pipelines.</p>
    </td>
  </tr>

  <tr>
    <td align="center">
      <img src="notebooks/03_sparsity/assets/sparsity.png" width="180px" alt="Sparsity Paradox">
    </td>
    <td valign="top">
      <h3><a href="notebooks/03_sparsity/03_pruning_and_sparsity.ipynb">3. Pruning & The Sparsity Paradox</a></h3>
      <p><b>Focus:</b> Execution Efficiency vs. Theoretical FLOPs.</p>
      <p>A counter-intuitive benchmark demonstrating that Unstructured Pruning (masking) yields zero speedup. Real gains require modifying tensor geometry (Structured Pruning).</p>
    </td>
  </tr>

  <tr>
    <td align="center">
      <img src="notebooks/04_sensitivity/assets/sensitivity.png" width="180px" alt="Sensitivity Scan">
    </td>
    <td valign="top">
      <h3><a href="notebooks/04_sensitivity/04_sensitivity_analysis.ipynb">4. Layer Sensitivity Analysis</a></h3>
      <p><b>Focus:</b> Mixed-Precision Quantization Strategy.</p>
      <p>A "sensitivity scan" identifies sensitive layers (require INT8) vs. more robust layers (safe for INT4), creating an optimal mixed-precision policy.</p>
    </td>
  </tr>

  <tr>
    <td align="center">
      <img src="notebooks/05_qat/assets/qat_success.png" width="180px" alt="QAT Success">
    </td>
    <td valign="top">
      <h3><a href="notebooks/05_qat/05_qat_from_scratch.ipynb">5. Quantization Aware Training (QAT)</a></h3>
      <p><b>Focus:</b> Training Dynamics & The Straight Through Estimator.</p>
      <p>Implementing the "Straight Through Estimator" (STE) to train 4-bit models from scratch, recovering accuracy where standard Post-Training Quantization fails.</p>
    </td>
  </tr>

  <tr>
    <td align="center">
      <i>Coming Soon</i>
    </td>
    <td valign="top">
      <h3>6. Future Roadmap: Custom Kernels</h3>
      <p>Writing OpenAI Triton kernels to bypass compiler limitations and execute INT4 operations efficiently on GPU.</p>
    </td>
  </tr>
</table>