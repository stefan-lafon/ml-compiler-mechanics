# 04_sensitivity_analysis.ipynb: Layer Sensitivity Analysis

This notebook answers the most critical question in advanced quantization: **Which layers can we crush to 4-bits, and which ones will break the model?**

If you try to convert an entire Neural Network to INT4, the accuracy usually tanks. But not all layers are created equal. Some are **Highly Sensitive** to noise, while others are extremely **Robust**.

We build a **Sensitivity Scanner** that iterates through ResNet18, testing each layer one by one to see how much damage it causes when quantized.

### The Strategy: The Perturbation Test
We don't guess. We test:
1.  **Isolate:** We loop through every layer in the model.
2.  **Attack:** We force *only* that specific layer into INT4 simulation, while keeping the rest of the network in high-precision FP32.
3.  **Measure:** We calculate the **KL Divergence** (Drift) between the original predictions and the perturbed ones.

### The Results (Visualized)
The output is a "Sensitivity Profile" of the network.

![Sensitivity Analysis Chart](assets/sensitivity.png)

*(Note: Run the notebook to generate this profile for your specific hardware/model)*

**What to look for:**
You will consistently see three things in the chart:
* **The Early Spikes (Sensitive):** The very first Convolutional layer is almost always sensitive. If you corrupt the raw input features, the error cascades through the whole network.
* **The Downsample Trap:** Look at the spikes in the middle. These usually correspond to `1x1` downsampling layers in the ResNet residual blocks. They are structural bottlenecks and often cannot handle low precision.
* **The Valleys (Robust):** The vast majority of the deep `3x3` convolutions are flat. These are robust. You can aggressively quantize them to INT4 without hurting accuracy.

### TL;DR
* **Don't treat all layers the same:** A flat policy (e.g., "Everything INT8") is inefficient. A Mixed-Precision policy is optimal.
* **The 15% Target:** By keeping the sensitive layers at INT8 and crushing the robust ones to INT4, we can usually get the model size down to ~15% of the original, with near-zero accuracy loss.

### Quick Start
1.  **No Downloads Needed:** The notebook generates synthetic calibration data so you can run it immediately without downloading ImageNet.
2.  **Run All Cells:** The script will output a final "Proposed Policy" showing exactly which layers should be 8-bit vs 4-bit.