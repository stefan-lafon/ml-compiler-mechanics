# 02_quantization_and_precision.ipynb: TFLite Dynamic Range Quantization

This notebook explores the "free lunch" of edge AI: **Dynamic Range Quantization**.

We take a pre-trained MobileNetV2 and crush its weights from 32-bit floats down to 8-bit integers. Mathematically, this guarantees a **4x reduction in file size**.

The real question isn't "will it be smaller?" (it will). The question is: **Did we break the model in the process?**

### The Experiment
We run a side-by-side comparison on a few standard test images (Chelsea the cat, the astronaut, etc.) to see exactly what we lose when we throw away precision.

We compare:
1.  **The Heavyweight (FP32):** The standard, unoptimized TFLite model.
2.  **The Lightweight (INT8):** We use `tf.lite.Optimize.DEFAULT`. This quantizes the weights to save disk space, but keeps the runtime math in floating point so the predictions stay stable.

### The Results (Visualized)
I've set up a plot to visualize "Confidence Drift"—basically, how much less sure the model becomes after quantization.

![Quantization Confidence Chart](assets/quantization_confidence.png)

*(Note: The chart above is from a sample run. The notebook will generate a fresh one based on your specific calibration data.)*

**How to read the chart:**
* **The Size Bar:** You'll see the file size drop from ~13MB to ~3.5MB. That's the 4x win we expected.
* **The Drift Bars:** This shows the drop in probability for the correct label.
    * *Small drop (e.g., 5%):* Totally fine. The model still knows it's an astronaut, it's just slightly less arrogant about it.
    * *Massive drop / Label Flip:* This is where quantization failed. (Spoiler: Dynamic range quantization is usually safe enough that this rarely happens).

### TL;DR
* **It's mostly safe:** Dynamic Range Quantization is the conservative choice. Because activations stay in FP32, you rarely see accuracy tank.
* **The 4x Rule:** You will almost always get a 4x size reduction. If you're deploying to mobile, this is usually the first optimization you should turn on.

### Quick Start
1.  **No GPU Required:** This runs fast on a standard CPU runtime.
2.  **Just Run It:** The script handles downloading the test images (`skimage`) and running the TFLite converter automatically.