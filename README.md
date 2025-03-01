# High-Resolution Cell Imaging: Weakly-Supervised Multiple-Instance Learning

## Overview
This project implements a multiple-instance learning (MIL) pipeline for high-resolution cell imaging. Given that whole-slide images are typically megapixel or gigapixel in size, we extract fixed-size patches and process them with a custom MIL model. The model uses a customized ResNet for patch-level feature extraction, positional encoding to capture spatial information, and an attention aggregator to produce an image-level prediction.

## Method
### Data Preprocessing
- **Patch Extraction:** High-resolution images (e.g., 4096×4096) are divided into fixed-size patches (e.g., 224×224). The spatial coordinates of each patch are recorded (normalized to [0,1]) to preserve the original layout.
- **Synthetic Data Generation:** To test the pipeline, we developed a synthetic data generator that:
  - Produces backgrounds using random noise and Gaussian blur.
  - Randomly adds multiple markers (circles, ellipses, or rectangles) with varying sizes, orientations, colors, and thicknesses in positive samples.
  - If at least one marker is drawn, the label is set to 1 (Presence of disease); otherwise, it is set to 0 (Absense).
  - Logs image filenames and binary labels into a CSV file.
### Model Architecture
- **Patch-Level Feature Extraction (ResNetPatchCNN):** We use a custom ResNet composed of:
  - An initial 7×7 convolution with stride 2 and max pooling.
  - Three residual layers (each built with 2 basic residual blocks) resulting in 13 convolutional layers overall.
  - Global average pooling and a fully connected layer to generate a fixed-dimensional feature vector (e.g., 128 dimensions).
- **Positional Encoding:** Instead of concatenating raw spatial coordinates, a learned positional encoding (an MLP-based encoder) maps the 2D normalized coordinates into a higher-dimensional embedding (e.g., 32 dimensions). This richer spatial representation is concatenated with the patch features.
- **MIL Aggregation:** An attention module (implemented as a two-layer MLP with tanh activation) computes a softmax over the patch embeddings to produce attention weights. These weights are used to compute a weighted sum that serves as an image-level feature vector.
- **Image-Level Classification:** The aggregated feature vector is passed through a classifier (fully connected layers ending in a sigmoid activation) to predict the image-level label.
- **Mini-Batching Within Image:** To manage memory (given the potential for >100K patches per image), patches are processed in mini-batches during the forward pass before concatenation and aggregation.
## Training and Evaluation
- **Loss Function:** The model is trained end-to-end using binary cross-entropy loss with only image-level supervision.
- **Optimizer:** We used AdamW with lr=1e-3, betas=(0.9, 0.999), and weight_decay=0.05
- **Attention Visualization:** After training, attention weights are extracted and overlaid on the original image. A custom visualization function draws bold **blue boxes** over each patch, with opacity scaled by the corresponding attention weight. This helps verify whether the model focuses on regions containing markers.
### Example
  - True Label==1, Model predicted label==1
  - Generated Image vs Attention Weights Visualization

  <img src="synthetic_8.jpg" alt="Original Image with Marker" width="250"/> <img src="attention_overlay.png" alt="Attention Overlay" width="255"/>

## Conclusion
We evaluated whether incorporating positional encoding into patch-level features, coupled with an attention-based aggregation mechanism, enables the model to detect disease-causing markers and accurately classify image-level labels. 

## References
[1] Ilse, M., Tomczak, J. M., & Welling, M. (2018). *Attention-based Deep Multiple Instance Learning*. International Conference on Machine Learning. 2018.

[2] Campanella, G., Hanna, M. G., Geneslaw, L., Miraflor, A., Silva, V. W. K., Busam, K. J., Brogi, E, Reuter, V. E., Klimstra, D. S., & Fuchs, T. J. (2019). *Clinical-grade Computational Pathology Using Weakly Supervised Deep Learning on Whole Slide Images*. Nature Medicine. 2019.


## Getting Started
### Generate Synthetic Data
```
python generate_synthetic_data.py
```
### Model Training
```
python main.py --image_dir synthetic_images --label_file synthetic_images/labels.csv --patch_size 224 --image_batch_size 64  --num_epochs 10 --lr 1e-3 --checkpoint_dir ./checkpoints
```
### Visualize Attention Weights
```
python visualize_attention.py
```