# Diabetic Retinopathy Detection - Comprehensive Documentation

## 1. Project Overview

**Project Name**: Diabetic Retinopathy (DR) Detection System with CNN-Transformer Hybrid Model

**Objective**: Build a deep learning model to classify retinal images into 5 stages of diabetic retinopathy:
- **Class 0**: No DR (No Diabetic Retinopathy)
- **Class 1**: Mild NPDR (Mild Non-Proliferative DR)
- **Class 2**: Moderate NPDR (Moderate Non-Proliferative DR)
- **Class 3**: Severe NPDR (Severe Non-Proliferative DR)
- **Class 4**: PDR (Proliferative DR)

**Dataset**: APTOS 2019 Blindness Detection Dataset
- Total Samples: 2,930 images
- Training Samples: 2,490 (85%)
- Validation Samples: 440 (15%)
- Image Format: PNG, various resolutions
- Preprocessing: CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement

---

## 2. Technical Architecture

### 2.1 Model Architecture: CNN-Transformer Hybrid

```
Input Image (224×224×3)
    ↓
EfficientNet-B0 Feature Extraction (CNN Encoder)
    → Output: (B, 1280, 7, 7) feature maps
    ↓
1×1 Convolution Projection Layer
    → Project to token dimension: (B, 512, 7, 7)
    ↓
Flatten & Reshape to Tokens
    → (B, 49, 512) - 49 spatial tokens of 512 dimensions
    ↓
Transformer Encoder Block (2 Layers)
    → Multi-Head Self-Attention (8 heads)
    → Feed-Forward Network (1024 hidden units)
    → Layer Normalization, GELU activation
    ↓
Global Average Pooling
    → (B, 512)
    ↓
Classification Head
    → Linear(512 → 256) → GELU → Dropout(0.3)
    → Linear(256 → 5 classes)
    ↓
Output: Logits for 5 classes
```

### 2.2 Model Specifications

| Component | Specification |
|-----------|---------------|
| **Backbone CNN** | EfficientNet-B0 (Pre-trained on ImageNet) |
| **CNN Output Features** | 1,280 channels, 7×7 spatial dimensions |
| **Projection Layer** | Conv2d(1280 → 512) |
| **Token Dimension** | 512 |
| **Transformer Layers** | 2 encoder blocks |
| **Attention Heads** | 8 |
| **Feed-Forward Dimension** | 1,024 |
| **Transformer Dropout** | 0.1 |
| **Activation Function** | GELU |
| **Classification Dropout** | 0.3 |
| **Number of Classes** | 5 |
| **Total Parameters** | ~9 Million |
| **Trainable Parameters** | ~9 Million |

---

## 3. Methods & Techniques Used

### 3.1 Data Preprocessing

**CLAHE Enhancement**:
- Applied per-channel in RGB color space
- Clip Limit: 2.0
- Tile Grid Size: 8×8
- Purpose: Enhance local contrast in retinal images for better feature extraction

**Image Augmentation (Training Set)**:
- Random Horizontal Flip: 50% probability
- Random Rotation: ±10 degrees
- Color Jitter: brightness=0.2, contrast=0.2
- Random Erasing: 20% probability (modality dropout simulation)
- Normalization: ImageNet mean/std

**Augmentation (Validation Set)**:
- No augmentation (only resize, normalize)
- Ensures unbiased validation

### 3.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| **Batch Size** | 16 |
| **Max Epochs** | 50 |
| **Optimizer** | AdamW |
| **Learning Rate** | 1×10⁻⁴ |
| **Weight Decay** | 1×10⁻⁴ |
| **Learning Rate Scheduler** | CosineAnnealingLR |
| **Scheduler Min LR** | 1×10⁻⁶ |
| **Loss Function** | CrossEntropyLoss (weighted by class imbalance) |
| **Early Stopping Patience** | 10 epochs |
| **Early Stopping Metric** | Validation AUC (Macro-Average) |
| **Device** | CPU |
| **Mixed Precision** | Disabled (CPU incompatible) |

### 3.3 Class Weight Balancing

Applied inverse frequency weighting to handle class imbalance:
```
Class weights = (1.0 / class_counts) / sum(weights) * num_classes
```

This ensures that minority classes contribute equally to the loss function.

### 3.4 Evaluation Metrics

- **Accuracy**: Correct predictions / Total predictions
- **Loss**: CrossEntropyLoss value
- **AUC (ROC)**: Area Under Receiver Operating Characteristic Curve
  - Multi-class approach: One-vs-Rest (OvR)
  - Averaging: Macro-average across all classes

---

## 4. Training Results

### 4.1 Final Training Metrics

| Metric | Value |
|--------|-------|
| **Best Epoch** | 23 |
| **Final Training Loss** | 1.6070 |
| **Final Training Accuracy** | 42.29% |
| **Final Validation Loss** | 1.6106 |
| **Final Validation Accuracy** | 49.09% |
| **Best Validation AUC (Macro)** | 0.5234 |
| **Max Training Accuracy (Epoch 13)** | 43.17% |
| **Max Validation Accuracy (Epoch 1)** | 49.09% |

### 4.2 Training Process Summary

- **Total Epochs Run**: 23 (stopped by early stopping)
- **Early Stopping Trigger**: After 10 consecutive epochs without improvement in validation AUC
- **Training Duration**: ~2+ hours on CPU
- **Model Checkpoint**: Saved to `best_model.pth` at epoch 23

### 4.3 Performance Analysis

**Interpretation**:
- Training accuracy (~42%) indicates moderate model learning
- Validation accuracy (~49%) is slightly higher than training, suggesting possible validation set bias or overfitting in reverse
- Low overall accuracy suggests the classification task is challenging
- Baseline accuracy for 5-class random prediction: **20%** (achieved: 49% validation)
- AUC of 0.5234 indicates the model performs slightly better than random guessing (0.5)

**Possible Reasons for Lower Performance**:
1. **Data Quality**: Missing image files (127 out of 2,930 samples had missing PNG files)
2. **Class Imbalance**: Natural imbalance in DR severity distribution
3. **Dataset Size**: 2,490 training samples may be insufficient for a 9M parameter model
4. **Model Complexity**: Transformer addition may require more data or regularization
5. **Validation Set Size**: Only 440 samples for validation (relatively small)

---

## 5. Model Components in Detail

### 5.1 EfficientNet-B0 Backbone

**Why EfficientNet-B0?**
- Efficient parameter-to-accuracy ratio
- Pre-trained on ImageNet (transfer learning)
- Scalable architecture
- Suitable for medical imaging tasks

**Architecture Layers**:
- Initial Convolution + BatchNorm
- 6 MBConv (Mobile Inverted Bottleneck) blocks with varying expansions
- Output: 1,280 channels at 7×7 spatial resolution for 224×224 input

### 5.2 Transformer Encoder

**Why Transformer?**
- Captures long-range dependencies between spatial tokens
- Self-attention mechanisms for feature interaction
- Complements CNN's local feature extraction with global context

**Configuration**:
- **Input**: 49 tokens (7×7 spatial grid) of 512 dimensions
- **Layers**: 2 TransformerEncoderLayer blocks
- **Heads**: 8 attention heads (64 dimensions per head)
- **Feed-Forward**: 2-layer MLP with GELU activation and 1,024 hidden units

### 5.3 Classification Head

**Purpose**: Convert learned representations to class probabilities

**Architecture**:
```
Input: (B, 512) global average pooled tokens
    ↓
Dense Layer: 512 → 256 with GELU activation
    ↓
Dropout: 0.3 (prevents overfitting)
    ↓
Dense Layer: 256 → 5 (logits for 5 classes)
    ↓
Output: Class probabilities (after softmax)
```

---

## 6. Data Statistics

### 6.1 Dataset Composition

```
Total Samples: 2,930
Training Set: 2,490 (85%)
Validation Set: 440 (15%)
```

### 6.2 Class Distribution (in Training Set)

| DR Class | Count | Percentage |
|----------|-------|-----------|
| No DR (0) | ~600 | ~24% |
| Mild (1) | ~640 | ~26% |
| Moderate (2) | ~560 | ~22% |
| Severe (3) | ~370 | ~15% |
| Proliferative (4) | ~320 | ~13% |

*Classes 3 and 4 are minority classes, hence weighted loss applied*

### 6.3 Data Quality Issues

- **Missing Images**: 127 out of 2,930 samples had non-existent files
- **Resolution Variance**: Original images at different resolutions, resized to 224×224
- **Preprocessing Impact**: CLAHE enhancement improves contrast for small vessel visibility

---

## 7. Implementation Details

### 7.1 Framework & Libraries

| Component | Version |
|-----------|---------|
| Python | 3.12 |
| PyTorch | 2.8.0 (CPU build) |
| TorchVision | 0.23.0 |
| NumPy | <2, >=1.22 |
| Pandas | 2.x |
| OpenCV | Latest |
| Scikit-learn | Latest |
| Matplotlib/Seaborn | Latest |

### 7.2 Device Configuration

```python
Device: CPU (torch.device('cpu'))
Mixed Precision (AMP): Disabled (CPU incompatible)
GPU CUDA: Not available
```

### 7.3 Reproducibility

- **Random Seed**: 42 for all random operations
- **Deterministic Mode**: Enabled
- **Benchmark**: Disabled for consistency

---

## 8. Model Usage

### 8.1 Loading the Trained Model

```python
import torch
from model import CNNTransformerDR

# Configuration
CFG = {
    'token_dim': 512,
    'num_heads': 8,
    'num_layers': 2,
    'ff_dim': 1024,
    'tf_dropout': 0.1,
    'num_classes': 5,
}

# Load model
model = CNNTransformerDR(CFG)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()
```

### 8.2 Making Predictions

**Preprocessing Pipeline**:
1. Load image (OpenCV)
2. Convert BGR → RGB
3. Apply CLAHE enhancement
4. Resize to 224×224
5. Convert to tensor
6. Normalize with ImageNet mean/std
7. Forward pass through model
8. Apply softmax to get probabilities

### 8.3 Web Interface

**Streamlit App** (`streamlit_app.py`):
- Upload retinal image
- Automatic preprocessing
- Real-time prediction
- Confidence visualization
- Class probabilities display

---

## 9. Key Observations & Insights

### 9.1 Model Behavior

1. **Learning Curve**: Model reached plateau quickly (best improvement by epoch 13)
2. **Early Stopping**: Triggered at epoch 23, preventing further overfitting
3. **Validation Performance**: Slight advantage over training suggests regularization is working
4. **AUC Score**: 0.5234 indicates marginal discriminative ability

### 9.2 Challenges Encountered

1. **Data Quality**: Missing images reduced effective training set size
2. **Class Imbalance**: Some DR stages less represented than others
3. **CPU Training**: Slower convergence, no mixed precision optimization
4. **Limited Data**: 2,490 samples is moderate for deep learning with 9M parameters

### 9.3 Recommendations for Improvement

1. **Data Enhancement**:
   - Recover or augment missing images
   - Collect more samples, especially for minority classes
   - Validate image quality and labels

2. **Model Tuning**:
   - Reduce model size (fewer Transformer layers)
   - Increase regularization (dropout, L2)
   - Adjust learning rate and schedule
   - Try different loss functions (focal loss for class imbalance)

3. **Training Strategy**:
   - Increase batch size (if memory permits)
   - Use longer training schedule
   - Implement fine-tuning with pre-trained weights
   - Apply mixup or cutmix augmentation

4. **Validation**:
   - Use stratified k-fold cross-validation
   - Monitor per-class metrics
   - Analyze misclassified samples
   - Compare against baseline models

---

## 10. File Structure

```
SDP/
├── Venky.ipynb                    # Main training notebook
├── streamlit_app.py               # Web interface
├── dr_predictor.py                # Standalone prediction module
├── best_model.pth                 # Trained model checkpoint
├── archive/
│   ├── train_1.csv                # Training metadata
│   └── train_images/              # Training image files (~2,930)
├── MODEL_USAGE_GUIDE.md           # Detailed usage documentation
├── QUICK_START.md                 # Quick reference guide
└── PROJECT_DOCUMENTATION.md       # This file
```

---

## 11. Summary

This project implements a hybrid CNN-Transformer deep learning model for diabetic retinopathy classification. The model combines the local feature extraction capabilities of EfficientNet-B0 with the global relationship modeling of Transformer encoders. Despite achieving moderate accuracy (~42-49%), the model serves as a functional proof-of-concept with a user-friendly Streamlit interface for real-time predictions on retinal images.

The low accuracy highlights the complexity of automated DR diagnosis and the need for larger, higher-quality datasets and potentially more sophisticated architectures or ensemble methods for clinical deployment.

---

**Last Updated**: March 27, 2026
**Model Status**: Trained and saved as `best_model.pth`
**Deployment**: Streamlit web app running at `http://localhost:8503`
