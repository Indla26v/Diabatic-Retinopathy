# How to Use the Trained Model for Predictions

## 📋 Table of Contents
1. [Model Overview](#model-overview)
2. [Model Architecture](#model-architecture)
3. [Loading the Model](#loading-the-model)
4. [Image Preprocessing](#image-preprocessing)
5. [Making Predictions](#making-predictions)
6. [Interpreting Results](#interpreting-results)
7. [Complete Example Code](#complete-example-code)
8. [Using the Interactive Interface](#using-the-interactive-interface)

---

## Model Overview

The trained model is a **CNN-Transformer Hybrid** architecture specifically designed for **Diabetic Retinopathy (DR) Classification**. It classifies retinal fundus images into **5 severity levels**:

| Class | Label | Description |
|-------|-------|-------------|
| 0 | **No DR** | No diabetic retinopathy detected |
| 1 | **Mild** | Only microaneurysms present |
| 2 | **Moderate** | More lesions but less than severe |
| 3 | **Severe** | Extensive hemorrhages in multiple quadrants |
| 4 | **Proliferative DR** | Neovascularization present |

---

## Model Architecture

### Key Components:

1. **Backbone: EfficientNet-B0**
   - Pre-trained on ImageNet
   - Outputs 1280 feature maps at 7×7 resolution (49 spatial tokens)
   - Total parameters: ~4.2M

2. **Projection Layer**
   - Converts 1280 features → 512 (token_dim)
   - Uses 1×1 convolution for efficiency

3. **Transformer Encoder**
   - 2 layers of multi-head attention
   - 8 attention heads
   - Feed-forward dimension: 1024
   - Dropout: 0.1
   - Activation: GELU

4. **Classification Head**
   - Linear layer: 512 → 256
   - GELU activation
   - Dropout: 0.3
   - Output layer: 256 → 5 (final class predictions)

### Configuration:
```python
CFG = {
    'img_size'    : 224,      # Input image size
    'num_classes' : 5,        # Number of DR severity classes
    'token_dim'   : 512,      # Transformer embedding dimension
    'num_heads'   : 8,        # Multi-head attention heads
    'num_layers'  : 2,        # Transformer layers
    'ff_dim'      : 1024,     # Feed-forward dimension
    'tf_dropout'  : 0.1,      # Transformer dropout
}
```

---

## Loading the Model

### Step 1: Import Required Libraries

```python
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
```

### Step 2: Define the Model Architecture

Copy the `CNNTransformerDR` class from your Venky.ipynb:

```python
class CNNTransformerDR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # CNN encoder
        eff = tv_models.efficientnet_b0(
            weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
        self.cnn_encoder = eff.features
        
        # Project 1280 -> token_dim
        self.proj = nn.Conv2d(
            1280, cfg['token_dim'],
            kernel_size=1, bias=False)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg['token_dim'],
            nhead=cfg['num_heads'],
            dim_feedforward=cfg['ff_dim'],
            dropout=cfg['tf_dropout'],
            activation='gelu',
            norm_first=True,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg['num_layers']
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(cfg['token_dim'], 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, cfg['num_classes']),
        )
    
    def forward(self, x):
        feat = self.cnn_encoder(x)      # (B, 1280, 7, 7)
        feat = self.proj(feat)           # (B, 512, 7, 7)
        
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).permute(0, 2, 1)  # (B, 49, 512)
        
        tokens = self.transformer(tokens)  # (B, 49, 512)
        pooled = tokens.mean(dim=1)        # (B, 512)
        
        return self.classifier(pooled)     # (B, 5)
```

### Step 3: Load the Checkpoint

```python
# Configuration
CFG = {
    'img_size'    : 224,
    'num_classes' : 5,
    'token_dim'   : 512,
    'num_heads'   : 8,
    'num_layers'  : 2,
    'ff_dim'      : 1024,
    'tf_dropout'  : 0.1,
}

# Determine device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Create model
model = CNNTransformerDR(CFG).to(DEVICE)

# Load saved weights
checkpoint_path = 'best_model.pth'  # Path to your saved model
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(checkpoint)

# Set to evaluation mode
model.eval()
print("✓ Model loaded successfully!")
```

---

## Image Preprocessing

### Why Preprocessing Matters:

The model expects images preprocessed in a specific way. Skipping or incorrect preprocessing will lead to poor predictions.

### Preprocessing Steps:

#### Step 1: Load and Enhance Image

```python
def apply_clahe(img_np):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    
    Purpose: Enhance contrast while preventing over-amplification
    Helps make blood vessels and lesions more visible
    
    Args:
        img_np: Image as numpy array (H, W, 3) in RGB format
    
    Returns:
        Enhanced image with applied CLAHE
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv2.split(img_np)
    enhanced = [clahe.apply(ch) for ch in channels]
    return cv2.merge(enhanced)

# Load image
image_path = 'retinal_image.png'
image = cv2.imread(image_path)  # Loads as BGR
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
image = apply_clahe(image)  # Apply enhancement
```

#### Step 2: Normalize Using ImageNet Statistics

```python
# ImageNet normalization statistics
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Create transformation pipeline
transform = T.Compose([
    T.Resize((224, 224)),      # Resize to model input size
    T.ToTensor(),              # Convert to torch tensor (0-1)
    T.Normalize(mean, std),    # Normalize using ImageNet stats
])

# Apply transformation
image_pil = Image.fromarray(image)
image_tensor = transform(image_pil)  # (3, 224, 224)
```

#### Step 3: Create Batch

```python
# Add batch dimension
image_tensor = image_tensor.unsqueeze(0)  # (1, 3, 224, 224)

# Move to device (GPU/CPU)
image_tensor = image_tensor.to(DEVICE)
```

### Complete Preprocessing Function:

```python
def preprocess_image(image_path):
    """
    Complete preprocessing pipeline for inference
    
    Args:
        image_path: Path to retinal fundus image
    
    Returns:
        image_tensor: Preprocessed image ready for model
        original_image: Original image for visualization
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    
    # Apply CLAHE enhancement
    image = apply_clahe(image)
    
    # Transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    
    image_pil = Image.fromarray(image)
    image_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
    
    return image_tensor, original_image
```

---

## Making Predictions

### Basic Prediction Function:

```python
def predict(model, image_tensor):
    """
    Make prediction on image tensor
    
    Args:
        model: CNN-Transformer model in eval mode
        image_tensor: Preprocessed image tensor (1, 3, 224, 224)
    
    Returns:
        probabilities: Class probabilities (5,)
        predicted_class: Index of predicted class (0-4)
        confidence: Confidence score for predicted class
    """
    model.eval()
    
    with torch.no_grad():
        # Use mixed precision for faster inference (optional)
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            output = model(image_tensor)  # (1, 5)
    
    # Convert to probabilities
    probs = torch.softmax(output.float(), dim=1)  # (1, 5)
    probs_np = probs.cpu().numpy()[0]  # (5,)
    
    # Get prediction
    predicted_class = np.argmax(probs_np)
    confidence = probs_np[predicted_class]
    
    return probs_np, predicted_class, confidence
```

### Full Inference Pipeline:

```python
# Load and preprocess image
image_tensor, original_image = preprocess_image('retinal_image.png')

# Make prediction
probabilities, predicted_class, confidence = predict(model, image_tensor)

# Print results
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
print(f"Predicted Class: {CLASS_NAMES[predicted_class]}")
print(f"Confidence: {confidence:.2%}")
print("\nClass Probabilities:")
for i, prob in enumerate(probabilities):
    print(f"  {CLASS_NAMES[i]:20s}: {prob:.4f} ({prob:.2%})")
```

---

## Interpreting Results

### Understanding Output:

The model outputs:
1. **5 probability scores** (one per class, sum to 1.0)
2. **Predicted class** (the one with highest probability)
3. **Confidence score** (highest probability value)

### Example Output:

```
Predicted Class: Moderate
Confidence: 87.25%

Class Probabilities:
  No DR                 : 0.0234 (2.34%)
  Mild                  : 0.0512 (5.12%)
  Moderate              : 0.8725 (87.25%)  ← Predicted
  Severe                : 0.0422 (4.22%)
  Proliferative DR      : 0.0107 (1.07%)
```

### Confidence Thresholds:

| Confidence | Interpretation | Action |
|------------|-----------------|--------|
| ≥ 85% | **High confidence** | High reliability prediction |
| 70-85% | **Medium confidence** | Generally reliable but review |
| < 70% | **Low confidence** | May need manual review or specialist |

### Clinical Interpretation:

- **Class 0 (No DR)**: Normal eye, no diabetes-related damage
- **Class 1 (Mild)**: Early stage, only microaneurysms
- **Class 2 (Moderate)**: Clear retinopathy but focal/diffuse hemorrhages
- **Class 3 (Severe)**: Severe hemorrhages, requires urgent attention
- **Class 4 (Proliferative)**: Advanced stage with neovascularization, needs specialist

---

## Complete Example Code

```python
"""
Complete example: Load model, predict on image
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image

# ==================== Configuration ====================
CFG = {
    'img_size'    : 224,
    'num_classes' : 5,
    'token_dim'   : 512,
    'num_heads'   : 8,
    'num_layers'  : 2,
    'ff_dim'      : 1024,
    'tf_dropout'  : 0.1,
}

CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 
               'Severe', 'Proliferative DR']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Model Definition ====================
class CNNTransformerDR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        eff = tv_models.efficientnet_b0(
            weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
        self.cnn_encoder = eff.features
        self.proj = nn.Conv2d(1280, cfg['token_dim'], kernel_size=1, bias=False)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg['token_dim'],
            nhead=cfg['num_heads'],
            dim_feedforward=cfg['ff_dim'],
            dropout=cfg['tf_dropout'],
            activation='gelu',
            norm_first=True,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, 
                                                  num_layers=cfg['num_layers'])
        
        self.classifier = nn.Sequential(
            nn.Linear(cfg['token_dim'], 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, cfg['num_classes']),
        )
    
    def forward(self, x):
        feat = self.cnn_encoder(x)
        feat = self.proj(feat)
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).permute(0, 2, 1)
        tokens = self.transformer(tokens)
        pooled = tokens.mean(dim=1)
        return self.classifier(pooled)

# ==================== Utility Functions ====================
def apply_clahe(img_np):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv2.split(img_np)
    enhanced = [clahe.apply(ch) for ch in channels]
    return cv2.merge(enhanced)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = apply_clahe(image)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    
    image_pil = Image.fromarray(image)
    image_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
    return image_tensor

def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output.float(), dim=1)
    
    probs_np = probs.cpu().numpy()[0]
    predicted_class = np.argmax(probs_np)
    confidence = probs_np[predicted_class]
    
    return probs_np, predicted_class, confidence

# ==================== Main Inference ====================
if __name__ == "__main__":
    # Load model
    model = CNNTransformerDR(CFG).to(DEVICE)
    checkpoint = torch.load('best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    print("✓ Model loaded!")
    
    # Preprocess image
    image_tensor = preprocess_image('retinal_image.png')
    print("✓ Image preprocessed!")
    
    # Make prediction
    probs, pred_class, confidence = predict(model, image_tensor)
    
    # Display results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Predicted Class: {CLASS_NAMES[pred_class]}")
    print(f"Confidence: {confidence:.2%}")
    print("\nProbabilities for all classes:")
    for i, prob in enumerate(probs):
        bar = "█" * int(prob * 50)
        print(f"  {CLASS_NAMES[i]:20s}: {bar} {prob:.4f}")
    print("="*50)
```

---

## Using the Interactive Interface

### Running the Streamlit App:

```bash
# Navigate to your project directory
cd c:\Users\venka\OneDrive\Desktop\SDP

# Install Streamlit if not already installed
pip install streamlit

# Run the app
streamlit run streamlit_app.py
```

### Features:

✅ **Image Upload**: Drag & drop or click to upload retinal images  
✅ **Automatic Preprocessing**: CLAHE + normalization applied automatically  
✅ **Real-time Prediction**: Get instant DR severity prediction  
✅ **Confidence Visualization**: See probabilities for all classes  
✅ **Clinical Interpretation**: Color-coded severity levels  
✅ **User-Friendly**: No coding required!

### Accessing the Interface:

- Open your browser and go to: `http://localhost:8501`
- The app runs locally on your machine
- Upload images and get instant predictions
- No internet connection needed (after initial setup)

---

## Performance Notes

### Inference Speed:
- **GPU (CUDA)**: ~100-150ms per image
- **CPU**: ~500-800ms per image

### Memory Requirements:
- **Model size**: ~17 MB
- **GPU memory needed**: ~1-2 GB
- **RAM needed**: ~4 GB

### Best Practices:

1. **Always preprocess** images exactly as specified
2. **Use the saved checkpoint** (`best_model.pth`)
3. **Never retrain** the model without proper data handling
4. **Validate predictions** with medical professionals
5. **Keep confidence scores** as indication of reliability

---

**Questions?** Refer back to the configuration, preprocessing, or prediction sections!
