# Quick Start Guide - Diabetic Retinopathy Detection Interface

## 📦 What You Get

I've created **3 tools** to interact with your trained model:

1. **🌐 Streamlit Web Interface** (`streamlit_app.py`)
   - Point-and-click GUI
   - No coding required
   - Real-time predictions with visualization

2. **📖 Model Usage Guide** (`MODEL_USAGE_GUIDE.md`)
   - Complete documentation
   - How to load the model
   - Image preprocessing explained
   - Code examples

3. **🐍 Python Prediction Module** (`dr_predictor.py`)
   - Standalone prediction script
   - Easy batch processing
   - Can be imported as a module
   - Useful for automation

---

## ⚡ Quick Start (5 minutes)

### Option 1: Web Interface (Easiest)

```bash
# Install Streamlit
pip install streamlit

# Run the interface
streamlit run streamlit_app.py

# Open in browser: http://localhost:8501
```

**What you can do:**
- Upload retinal images
- Get instant DR predictions
- See confidence scores
- View probability chart

---

### Option 2: Python Script (For Batch Processing)

```python
from dr_predictor import predict_single_image, print_result

# Single image prediction
result = predict_single_image('your_image.png')
print_result(result)

# Or batch prediction
from dr_predictor import predict_multiple_images
results = predict_multiple_images(['img1.png', 'img2.png', 'img3.png'])
```

---

### Option 3: Using as Module

```python
from dr_predictor import DiabRetinoPathyPredictor

# Initialize predictor
predictor = DiabRetinoPathyPredictor('best_model.pth')

# Single image
result = predictor.predict_image('image.png', return_probs=True)

# Batch from directory
results = predictor.predict_directory('path/to/images/')

# Save results
from dr_predictor import save_results_as_json
save_results_as_json(results, 'predictions.json')
```

---

## 📋 Installation Requirements

```bash
# Create virtual environment (recommended)
python -m venv dr_env
dr_env\Scripts\activate

# Install all dependencies
pip install torch torchvision
pip install opencv-python pillow numpy pandas matplotlib seaborn scikit-learn
pip install streamlit
```

Or use requirements file:
```bash
pip install torch torchvision opencv-python pillow numpy pandas matplotlib seaborn scikit-learn streamlit
```

---

## 🎯 Understanding Model Output

### Class Levels (0-4):

| Class | Name | Severity | Action |
|-------|------|----------|--------|
| **0** | No DR | None | Regular screening |
| **1** | Mild | Early | Monitor closely |
| **2** | Moderate | Progressive | Refer to specialist |
| **3** | Severe | Advanced | Urgent referral |
| **4** | Proliferative | Critical | Immediate specialist |

### Confidence Score:

- **≥ 85%**: High confidence, reliable
- **70-85%**: Medium confidence, generally reliable
- **< 70%**: Low confidence, consider manual review

---

## 🖼️ Image Requirements

### Accepted Formats:
- PNG (.png)
- JPEG (.jpg, .jpeg)

### Image Characteristics:
- Retinal fundus photographs
- Preferably color images
- Square or rectangular
- Resolution: 100x100 to 4000x4000 pixels (auto-resized)

### Preprocessing (Automatic):

The system automatically:
1. **Loads** image from file
2. **Enhances** contrast using CLAHE
3. **Resizes** to 224×224 pixels
4. **Normalizes** using ImageNet statistics
5. **Creates** tensor for model

---

## 💾 Result Format

### Prediction Output:

```json
{
  "image_path": "retinal_image.png",
  "predicted_class": 2,
  "predicted_label": "Moderate",
  "confidence": 0.8725,
  "description": "More than just microaneurysms",
  "all_probabilities": {
    "No DR": 0.0234,
    "Mild": 0.0512,
    "Moderate": 0.8725,
    "Severe": 0.0422,
    "Proliferative DR": 0.0107
  }
}
```

---

## 🚀 Advanced Usage

### Batch Processing with CSV Export:

```python
from dr_predictor import DiabRetinoPathyPredictor
import pandas as pd

predictor = DiabRetinoPathyPredictor()

# Process directory
results = predictor.predict_directory('path/to/images/', return_probs=True)

# Convert to DataFrame
df = pd.DataFrame(results)

# Save to CSV
df.to_csv('dr_predictions.csv', index=False)
```

### Integration with Your Workflow:

```python
import glob
from dr_predictor import predict_multiple_images, save_results_as_json

# Find all images
images = glob.glob('patient_images/**/*.png', recursive=True)

# Batch predict
results = predict_multiple_images(images, return_probs=True)

# Save results
save_results_as_json(results, 'batch_results.json')
```

---

## ⚙️ Configuration

### Model Settings (in dr_predictor.py):

```python
CFG = {
    'img_size'    : 224,      # Input size
    'num_classes' : 5,        # DR severity classes
    'token_dim'   : 512,      # Transformer dimension
    'num_heads'   : 8,        # Attention heads
    'num_layers'  : 2,        # Transformer layers
    'ff_dim'      : 1024,     # Feed-forward size
    'tf_dropout'  : 0.1,      # Dropout rate
    'save_path'   : 'best_model.pth',
}
```

**Don't change these unless you have a specific reason!**

---

## 🐛 Troubleshooting

### Problem: "Model not found"
**Solution:** Make sure `best_model.pth` is in the same directory or provide the correct path.

### Problem: "Could not load image"
**Solution:** Check image format (PNG/JPG), file exists, and path is correct.

### Problem: Low confidence predictions
**Solution:** This might indicate an unclear image or atypical presentation. Consider manual review.

### Problem: CUDA out of memory
**Solution:** The model will automatically fall back to CPU. May be slower but will still work.

### Problem: Import errors
**Solution:** Ensure all packages are installed: `pip install -r requirements.txt`

---

## 📚 Learn More

For detailed information:
- **Model Architecture**: See `MODEL_USAGE_GUIDE.md`
- **Image Preprocessing**: See `MODEL_USAGE_GUIDE.md` → Image Preprocessing section
- **Making Predictions**: See `MODEL_USAGE_GUIDE.md` → Making Predictions section
- **Code Examples**: See `dr_predictor.py` for working examples

---

## ✅ Checklist Before Using

- [ ] Model checkpoint (`best_model.pth`) is in your working directory
- [ ] Python 3.7+ installed
- [ ] All dependencies installed (`pip install torch torchvision opencv-python ...`)
- [ ] Retinal images available for prediction
- [ ] GPU available (optional, CPU also works)

---

## 🎓 File Summary

| File | Purpose | Use Case |
|------|---------|----------|
| `streamlit_app.py` | Interactive web interface | Non-technical users, quick predictions |
| `dr_predictor.py` | Python module for predictions | Batch processing, automation, scripting |
| `MODEL_USAGE_GUIDE.md` | Complete documentation | Learning, understanding the model |
| `best_model.pth` | Trained model weights | **Required** - must be present |

---

## 🎯 Next Steps

1. **Install dependencies** → Run `pip install streamlit torch torchvision opencv-python pillow`
2. **Verify model file** → Check that `best_model.pth` exists in your directory
3. **Choose your method** → Web interface or Python script
4. **Test with sample image** → Try your first prediction!
5. **Read full guide** → Check `MODEL_USAGE_GUIDE.md` for details

---

## ❓ Questions?

Refer to:
- **Web Interface Help** → Sidebar in Streamlit app has detailed info
- **Code Examples** → `dr_predictor.py` has working examples
- **Model Details** → `MODEL_USAGE_GUIDE.md` has complete documentation

---

**Happy predicting! 🎯**
