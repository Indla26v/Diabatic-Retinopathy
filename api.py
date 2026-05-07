import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import base64
import requests
from typing import List
import google.generativeai as genai
import io
import time
import asyncio

# ==================== Configuration ====================
# Must match train_improved.py architecture exactly
CFG = {
    'img_size'    : 224,
    'num_classes' : 5,
    'token_dim'   : 512,
    'num_heads'   : 8,
    'num_layers'  : 2,
    'ff_dim'      : 1024,
    'tf_dropout'  : 0.15,
    'save_path'   : 'best_hybrid_model (1).pth',
}

CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# ==================== Model Definition ====================
class CNNTransformerDR(nn.Module):
    """CNN-Transformer Hybrid Model for DR Classification (EfficientNet-B3)
    Architecture matched for best_hybrid_model (1).pth"""

    def __init__(self, cfg):
        super().__init__()
        eff = tv_models.efficientnet_b3(weights=tv_models.EfficientNet_B3_Weights.DEFAULT)
        self.cnn_encoder = eff.features

        self.proj = nn.Conv2d(1536, cfg['token_dim'], kernel_size=1, bias=False)

        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg['token_dim']))
        self.pos_embed = nn.Parameter(torch.randn(1, 101, cfg['token_dim']) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg['token_dim'],
            nhead=cfg['num_heads'],
            dim_feedforward=cfg['ff_dim'],
            dropout=cfg['tf_dropout'],
            activation='gelu',
            norm_first=True,
            batch_first=True,
        )
        # Note: best_hybrid_model uses 4 layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.classifier = nn.Sequential(
            nn.Linear(cfg['token_dim'], 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, cfg['num_classes']),
        )

    def forward(self, x):
        feat = self.cnn_encoder(x)
        feat = self.proj(feat)
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).permute(0, 2, 1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
        tokens = self.transformer(tokens)
        pooled = tokens[:, 0]
        return self.classifier(pooled)

# ==================== App Setup ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model
    if os.path.exists(CFG['save_path']):
        model = CNNTransformerDR(CFG).to(DEVICE)
        checkpoint = torch.load(CFG['save_path'], map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint)
        model.eval()
        print("Model loaded successfully!")
    else:
        print(f"Warning: Model file not found at {CFG['save_path']}")
    yield
    model = None

app = FastAPI(title="DR Screening API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Utility Functions ====================
def apply_clahe(img_np):
    """Apply CLAHE enhancement to image"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv2.split(img_np)
    enhanced = [clahe.apply(ch) for ch in channels]
    return cv2.merge(enhanced)

def image_to_base64(img_np):
    """Convert numpy image to base64 data URI"""
    _, buffer = cv2.imencode('.png', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')

def compute_severity_score(probs_np):
    """
    Compute a continuous severity score (0.0 - 4.0) as the expected value
    of the class index weighted by softmax probabilities.
    
    This is the metric that AUC measures — the ability to rank patients
    by relative severity, not the exact class prediction.
    
    Score interpretation:
      0.0 = Definitely No DR
      1.0 = Mild-level severity
      2.0 = Moderate-level severity
      3.0 = Severe-level severity
      4.0 = Proliferative-level severity
    """
    class_indices = np.arange(len(probs_np))
    return float(np.dot(probs_np, class_indices))

def get_risk_tier(severity_score):
    """Map severity score to a risk tier for clinical triage"""
    if severity_score < 0.5:
        return "low"
    elif severity_score < 1.5:
        return "moderate"
    elif severity_score < 2.5:
        return "elevated"
    elif severity_score < 3.5:
        return "high"
    else:
        return "critical"

import time
import asyncio

# In-memory cache for explanations
explanation_cache = {}

def get_gemini_explanation(image_bytes, predicted_class_name, confidence):
    """Call Gemini API for detailed ophthalmic analysis of retinal findings"""
    try:
        # Create a simple cache key from image hash and prediction
        cache_key = f"{hash(image_bytes)[:10]}_{predicted_class_name}_{confidence:.1f}"
        
        # Check cache first
        if cache_key in explanation_cache:
            return explanation_cache[cache_key]
        
        api_key = "AIzaSyD3TesZUJrJi8a1k47eJtAkZfOcsaB466E"
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        image_pil = Image.open(io.BytesIO(image_bytes))
        
        confidence_pct = float(confidence) * 100.0
        
        prompt = f"""You are a senior ophthalmologist analyzing a retinal fundus image for diabetic retinopathy (DR).

The AI model predicted: {predicted_class_name} (confidence: {confidence_pct:.0f}%)

Please provide a detailed clinical analysis:
1. **Severity Classification**: Confirm or revise the {predicted_class_name} classification.
2. **Key Retinal Findings**: Identify and describe:
   - Microaneurysms (small red dots)
   - Haemorrhages/bleeding areas (larger red patches)
   - Abnormal blood vessel growth (neovascularization)
   - Hard exudates (yellow/white deposits)
   - Cotton-wool spots (white fluffy areas)
3. **Clinical Significance**: Explain what these findings mean for DR severity.
4. **Recommended Action**: Suggest next steps (monitoring, urgent referral, etc.).

Be concise but comprehensive. Focus on what you observe in the image."""
        
        # Retry logic with exponential backoff
        max_retries = 2
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content([prompt, image_pil])
                result = response.text
                # Cache the result
                explanation_cache[cache_key] = result
                return result
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        return f"API quota exceeded. Please try again later. {predicted_class_name} detected with {confidence_pct:.0f}% confidence."
                else:
                    print(f"Gemini error (attempt {attempt+1}): {error_str}")
                    if attempt == max_retries - 1:
                        return f"Unable to retrieve AI analysis. {predicted_class_name} detected with {confidence_pct:.0f}% confidence."
                    
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        return f"Unable to retrieve AI analysis. {predicted_class_name} detected."

# ==================== Screening Endpoint (Batch) ====================
@app.post("/screen")
async def screen_endpoint(files: List[UploadFile] = File(...)):
    """
    Accept multiple retinal images and return them ranked by severity.
    Uses expected-value severity scoring for robust relative ordering.
    Validation AUC: 95.44% — the model reliably ranks patients.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    if len(files) < 1:
        raise HTTPException(status_code=400, detail="Please upload at least 1 image")

    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 images per batch")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.Resize((CFG['img_size'], CFG['img_size'])),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    results = []

    for idx, file in enumerate(files):
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                results.append({
                    "index": idx,
                    "filename": file.filename,
                    "error": "Invalid image file",
                })
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Generate thumbnail for display (resize for bandwidth)
            thumb_size = 400
            h, w = image.shape[:2]
            scale = thumb_size / max(h, w)
            thumb = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            thumbnail_b64 = image_to_base64(thumb)

            # CLAHE enhancement
            clahe_image = apply_clahe(image)
            image_pil = Image.fromarray(clahe_image)
            image_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    output = model(image_tensor)
                    probs = torch.softmax(output.float(), dim=1)

            probs_np = probs.cpu().numpy()[0]
            predicted_class = int(np.argmax(probs_np))
            severity_score = compute_severity_score(probs_np)
            risk_tier = get_risk_tier(severity_score)
            
            # Save raw bytes temporarily to fetch explanations only for top patients later
            _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            results.append({
                "index": idx,
                "filename": file.filename,
                "thumbnail": thumbnail_b64,
                "predicted_class": predicted_class,
                "predicted_label": CLASS_NAMES[predicted_class],
                "severity_score": round(severity_score, 3),
                "risk_tier": risk_tier,
                "probabilities": probs_np.tolist(),
                "_img_bytes": img_encoded.tobytes()
            })
        except Exception as e:
            results.append({
                "index": idx,
                "filename": file.filename,
                "error": str(e),
            })

    # Sort by severity score descending (most severe first)
    valid_results = [r for r in results if "error" not in r]
    error_results = [r for r in results if "error" in r]
    valid_results.sort(key=lambda x: x["severity_score"], reverse=True)

    # Assign rank and fetch explanations for all patients
    for rank, r in enumerate(valid_results, 1):
        r["rank"] = rank
        
        # Add small delay between API calls to avoid rate limiting
        if rank > 1:
            time.sleep(0.5)
        
        # Get detailed clinical explanation for each patient
        r["gemini_explanation"] = get_gemini_explanation(
            r["_img_bytes"], 
            r["predicted_label"], 
            float(r["probabilities"][r["predicted_class"]])
        )
            
        del r["_img_bytes"]  # Remove from final response

    return {
        "total_images": len(files),
        "screened": len(valid_results),
        "errors": len(error_results),
        "model_val_auc": 0.9544,
        "ranked_results": valid_results,
        "error_results": error_results,
    }


# ==================== Single Predict Endpoint (kept for backward compat) ====================
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_b64 = image_to_base64(image)

    clahe_image = apply_clahe(image)
    clahe_b64 = image_to_base64(clahe_image)

    image_pil = Image.fromarray(clahe_image)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.Resize((CFG['img_size'], CFG['img_size'])),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    image_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            output = model(image_tensor)
            probs = torch.softmax(output.float(), dim=1)

    probs_np = probs.cpu().numpy()[0]
    predicted_class = int(np.argmax(probs_np))
    severity_score = compute_severity_score(probs_np)

    return {
        "predicted_class": predicted_class,
        "predicted_label": CLASS_NAMES[predicted_class],
        "severity_score": round(severity_score, 3),
        "risk_tier": get_risk_tier(severity_score),
        "probabilities": probs_np.tolist(),
        "original_image": original_b64,
        "clahe_image": clahe_b64
    }
