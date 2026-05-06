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

app = FastAPI(title="DR Detection API", lifespan=lifespan)

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

# ==================== Prediction Endpoint ====================
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
    confidence = float(probs_np[predicted_class])

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probs_np.tolist(),
        "original_image": original_b64,
        "clahe_image": clahe_b64
    }
