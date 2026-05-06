"""
Improved DR Training Script
Fixes: wrong image paths, uses proper val set, stronger augmentation,
larger image size, gradient clipping, label smoothing, LR warmup
"""

import os, random, warnings, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as tv_models

# ===================== Seed =====================
SEED = 42
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")

# ===================== Config =====================
CFG = {
    # FIXED PATHS — images are nested inside train_images/train_images/
    'train_csv'   : r'C:\Users\venka\OneDrive\Desktop\SDP\archive\train_1.csv',
    'train_imgdir': r'C:\Users\venka\OneDrive\Desktop\SDP\archive\train_images\train_images',
    'val_csv'     : r'C:\Users\venka\OneDrive\Desktop\SDP\archive\valid.csv',
    'val_imgdir'  : r'C:\Users\venka\OneDrive\Desktop\SDP\archive\val_images\val_images',
    'save_path'   : 'best_hybrid_model (1).pth',

    # Image — Using higher resolution for Kaggle quality
    'img_size'    : 224,
    'num_classes' : 5,

    # Model — Kaggle specs
    'token_dim'   : 512,
    'num_heads'   : 8,
    'num_layers'  : 3,
    'ff_dim'      : 1024,
    'tf_dropout'  : 0.15,

    # Training
    'batch_size'  : 16, # Adjust to 16 or 8 if you run out of memory locally
    'epochs'      : 30,
    'lr'          : 3e-4,
    'weight_decay': 1e-4,
    'eta_min'     : 1e-6,
    'patience'    : 8,
    'label_smooth': 0.1,
    'grad_clip'   : 1.0,
    'warmup_epochs': 2,
    'freeze_cnn_epochs': 2,  # freeze CNN backbone for first N epochs
    'gradient_accumulation_steps': 2 # Accumulate gradients over 2 steps to simulate batch size of 32
}

CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# ===================== Load Data =====================
train_df = pd.read_csv(CFG['train_csv'])
val_df   = pd.read_csv(CFG['val_csv'])

# Verify images exist
def verify_images(df, img_dir, label=""):
    found = sum(1 for _, r in df.iterrows()
                if os.path.exists(os.path.join(img_dir, r['id_code']+'.png')))
    print(f"{label}: {found}/{len(df)} images found")
    # Filter to only existing images
    df = df[df['id_code'].apply(
        lambda x: os.path.exists(os.path.join(img_dir, x+'.png'))
    )].reset_index(drop=True)
    return df

train_df = verify_images(train_df, CFG['train_imgdir'], "Train")
val_df   = verify_images(val_df,   CFG['val_imgdir'],   "Val")

print(f"\nTrain: {len(train_df)} | Val: {len(val_df)}")
print("Train distribution:")
print(train_df['diagnosis'].value_counts().sort_index())
print("Val distribution:")
print(val_df['diagnosis'].value_counts().sort_index())

# Class weights for loss
counts = train_df['diagnosis'].value_counts().sort_index()
class_weights = (1.0 / counts).values
class_weights = class_weights / class_weights.sum() * CFG['num_classes']
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"\nClass weights: {class_weights.round(3)}")

# ===================== Dataset =====================
def apply_clahe(img_np):
    """CLAHE per-channel on RGB uint8"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv2.split(img_np)
    enhanced = [clahe.apply(ch) for ch in channels]
    return cv2.merge(enhanced)

def crop_circle(img):
    """Crop the circular retinal area and remove black borders"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 50:
            img = img[y:y+h, x:x+w]
    return img

class DRDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df        = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = os.path.join(self.image_dir, row['id_code'] + '.png')
        image = cv2.imread(path)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = crop_circle(image)
            image = apply_clahe(image)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = int(row['diagnosis'])
        return image, label

# Transforms — stronger augmentation for training
sz = CFG['img_size']
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.Resize((sz+32, sz+32)),
    T.RandomCrop((sz, sz)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=30),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    T.ToTensor(),
    T.Normalize(mean, std),
    T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

val_transform = T.Compose([
    T.Resize((sz, sz)),
    T.ToTensor(),
    T.Normalize(mean, std),
])

train_dataset = DRDataset(train_df, CFG['train_imgdir'], train_transform)
val_dataset   = DRDataset(val_df,   CFG['val_imgdir'],   val_transform)

# Weighted sampler to handle class imbalance during training
sample_weights = []
class_counts = train_df['diagnosis'].value_counts().sort_index().values
for _, row in train_df.iterrows():
    w = 1.0 / class_counts[row['diagnosis']]
    sample_weights.append(w)
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(
    train_dataset, batch_size=CFG['batch_size'],
    sampler=sampler, num_workers=0, pin_memory=False)
val_loader = DataLoader(
    val_dataset, batch_size=CFG['batch_size']*2,
    shuffle=False, num_workers=0, pin_memory=False)

print(f"\nTrain batches: {len(train_loader)}")
print(f"Val batches  : {len(val_loader)}")

# ===================== Model =====================
class class CNNTransformerDR(nn.Module):
    """EfficientNet-B3 + Transformer Encoder (Full Kaggle Architecture)"""
    def __init__(self, cfg):
        super().__init__()
        
        eff = tv_models.efficientnet_b3(
            weights=tv_models.EfficientNet_B3_Weights.DEFAULT)
        self.cnn_encoder = eff.features  # outputs 1536 channels

        self.proj = nn.Sequential(
            nn.Conv2d(1536, cfg['token_dim'], kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg['token_dim']),
            nn.GELU(),
        )

        # Learnable positional embedding (max 100 tokens)
        self.pos_embed = nn.Parameter(
            torch.randn(1, 100, cfg['token_dim']) * 0.02)

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
            encoder_layer, num_layers=cfg['num_layers'])

        self.norm = nn.LayerNorm(cfg['token_dim'])

        self.classifier = nn.Sequential(
            nn.Linear(cfg['token_dim'], 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, cfg['num_classes']),
        )

    def forward(self, x):
        feat = self.cnn_encoder(x)
        feat = self.proj(feat)
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        return self.classifier(pooled)

model = CNNTransformerDR(CFG).to(DEVICE)
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal params    : {total:,}")
print(f"Trainable params: {trainable:,}")

# ===================== Loss, Optimizer, Scheduler =====================
criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor,
    label_smoothing=CFG['label_smooth'])

optimizer = optim.AdamW(
    model.parameters(),
    lr=CFG['lr'],
    weight_decay=CFG['weight_decay'])

# Warmup + Cosine schedule
def get_lr_lambda(epoch):
    if epoch < CFG['warmup_epochs']:
        return (epoch + 1) / CFG['warmup_epochs']
    progress = (epoch - CFG['warmup_epochs']) / max(1, CFG['epochs'] - CFG['warmup_epochs'])
    return max(CFG['eta_min'] / CFG['lr'], 0.5 * (1.0 + np.cos(np.pi * progress)))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)

scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

print("Criterion  : CrossEntropyLoss (weighted + label smoothing)")
print("Optimizer  : AdamW")
print("Scheduler  : Warmup + CosineAnnealing")

# ===================== Training Functions =====================
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    optimizer.zero_grad() # Moved to start of accumulation
    
    for idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                # Scale loss by accumulation steps
                loss = loss / CFG['gradient_accumulation_steps']
            scaler.scale(loss).backward()
            
            if (idx + 1) % CFG['gradient_accumulation_steps'] == 0 or (idx + 1) == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / CFG['gradient_accumulation_steps']
            loss.backward()
            
            if (idx + 1) % CFG['gradient_accumulation_steps'] == 0 or (idx + 1) == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
                optimizer.step()
                optimizer.zero_grad()

        # Unscale loss back for reporting
        total_loss += (loss.item() * CFG['gradient_accumulation_steps']) * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            probs = torch.softmax(outputs.float(), dim=1)
            preds = probs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total

    all_probs_np = np.array(all_probs, dtype=np.float64)
    all_probs_np = all_probs_np / all_probs_np.sum(axis=1, keepdims=True)

    try:
        auc = roc_auc_score(all_labels, all_probs_np,
                            multi_class='ovr', average='macro')
    except ValueError:
        auc = 0.0

    return avg_loss, accuracy, auc, np.array(all_labels), all_probs_np


# ===================== Training Loop =====================
print("\n" + "="*90)
print("TRAINING CONFIGURATION")
print("="*90)
print(f"Device: {DEVICE}")
print(f"Image size: {CFG['img_size']}x{CFG['img_size']}")
print(f"Total Epochs: {CFG['epochs']}")
print(f"Batch Size: {CFG['batch_size']}")
print(f"Learning Rate: {CFG['lr']}")
print(f"Label Smoothing: {CFG['label_smooth']}")
print(f"Freeze CNN Epochs: {CFG['freeze_cnn_epochs']}")
print(f"Warmup Epochs: {CFG['warmup_epochs']}")
print(f"Train Samples: {len(train_df)} | Val Samples: {len(val_df)}")
print("="*90)
print(f" {'Ep':>3s}   {'T-Loss':>7s}   {'T-Acc':>6s}  {'V-Loss':>7s}   {'V-Acc':>6s}   {'V-AUC':>6s}          {'LR':>10s}    {'Time':>6s}         Status")
print("-"*90)

best_auc = 0.0
patience_counter = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}

# Initially freeze CNN backbone
for param in model.cnn_encoder.parameters():
    param.requires_grad = False
print(f"[Epoch 1-{CFG['freeze_cnn_epochs']}] CNN backbone FROZEN")

for epoch in range(1, CFG['epochs'] + 1):
    start = time.time()

    # Unfreeze CNN after freeze_cnn_epochs
    if epoch == CFG['freeze_cnn_epochs'] + 1:
        for param in model.cnn_encoder.parameters():
            param.requires_grad = True
        # Lower LR for CNN params
        optimizer = optim.AdamW([
            {'params': model.cnn_encoder.parameters(), 'lr': CFG['lr'] * 0.1},
            {'params': model.proj.parameters()},
            {'params': model.pos_embed},
            {'params': model.transformer.parameters()},
            {'params': model.norm.parameters()},
            {'params': model.classifier.parameters()},
        ], lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)
        print(f"[Epoch {epoch}] CNN backbone UNFROZEN (fine-tune LR=lr*0.1)")

    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, scaler, DEVICE)
    val_loss, val_acc, val_auc, val_labels, val_probs = validate(
        model, val_loader, criterion, DEVICE)
    scheduler.step()

    elapsed = time.time() - start
    lr = optimizer.param_groups[-1]['lr']

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_auc'].append(val_auc)

    status = ""
    if val_auc > best_auc:
        best_auc = val_auc
        patience_counter = 0
        torch.save(model.state_dict(), CFG['save_path'])
        status = f"* BEST (AUC={val_auc:.4f})"
    else:
        patience_counter += 1
        status = f"  wait {patience_counter}/{CFG['patience']}"

    print(f" {epoch:3d}   {train_loss:.4f}   {train_acc:.4f}  {val_loss:.4f}   "
          f"{val_acc:.4f}   {val_auc:.4f}   {lr:.8f}  {elapsed:6.1f}s  {status}")

    if patience_counter >= CFG['patience']:
        print(f"\nEarly stopping at epoch {epoch}")
        break

# ===================== Final Evaluation =====================
print("\n" + "="*90)
print("FINAL EVALUATION ON VALIDATION SET")
print("="*90)

# Load best model
model.load_state_dict(torch.load(CFG['save_path'], map_location=DEVICE))
val_loss, val_acc, val_auc, val_labels, val_probs = validate(
    model, val_loader, criterion, DEVICE)

val_preds = val_probs.argmax(axis=1)

print(f"\nBest Validation Accuracy: {val_acc:.4f}")
print(f"Best Validation AUC:     {val_auc:.4f}")
print(f"\nClassification Report:")
print(classification_report(val_labels, val_preds, target_names=CLASS_NAMES))

cm = confusion_matrix(val_labels, val_preds)
print("Confusion Matrix:")
print(cm)

print(f"\n* Best model saved to {CFG['save_path']}")
print(f"* Training complete!")
