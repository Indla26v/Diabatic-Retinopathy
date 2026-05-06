"""
Standalone Prediction Script
For batch predictions or programmatic usage
Can be imported as a module or run directly
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import json

# ==================== Configuration ====================
CFG = {
    'img_size'    : 224,
    'num_classes' : 5,
    'token_dim'   : 256,
    'num_heads'   : 8,
    'num_layers'  : 2,
    'ff_dim'      : 512,
    'tf_dropout'  : 0.15,
    'save_path'   : 'best_hybrid_model (1).pth',
}

CLASS_NAMES = [
    'No DR', 
    'Mild', 
    'Moderate', 
    'Severe', 
    'Proliferative DR'
]

CLASS_DESCRIPTIONS = {
    0: "No diabetic retinopathy",
    1: "Mild - Only microaneurysms present",
    2: "Moderate - More than just microaneurysms",
    3: "Severe - Extensive hemorrhages",
    4: "Proliferative - Neovascularization present"
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== Model Definition ====================

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


# ==================== Utility Functions ====================
def apply_clahe(img_np):
    """Apply CLAHE enhancement to image"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv2.split(img_np)
    enhanced = [clahe.apply(ch) for ch in channels]
    return cv2.merge(enhanced)


def preprocess_image(image_path):
    """
    Preprocess image for model inference
    
    Args:
        image_path: Path to retinal fundus image
    
    Returns:
        image_tensor: Preprocessed tensor (1, 3, 224, 224)
        original_image: Original loaded image (for visualization)
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    
    # Apply CLAHE enhancement
    image = apply_clahe(image)
    
    # Apply transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.Resize((CFG['img_size'], CFG['img_size'])),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    
    image_pil = Image.fromarray(image)
    image_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
    
    return image_tensor, original_image


# ==================== ModelPredictor Class ====================
class DiabRetinoPathyPredictor:
    """
    Convenient wrapper for single or batch predictions
    
    Usage:
        predictor = DiabRetinoPathyPredictor('best_hybrid_model (1).pth')
        results = predictor.predict_image('retinal_image.png')
        results_batch = predictor.predict_batch(['img1.png', 'img2.png'])
    """
    
    def __init__(self, model_path=CFG['save_path']):
        """
        Initialize predictor with model checkpoint
        
        Args:
            model_path: Path to saved model weights
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load model from checkpoint"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = CNNTransformerDR(CFG).to(DEVICE)
        checkpoint = torch.load(self.model_path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print(f"✓ Model loaded from {self.model_path}")
    
    def predict_image(self, image_path, return_probs=False):
        """
        Predict DR class for single image
        
        Args:
            image_path: Path to retinal image
            return_probs: If True, return all class probabilities
        
        Returns:
            dict: Prediction results
                {
                    'image_path': str,
                    'predicted_class': int (0-4),
                    'predicted_label': str,
                    'confidence': float,
                    'description': str,
                    'all_probabilities': dict (optional)
                }
        """
        try:
            # Preprocess
            image_tensor, _ = preprocess_image(image_path)
            
            # Predict
            with torch.no_grad():
                output = self.model(image_tensor)
                probs = torch.softmax(output.float(), dim=1)
            
            probs_np = probs.cpu().numpy()[0]
            pred_class = np.argmax(probs_np)
            confidence = probs_np[pred_class]
            
            result = {
                'image_path': str(image_path),
                'predicted_class': int(pred_class),
                'predicted_label': CLASS_NAMES[pred_class],
                'confidence': float(confidence),
                'description': CLASS_DESCRIPTIONS[pred_class],
            }
            
            if return_probs:
                result['all_probabilities'] = {
                    CLASS_NAMES[i]: float(prob) 
                    for i, prob in enumerate(probs_np)
                }
            
            return result
        
        except Exception as e:
            return {
                'image_path': str(image_path),
                'error': str(e),
                'success': False
            }
    
    def predict_batch(self, image_paths, return_probs=False, verbose=True):
        """
        Predict DR class for multiple images
        
        Args:
            image_paths: List of paths to retinal images
            return_probs: If True, return all class probabilities
            verbose: If True, print progress
        
        Returns:
            list: List of prediction results for each image
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            if verbose:
                print(f"Processing {i+1}/{len(image_paths)}: {image_path}")
            
            result = self.predict_image(image_path, return_probs)
            results.append(result)
        
        return results
    
    def predict_directory(self, directory, file_extension='.png', 
                         return_probs=False, verbose=True):
        """
        Predict DR class for all images in a directory
        
        Args:
            directory: Path to directory containing images
            file_extension: File extension to search for (.png, .jpg, etc.)
            return_probs: If True, return all class probabilities
            verbose: If True, print progress
        
        Returns:
            list: List of prediction results
        """
        image_dir = Path(directory)
        image_paths = list(image_dir.glob(f'*{file_extension}'))
        
        if not image_paths:
            print(f"No images found with extension {file_extension}")
            return []
        
        print(f"Found {len(image_paths)} images")
        return self.predict_batch(image_paths, return_probs, verbose)


# ==================== Standalone Functions ====================
def predict_single_image(image_path, model_path=CFG['save_path']):
    """
    Quick prediction for single image
    
    Args:
        image_path: Path to retinal image
        model_path: Path to model checkpoint
    
    Returns:
        dict: Prediction results
    """
    predictor = DiabRetinoPathyPredictor(model_path)
    return predictor.predict_image(image_path, return_probs=True)


def predict_multiple_images(image_paths, model_path=CFG['save_path']):
    """
    Quick prediction for multiple images
    
    Args:
        image_paths: List of paths to retinal images
        model_path: Path to model checkpoint
    
    Returns:
        list: Prediction results for each image
    """
    predictor = DiabRetinoPathyPredictor(model_path)
    return predictor.predict_batch(image_paths, return_probs=True)


# ==================== JSON Export ====================
def save_results_as_json(results, output_path='predictions.json'):
    """
    Save prediction results to JSON file
    
    Args:
        results: Single result dict or list of results
        output_path: Path to save JSON file
    """
    # Convert to list if single result
    if isinstance(results, dict):
        results = [results]
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {output_path}")


def print_result(result):
    """Pretty print prediction result"""
    if 'error' in result:
        print(f"\n❌ Error processing {result['image_path']}")
        print(f"   {result['error']}")
        return
    
    print(f"\n{'='*60}")
    print(f"Image: {result['image_path']}")
    print(f"{'='*60}")
    print(f"Predicted Class: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Description: {result['description']}")
    
    if 'all_probabilities' in result:
        print(f"\nDetailed Probabilities:")
        for label, prob in result['all_probabilities'].items():
            bar = "█" * int(prob * 40)
            print(f"  {label:20s}: {bar} {prob:.4f}")
    print(f"{'='*60}")


# ==================== Main / Example Usage ====================
if __name__ == "__main__":
    """
    Example usage - uncomment to test
    """
    
    print("Diabetic Retinopathy Prediction Script")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Model size: {CFG['img_size']}x{CFG['img_size']}")
    print(f"Classes: {CFG['num_classes']}")
    print("="*60 + "\n")
    
    # Example 1: Single image prediction
    print("Example 1: Single Image Prediction")
    print("-"*60)
    try:
        result = predict_single_image('retinal_image.png')
        print_result(result)
    except Exception as e:
        print(f"Note: Please provide a valid image path. Error: {e}\n")
    
    # Example 2: Batch prediction
    print("\n\nExample 2: Batch Prediction")
    print("-"*60)
    print("Usage: results = predict_multiple_images(['img1.png', 'img2.png'])")
    print("Then: for r in results: print_result(r)")
    
    # Example 3: Using predictor class
    print("\n\nExample 3: Using DiabRetinoPathyPredictor Class")
    print("-"*60)
    print("""
    from dr_predictor import DiabRetinoPathyPredictor
    
    # Initialize
predictor = DiabRetinoPathyPredictor('best_hybrid_model (1).pth')
    
    # Single image
    result = predictor.predict_image('retinal_image.png', return_probs=True)
    
    # Batch
    results = predictor.predict_batch(['img1.png', 'img2.png'])
    
    # Directory
    results = predictor.predict_directory('path/to/images/')
    
    # Save results
    from dr_predictor import save_results_as_json
    save_results_as_json(results, 'predictions.json')
    """)
    
    print("\n✓ Script ready for use!")
