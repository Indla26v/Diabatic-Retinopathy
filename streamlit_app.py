"""
Interactive Web Interface for Diabetic Retinopathy Detection
Using CNN-Transformer Hybrid Model
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== Configuration ====================
# Must match train_improved.py architecture exactly
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

CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 
               'Severe', 'Proliferative DR']

CLASS_COLORS = {
    0: '#28CD41',  # Apple Green - No DR
    1: '#FFD60A',  # Apple Yellow - Mild
    2: '#FF9F0A',  # Apple Orange - Moderate
    3: '#FF3B30',  # Apple Red - Severe
    4: '#C1151B',  # Dark Red - Proliferative
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Model Definition ====================
class CNNTransformerDR(nn.Module):
    """CNN-Transformer Hybrid Model for DR Classification (EfficientNet-B0)
    Architecture must match train_improved.py exactly."""

    def __init__(self, cfg):
        super().__init__()
        
        # CNN encoder
        eff = tv_models.efficientnet_b0(
            weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
        self.cnn_encoder = eff.features
        
        # Project 1280 -> token_dim with BatchNorm + GELU
        self.proj = nn.Sequential(
            nn.Conv2d(1280, cfg['token_dim'], kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg['token_dim']),
            nn.GELU(),
        )
        
        # Learnable positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, 100, cfg['token_dim']) * 0.02)
        
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
        
        self.norm = nn.LayerNorm(cfg['token_dim'])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(cfg['token_dim'], 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, cfg['num_classes']),
        )
    
    def forward(self, x):
        feat = self.cnn_encoder(x)
        feat = self.proj(feat)
        
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).permute(0, 2, 1)
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
        
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        
        return self.classifier(pooled)


# ==================== Utility Functions ====================
@st.cache_resource
def load_model(model_path):
    """Load trained model from checkpoint"""
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.info("Please ensure the model checkpoint is saved at the specified path.")
        return None
    
    model = CNNTransformerDR(CFG).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def apply_clahe(img_np):
    """Apply CLAHE enhancement to image"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv2.split(img_np)
    enhanced = [clahe.apply(ch) for ch in channels]
    return cv2.merge(enhanced)


def preprocess_image(image_path):
    """
    Load and preprocess image for model inference
    
    Steps:
    1. Read image from path
    2. Convert BGR to RGB
    3. Apply CLAHE enhancement
    4. Normalize and create tensor
    
    Returns:
        image_tensor: Model-ready tensor (1, 3, 224, 224)
        original_image: Original RGB image (for display)
        clahe_image: CLAHE enhanced image (for display)
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    
    # Apply CLAHE enhancement
    clahe_image = apply_clahe(image)
    
    # Convert to PIL for transforms
    image_pil = Image.fromarray(clahe_image)
    
    # Define transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.Resize((CFG['img_size'], CFG['img_size'])),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    
    # Apply transforms
    image_tensor = transform(image_pil).unsqueeze(0)
    
    return image_tensor.to(DEVICE), original_image, clahe_image


def predict(model, image_tensor):
    """
    Make prediction using the model
    
    Args:
        model: CNN-Transformer model
        image_tensor: Preprocessed image tensor (1, 3, 224, 224)
    
    Returns:
        probabilities, predicted_class, confidence
    """
    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            output = model(image_tensor)
            probs = torch.softmax(output.float(), dim=1)
    
    probs_np = probs.cpu().numpy()[0]
    predicted_class = np.argmax(probs_np)
    confidence = probs_np[predicted_class]
    
    return probs_np, predicted_class, confidence


def create_confidence_chart(probabilities):
    """Create a bar chart of class probabilities"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [CLASS_COLORS[i] for i in range(len(CLASS_NAMES))]
    bars = ax.barh(CLASS_NAMES, probabilities, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold', color='#1D1D1F')
    ax.set_title('Model Prediction Confidence', fontsize=14, fontweight='bold', color='#1D1D1F')
    ax.set_xlim([0, 1])
    ax.tick_params(colors='#1D1D1F')
    ax.spines['bottom'].set_color('#1D1D1F')
    ax.spines['left'].set_color('#1D1D1F')

    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        ax.text(prob + 0.02, i, f'{prob:.2%}', 
                va='center', fontweight='bold', fontsize=11, color='#1D1D1F')
    
    # Make background transparent to fit the theme
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    plt.tight_layout()
    return fig


# ==================== Streamlit App ====================
def main():
    st.set_page_config(
        page_title="Diabetic Retinopathy Detector",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for Apple 'Liquid Glass' minimalist aesthetic
    st.markdown("""
    <style>
    /* Typography settings */
    html, body, [class*="css"] {
        font-family: 'SF Pro Display', 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1D1D1F;
        -webkit-font-smoothing: antialiased;
    }
    
    /* Global Background */
    .stApp {
        background-color: #FFFFFF;
    }

    /* Primary Text & Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1D1D1F !important;
        letter-spacing: -0.02em;
    }
    
    .stMarkdown p, .stText {
        color: #1D1D1F !important;
    }

    /* Target specific subtext elements to be gray */
    .st-emotion-cache-1cvhndf p, .st-emotion-cache-1629p8f h1, .st-emotion-cache-10trblm {
        color: #86868B !important;
    }
    
    h1 { font-weight: 700; font-size: 2.5rem; }
    h2 { font-weight: 600; font-size: 1.8rem; }
    h3 { font-weight: 500; font-size: 1.4rem; }

    /* Main header styling */
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
        padding-top: 1rem;
        animation: fadeIn 0.8s ease-in-out;
    }
    .main-header h1 {
        margin-bottom: 0.4rem;
    }
    .main-header p {
        font-size: 1.1rem;
        color: #86868B !important;
        font-weight: 400;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
        animation: slideUp 0.6s ease-out;
    }
    .glass-card:hover {
        transform: scale(1.01);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.08);
    }

    /* Metric/Prediction Box */
    .prediction-box {
        padding: 24px;
        border-radius: 20px;
        margin: 24px 0;
        background-color: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        text-align: center;
        transition: transform 0.3s ease-out;
        animation: scaleIn 0.5s ease-out;
    }
    .prediction-box:hover {
        transform: scale(1.02);
    }
    
    .result-success { background-color: rgba(40, 205, 65, 0.1); border: 1px solid rgba(40, 205, 65, 0.3); }
    .result-mild { background-color: rgba(255, 214, 10, 0.1); border: 1px solid rgba(255, 214, 10, 0.3); }
    .result-moderate { background-color: rgba(255, 159, 10, 0.1); border: 1px solid rgba(255, 159, 10, 0.3); }
    .result-severe { background-color: rgba(255, 59, 48, 0.1); border: 1px solid rgba(255, 59, 48, 0.3); }
    .result-proliferative { background-color: rgba(193, 21, 27, 0.1); border: 1px solid rgba(193, 21, 27, 0.3); }

    /* Buttons */
    .stButton>button {
        background-color: #28CD41 !important;
        color: #FFFFFF !important;
        border-radius: 24px !important;
        border: none !important;
        padding: 0.6rem 1.8rem !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        box-shadow: 0 4px 12px rgba(40, 205, 65, 0.2) !important;
    }
    .stButton>button:hover {
        background-color: #24b83a !important;
        transform: scale(1.03) translateY(-1px) !important;
        box-shadow: 0 8px 16px rgba(40, 205, 65, 0.3) !important;
    }

    /* Sliders and Checkboxes */
    .stCheckbox label span {
        color: #1D1D1F !important;
    }
    div[data-baseweb="checkbox"] > div {
        background-color: #28CD41 !important;
    }

    /* File Uploader border */
    .stFileUploader > div > div {
        border: 1.5px dashed rgba(0, 0, 0, 0.1) !important;
        border-radius: 16px !important;
        background-color: #FAFAFA !important;
        transition: all 0.3s ease !important;
        color: #1D1D1F !important;
    }
    .stFileUploader > div > div:hover {
        background-color: #F5F5F7 !important;
        border-color: rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #F2F2F7 !important;
    }
    [data-testid="stSidebar"] p {
        color: #86868B !important;
    }
    [data-testid="stSidebar"] strong {
        color: #1D1D1F !important;
    }
    
    /* Image hover */
    img {
        border-radius: 12px;
        transition: transform 0.3s ease-out;
    }
    img:hover {
        transform: scale(1.02);
    }

    /* Table Styling */
    table {
        border-collapse: collapse !important;
        border-radius: 16px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03) !important;
    }
    th {
        background-color: #FAFAFA !important;
        color: #86868B !important;
        font-weight: 500 !important;
        border-bottom: 1px solid rgba(0,0,0,0.05) !important;
    }
    td {
        background-color: #FFFFFF !important;
        border-bottom: 1px solid rgba(0,0,0,0.03) !important;
    }

    /* Transitions & Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.96); }
        to { opacity: 1; transform: scale(1); }
    }

    /* Thin light-gray dividers */
    hr {
        border-top: 1px solid #F2F2F7 !important;
        margin: 1.5rem 0 !important;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Diabetic Retinopathy Detection</h1>
        <p>AI-powered diagnosis using CNN-Transformer hybrid model</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        **Model**: CNN-Transformer Hybrid  
        """)
        with st.expander("View Technical Details"):
            st.markdown("""
            **Backbone**: EfficientNet-B0  
            **Classes**: 5 (No DR, Mild, Moderate, Severe, Proliferative)  
            **Input Size**: 224×224 pixels  
            **Device**: GPU if available, CPU otherwise
            """)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.header("How to Use")
        st.markdown("""
        1. **Upload** a retinal fundus image (PNG, JPG)
        2. **System** will automatically preprocess the image:
           - Apply CLAHE enhancement
           - Resize to 224×224
           - Normalize using ImageNet statistics
        3. **Model** will predict the DR severity level
        4. **View** confidence scores for each class
        """)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.header("Severity Levels")
        severity_info = {
            "No DR (0)": "No diabetic retinopathy detected",
            "Mild (1)": "Only microaneurysms present",
            "Moderate (2)": "More than just microaneurysms but <4 quadrants of hemorrhages",
            "Severe (3)": "Extensive hemorrhages in all 4 quadrants",
            "Proliferative (4)": "Neovascularization of disc or elsewhere"
        }
        for level, desc in severity_info.items():
            st.write(f"**{level}**: {desc}")
    
    # Main content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a retinal fundus image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a color fundus photograph of the eye"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Model Settings")
        model_path = st.text_input(
            "Model checkpoint path",
            value=CFG['save_path'],
            help="Path to the saved model weights"
        )
        
        use_gpu = st.checkbox(
            "Use GPU if available",
            value=True,
            help="Enable GPU acceleration"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model loading and inference
    if uploaded_file is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        # Save uploaded file temporarily
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Model Inference")
        
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown("**Original Image**")
            image = Image.open("temp_image.png")
            st.image(image, use_column_width=True)
            
        # Load model
        model = load_model(model_path)
        if model is None:
            st.stop()
        
        # Preprocess image
        with st.spinner("Processing & Enhancing image..."):
            image_tensor, original_image, clahe_image = preprocess_image("temp_image.png")
            if image_tensor is None:
                st.error("Failed to process image. Please try another image.")
                st.stop()
        
        with col_img2:
            st.markdown("**CLAHE Enhanced**")
            st.image(clahe_image, use_column_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Make prediction
        with st.spinner("Running deep learning model inference..."):
            probabilities, predicted_class, confidence = predict(model, image_tensor)
        
        # Display prediction result
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Diagnosis Results")
        
        result_class = CLASS_NAMES[predicted_class]
        result_color = CLASS_COLORS[predicted_class]
        
        # Color-coded result box
        severity_mapping = {
            0: "result-success",
            1: "result-mild",
            2: "result-moderate",
            3: "result-severe",
            4: "result-proliferative"
        }
        severity_css = severity_mapping.get(predicted_class, "result-warning")
        
        st.markdown(f"""
        <div class="prediction-box {severity_css}">
            <h2 style="color: {result_color}; margin: 0;">
                {result_class}
            </h2>
            <p style="font-size: 16px; margin: 10px 0 0 0;">
                <b>Confidence:</b> {confidence:.2%}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed probabilities and chart in columns
        col_data1, col_data2 = st.columns([1, 2], gap="large")
        
        with col_data1:
            st.markdown("**Detailed Probabilities**")
            st.markdown("""
            | Class | Probability |
            |---|---|
            """ + "\n".join([
                f"| {CLASS_NAMES[i]} | {prob:.2%} |" 
                for i, prob in enumerate(probabilities)
            ]))
        
        with col_data2:
            st.markdown("**Confidence Distribution**")
            fig = create_confidence_chart(probabilities)
            st.pyplot(fig, use_container_width=True)
        
        # Confidence interpretation
        st.markdown("<hr>", unsafe_allow_html=True)
        
        if confidence >= 0.85:
            st.success(f"High confidence prediction (≥85%)")
        elif confidence >= 0.70:
            st.info(f"Medium confidence prediction (70-85%)")
        else:
            st.warning(f"Low confidence prediction (<70%)")
        
        # Clinical note
        st.markdown("""
        <p style="font-size: 0.95rem; color: #86868B;">
        <br>
        <strong>Clinical Note</strong>: This AI model is intended to assist healthcare professionals 
        in screening for diabetic retinopathy. It should NOT be used as the sole basis 
        for clinical diagnosis. Always consult with a qualified ophthalmologist for 
        definitive diagnosis and treatment decisions.
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Cleanup
        if os.path.exists("temp_image.png"):
            os.remove("temp_image.png")
    
    else:
        st.info("Upload an image to get started!")


if __name__ == "__main__":
    main()
