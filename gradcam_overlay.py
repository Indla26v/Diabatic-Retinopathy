import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_clincal_gradcam_overlay(original_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlays a Grad-CAM heatmap onto a retinal fundus image preserving fine clinical details.
    
    Args:
        original_img (np.ndarray): The original retinal fundus image in RGB format as a NumPy array (H, W, 3).
        heatmap (np.ndarray): The raw Grad-CAM heatmap activation array (H, W) or similar, typically with values in [0, 1].
        alpha (float): Transparency blending factor for the heatmap (default 0.4 ensures underlying vessels remain visible).
        
    Returns:
        np.ndarray: High-quality overlaid image in RGB format (H, W, 3) as a NumPy array.
    """
    # 1. Resize heatmap to match the original image dimensions exactly
    height, width = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (width, height))
    
    # 2. Normalize the heatmap to [0, 1] if not already bounded
    heatmap_normalized = heatmap_resized - np.min(heatmap_resized)
    if np.max(heatmap_normalized) != 0:
        heatmap_normalized = heatmap_normalized / np.max(heatmap_normalized)
        
    # Scale to [0, 255] and convert to uint8
    heatmap_uint8 = np.uint8(255 * heatmap_normalized)
    
    # 3. Apply Gaussian blur to the heatmap
    # This prevents the "blocky" low-resolution look from early convolutional layers
    # We dynamically calculate a kernel size based on image dimensions to ensure smooth interpolation
    kernel_size = max(11, width // 25)
    if kernel_size % 2 == 0: 
        kernel_size += 1 # Kernel size must be odd for GaussianBlur
        
    heatmap_blurred = cv2.GaussianBlur(heatmap_uint8, (kernel_size, kernel_size), 0)
    
    # 4. Apply Color Mapping
    # Using Matplotlib's 'jet' colormap to map grayscale intensities to Blue->Red gradients.
    # Alternatively, 'turbo' is also excellent for medical imaging.
    cmap = plt.get_cmap('jet')
    # Apply colormap (returns RGBA values between 0 and 1, we drop the Alpha channel)
    heatmap_colored = cmap(heatmap_blurred / 255.0)[:, :, :3]
    heatmap_colored = np.uint8(255 * heatmap_colored)
    
    # 5. Blend the images
    # We use cv2.addWeighted to merge the two arrays using the requested Alpha transparency
    original_img = np.uint8(original_img)
    overlay = cv2.addWeighted(heatmap_colored, alpha, original_img, 1 - alpha, 0)
    
    return overlay

# Example usage for testing:
if __name__ == "__main__":
    # Mocking dummy arrays
    dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
    dummy_heatmap = np.random.rand(16, 16) # low-res heatmap mock
    
    result = generate_clincal_gradcam_overlay(dummy_img, dummy_heatmap)
    print("Overlay shape:", result.shape)

