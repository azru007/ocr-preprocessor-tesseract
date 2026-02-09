import cv2
import numpy as np
import sys
import os

def visualize_heatmap(heatmap_path, original_image_path, output_path):
    # 1. Load Raw Heatmap (Grayscale)
    heatmap_raw = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
    if heatmap_raw is None:
        print(f"Error: Could not load heatmap from {heatmap_path}")
        return

    # 2. Apply Colormap (Jet)
    heatmap_color = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_JET)

    # 3. Load Original Image (Optional, for overlay)
    if os.path.exists(original_image_path):
        original = cv2.imread(original_image_path)
        if original is not None:
             # Resize heatmap to match original image
            heatmap_resized = cv2.resize(heatmap_color, (original.shape[1], original.shape[0]))
            
            # Blend
            alpha = 0.5
            cv2.addWeighted(heatmap_resized, alpha, original, 1 - alpha, 0, original)
            
            # Save overlay
            cv2.imwrite(output_path, original)
            print(f"Saved overlay heatmap to: {output_path}")
            return

    # Fallback: Save just the colored heatmap
    cv2.imwrite(output_path, heatmap_color)
    print(f"Saved colored heatmap to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 visualize_heatmap.py <heatmap_raw> <original_image> <output_path>")
        sys.exit(1)

    heatmap_path = sys.argv[1]
    original_image_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "heatmap_vis.png"

    visualize_heatmap(heatmap_path, original_image_path, output_path)
