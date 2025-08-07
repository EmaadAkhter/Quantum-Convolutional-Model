import matplotlib.pyplot as plt
from PIL import Image
import os

def visualize_prediction(fname, res):
    img = Image.open(fname).convert('L')
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {res['digit']} (Confidence: {res['confidence']:.2f})")
    plt.axis('off')
    # Save in the same directory, with _pred.png suffix
    base, _ = os.path.splitext(fname)
    out_path = f"{base}_pred.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()