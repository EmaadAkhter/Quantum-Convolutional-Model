import matplotlib.pyplot as plt
from PIL import Image

def visualize_prediction(fname, res):
    img = Image.open(fname).convert('L')
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {res['digit']} (Confidence: {res['confidence']:.2f})")
    plt.axis('off')
    plt.savefig(fname.replace('.jpg', f'{fname}_pred.png'), bbox_inches='tight')