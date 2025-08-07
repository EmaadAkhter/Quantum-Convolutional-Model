import os
import numpy as np
from PIL import Image, ImageOps
from quantum_layers import qconv
from tensorflow import keras
from config import SAVE_PATH


def preprocess_custom(path):
    img = Image.open(path).convert('L')
    img = ImageOps.invert(img)
    bbox = img.getbbox()
    img = img.crop(bbox)
    side = max(img.size)
    img = ImageOps.pad(img, (side, side), centering=(0.5, 0.5))
    img = img.resize((28,28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)/255.0
    return arr.reshape(28,28,1)


def predict(path):
    model = keras.models.load_model(os.path.join(SAVE_PATH, "qmodel.keras"))
    img = preprocess_custom(path)
    qimg = qconv(img)
    inp = np.expand_dims(qimg, 0)
    probs = model.predict(inp)[0]
    digit = int(np.argmax(probs))

    if float(probs[digit]) < 0.6:
        return {"digit": "unknown", "confidence": float(probs[digit])}
    
    return {"digit": digit, "confidence": float(probs[digit])}
