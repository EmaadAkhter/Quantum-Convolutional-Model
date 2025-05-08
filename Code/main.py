from config import PREPROCESS, SAVE_PATH, EPOCHS
from data_utils import load_mnist
from quantum_layers import qconv
from model import build_model
from predict_utils import predict
import numpy as np
import matplotlib.pyplot as plt
import os

# Load data
x_train, y_train, x_test, y_test = load_mnist()

# Generate or load quantum features
if PREPROCESS:
    qx_train = np.array([qconv(img) for img in x_train], dtype=np.float32)
    qx_test  = np.array([qconv(img) for img in x_test],  dtype=np.float32)
    np.save(os.path.join(SAVE_PATH,"q_train.npy"), qx_train)
    np.save(os.path.join(SAVE_PATH,"q_test.npy"),  qx_test)
else:
    qx_train = np.load(os.path.join(SAVE_PATH,"q_train.npy"))
    qx_test  = np.load(os.path.join(SAVE_PATH,"q_test.npy"))

# Train classifier
model = build_model()
model.fit(qx_train, y_train, validation_data=(qx_test,y_test), batch_size=4, epochs=EPOCHS)
model.save(os.path.join(SAVE_PATH, "qmodel.keras"))

# Example predictions
for fname in ['test_picture/0.jpg','test_picture/1.jpg','test_picture/2.jpg','test_picture/3.jpg']:
    res = predict(fname)
    print(f"{fname}: {res}")
