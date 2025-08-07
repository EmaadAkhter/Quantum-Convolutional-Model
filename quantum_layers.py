import numpy as np
import pennylane as qml
from pennylane.templates import RandomLayers
from config import LAYERS

# Set device
dev = qml.device("default.qubit", wires=4)
params = np.random.uniform(0, 2*np.pi, size=(LAYERS, 6))

@qml.qnode(dev)
def qcircuit(inputs):
    for wire in range(4):
        qml.RY(np.pi * inputs[wire], wires=wire)
    RandomLayers(params, wires=range(4))
    return [qml.expval(qml.PauliZ(wire)) for wire in range(4)]


def qconv(image):
    out = np.zeros((14, 14, 4), dtype=np.float32)
    for i in range(0, 28, 2):
        for j in range(0, 28, 2):
            patch = [
                image[i, j, 0], image[i, j+1, 0],
                image[i+1, j, 0], image[i+1, j+1, 0]
            ]
            out[i//2, j//2, :] = qcircuit(patch)
    return out