
# Quantum Convolutional Model

This project integrates quantum computing with classical machine learning to build a hybrid Quantum Convolutional Neural Network (QCNN) for digit classification using the MNIST dataset. It leverages PennyLane for quantum circuit simulation and TensorFlow/Keras for training a classical neural network on quantum-processed features.




## Features:

- Quantum Feature Extraction: A 2×2 sliding window is used to feed image patches into a 4-qubit quantum circuit, extracting 4 quantum features per patch (i.e., quantum convolution).

- Preprocessing Option: Quantum-transformed image data is optionally precomputed and stored to reduce training time on subsequent runs.

- Model Architecture: A simple feedforward neural network is trained on the 14×14×4 quantum feature maps.

- Prediction on Custom Images: Custom handwritten digits (in image format) can be passed through the same quantum pipeline and classified by the trained model.

- Visualization: Quantum channel outputs (Q0–Q3) are visualized alongside original input images.


## Tech Stack:

- Quantum Layer: PennyLane, NumPy

- Classical Layer: TensorFlow/Keras

- Image Processing: PIL, Matplotlib

- Dataset: MNIST

## Graphs

### Quantum-processed output

  
  ![Image](https://github.com/EmaadAkhter/Quantum-Convolutional-Model/blob/main/Code/assets/viz.png)

### Training Performance

![Image](https://github.com/EmaadAkhter/Quantum-Convolutional-Model/blob/main/Code/assets/training_plot.png)

### Prediction visualisztion
<img src="https://github.com/EmaadAkhter/Quantum-Convolutional-Model/blob/main/Code/assets/0_pred.png" width="400" /><img src="https://github.com/EmaadAkhter/Quantum-Convolutional-Model/blob/main/Code/assets/1_pred.png" width="400" height="341.1" />
<img src="https://github.com/EmaadAkhter/Quantum-Convolutional-Model/blob/main/Code/assets/2_pred.png" width="400" /><img src="https://github.com/EmaadAkhter/Quantum-Convolutional-Model/blob/main/Code/assets/3_pred.png" width="400" />




## Deployment

To deploy this project run

```bash
   git clone https://github.com/EmaadAkhter/Quantum-Convolutional-Model.git  
```
Go to the project directory
```bash
  cd Quantum-Convolutional-Model
  cd code
```
Run the setup file
```bash
  sh setup.sh
```
Run the python script
```bash
  python main.py
```

## Acknowledgements

 - [Quanvolutional Neural Networks](https://pennylane.ai/qml/demos/tutorial_quanvolution)
 
 - [pennylane](https://pennylane.ai/)
 
