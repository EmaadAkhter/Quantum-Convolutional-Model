from tensorflow import keras


def build_model():
    """Simple classifier on quantum features."""
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(14, 14, 4)),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model