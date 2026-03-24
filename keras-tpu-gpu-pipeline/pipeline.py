import os
import argparse

# The backend can be set via environment variable: 'jax', 'torch', or 'tensorflow'
backend = os.environ.get("KERAS_BACKEND", "jax")
print(f"Using Keras backend: {backend}")

import keras
import numpy as np

def build_model(model_type="classifier"):
    """Builds a Keras 3 model that is backend-agnostic."""
    if model_type == "classifier":
        model = keras.Sequential([
            keras.layers.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ])
    return model

def train_or_infer(model, mode="infer"):
    if mode == "train":
        print(f"Simulating training on {backend}...")
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        # dummy data
        x_train = np.random.random((64, 28, 28, 1)).astype("float32")
        y_train = np.random.randint(0, 10, (64,))
        model.fit(x_train, y_train, epochs=1)
        model.save("mnist_model.keras")
        print("Model saved to mnist_model.keras")
    else:
        print(f"Running inference on {backend}...")
        x_test = np.random.random((1, 28, 28, 1)).astype("float32")
        prediction = model.predict(x_test)
        print(f"Prediction shape: {prediction.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="infer", choices=["train", "infer"])
    args = parser.parse_args()
    
    model = build_model()
    train_or_infer(model, mode=args.mode)
