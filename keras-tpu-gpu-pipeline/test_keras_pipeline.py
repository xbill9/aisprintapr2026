# Test Client for Keras TPU-GPU Pipeline

import asyncio
from server import check_backend_compatibility, generate_training_script

async def test_keras_pipeline():
    print("--- Testing Backend Compatibility Check ---")
    # Simulate a standard CNN model config
    compat_result = await check_backend_compatibility(model_configuration="StandardCNN")
    print(f"Compatibility: {compat_result['compatibility']}")
    print(f"Supported Backends: {compat_result['backends']}")
    
    print("\n--- Testing Training Script Generation (JAX) ---")
    jax_script = await generate_training_script(model_name="MNIST_Classifier", backend="jax")
    print(f"Generated: {jax_script['script_name']}")
    print(f"Instructions: {jax_script['instructions']}")

    print("\n--- Testing Training Script Generation (Torch) ---")
    torch_script = await generate_training_script(model_name="MNIST_Classifier", backend="torch")
    print(f"Generated: {torch_script['script_name']}")
    print(f"Instructions: {torch_script['instructions']}")

if __name__ == "__main__":
    asyncio.run(test_keras_pipeline())
