# Grand Demo: Keras TPU-GPU Pipeline

import asyncio
from server import init_training_job, convert_weights, check_backend_compatibility, get_docker_template

async def keras_demo():
    print("🚀 TPU Sprint Demo: Keras TPU-GPU Pipeline")
    print("=" * 60)
    
    # Step 1: Initialize training job on TPU
    print("\n[Step 1] Initiating Training Job (JAX on TPU v5e-4)...")
    job = await init_training_job(model_name="ResNet50_Classifier", tpu_version="v5e-4")
    print(f"  JOB_ID: {job['job_id']} | BACKEND: {job['backend']}")
    print(f"  SUBMISSION_COMMAND:\n  {job['submission_command']}")
    
    # Step 2: Compatibility check
    print("\n[Step 2] Checking Cross-Backend Compatibility...")
    compat = await check_backend_compatibility(model_configuration="ResNet50_Classifier")
    print(f"  COMPATIBILITY: {compat['compatibility']} | BACKENDS: {compat['backends']}")
    
    # Step 3: Automatic weight conversion
    print("\n[Step 3] Generating TPU-to-GPU Weight Conversion Script...")
    conv = await convert_weights(weights_path="gs://tpu-sprint-demo/resnet50.keras", source_backend="jax", target_backend="torch")
    print(f"  TARGET BACKEND: {conv['target_backend']}")
    print(f"  SNIPPET:\n{conv['snippet']}")
    
    # Step 4: Dockerize for serving
    print("\n[Step 4] Retrieving Multi-Backend Serving Template...")
    docker = get_docker_template()
    print(f"  BASE IMAGE: {docker.split('FROM')[-1].split('RUN')[0].strip()}")
    print(f"  DEFAULT BACKEND: {docker.split('ENV KERAS_BACKEND=')[-1].split('WORKDIR')[0].strip()}")
    
    print("\n" + "=" * 60)
    print("✅ Keras 3 Pipeline Ready: Trained on TPU, served on Serverless GPU!")

if __name__ == "__main__":
    asyncio.run(keras_demo())
