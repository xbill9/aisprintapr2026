import asyncio
import os
from main import verify_model_health

async def main():
    # Ensure environment variables are set (or they will use defaults from main.py)
    print("Running verify_model_health locally...")
    result = await verify_model_health()
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
