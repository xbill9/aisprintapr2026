import asyncio
import os

async def run_command(cmd: list[str], timeout: int = 60):
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return process.returncode or 0, stdout.decode().strip(), stderr.decode().strip()
    except Exception as e:
        return -1, "", str(e)

async def main():
    rc, stdout, stderr = await run_command(["gcloud", "--version"])
    print(f"RC: {rc}")
    print(f"STDOUT: {stdout}")
    print(f"STDERR: {stderr}")

if __name__ == "__main__":
    asyncio.run(main())
