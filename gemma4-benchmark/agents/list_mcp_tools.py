
import asyncio
import os
from mcp import ClientSession
from mcp.client.http import http_client

async def main():
    url = os.environ.get("MCP_SERVER_URL")
    async with http_client(url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")

if __name__ == "__main__":
    asyncio.run(main())
