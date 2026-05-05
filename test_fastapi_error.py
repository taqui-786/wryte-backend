import asyncio
from httpx import AsyncClient
from app.app import app
import uvicorn
from threading import Thread
import time

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="debug")

t = Thread(target=run_server, daemon=True)
t.start()
time.sleep(2)

async def test():
    async with AsyncClient(base_url="http://127.0.0.1:8001") as client:
        response = await client.get("/get-state/786")
        print("STATUS:", response.status_code)
        print("HEADERS:", response.headers)
        print("TEXT:", response.text)

asyncio.run(test())
