from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
import asyncio

app = FastAPI()

# Serve the HTML file
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/stream")
async def stream():
    async def event_generator():
        for i in range(1, 11):
            yield {
                "event": "message",
                "data": f"Streaming chunk {i}"
            }
            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())
