from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, Response
import uvicorn
import asyncio

app = FastAPI()
latest_detection = {"violations": 0}
latest_frame = None  # Global variable to store the latest frame bytes

@app.get("/")
def index():
    return HTMLResponse("""
    <html>
    <body>
    <h1>Pizza Violation Detection Streaming Service</h1>
    <p>Open a WebSocket connection to /ws for real-time updates.</p>
    </body>
    </html>
    """)

@app.get("/metadata")
def get_metadata():
    return latest_detection

@app.post("/metadata")
async def update_metadata(request: Request):
    global latest_detection
    data = await request.json()
    latest_detection.update(data)
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        await asyncio.sleep(1)
        await websocket.send_json(latest_detection)

@app.post("/frame")
async def update_frame(request: Request):
    global latest_frame
    latest_frame = await request.body()
    return {"status": "ok"}

@app.get("/frame")
def get_frame():
    if latest_frame is not None:
        return Response(content=latest_frame, media_type="image/jpeg")
    return Response(status_code=404)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 