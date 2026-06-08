import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import session, workout, voice
from core.camera_manager import CameraManager

app = FastAPI(title="Fitwise AI Coach Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session.router)
app.include_router(workout.router)
app.include_router(voice.router)

@app.on_event("startup")
async def startup_event():
    # Pre-load resources if needed
    print("Fitwise AI Coach Backend Starting...")
    camera_manager = CameraManager.get_instance()
    camera_manager.start()

@app.on_event("shutdown")
async def shutdown_event():
    print("Fitwise AI Coach Backend Shutting down...")
    camera_manager = CameraManager.get_instance()
    camera_manager.stop()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Fitwise AI Coach API is running"}

@app.get("/client")
def serve_client():
    import os
    from fastapi.responses import FileResponse
    client_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_client.html")
    if os.path.exists(client_path):
        return FileResponse(client_path)
    return {"error": "test_client.html not found"}
