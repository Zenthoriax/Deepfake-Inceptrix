from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.routes import router
import os

app = FastAPI(
    title="Deep Sentinel Fallback API",
    description="API to test fallback mechanisms for Models, Pipelines, and Infrastructure",
    version="1.0"
)

app.include_router(router)

# Mount static files folder
os.makedirs("backend-api/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="backend-api/static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("backend-api/static/index.html")
