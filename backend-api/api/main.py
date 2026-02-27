from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Deep Sentinel Fallback API",
    description="API to test fallback mechanisms for Models, Pipelines, and Infrastructure",
    version="1.0"
)

app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "Welcome to Deep Sentinel API. Go to /docs for Swagger UI"}
