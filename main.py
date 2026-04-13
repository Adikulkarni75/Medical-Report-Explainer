from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.routes import router

app = FastAPI(
    title="Medical Report Explainer",
    description="Upload a medical PDF and get plain-language explanations",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(router, prefix="/api")

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")