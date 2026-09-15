# main.py
"""App entrypoint. Run with: uvicorn app.main:app --host 0.0.0.0 --port 8010"""
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .route import router as catalog_router
from .auth import router as auth_router
from .database import init_db

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Virtual Curtain Try-On API", version="2.0.0")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_dir = os.path.join(BASE_DIR, "frontend")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # tighten in production
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Fabric/Room catalog CRUD (S3 + MySQL) and JWT auth.
app.include_router(catalog_router)
app.include_router(auth_router)

# Frontend static files only -- fabric/room/curtain images now live in S3,
# so those local static mounts are gone.
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/static/api.js")
def serve_api_js():
    return FileResponse(
        os.path.join(frontend_dir, "api.js"),
        media_type="application/javascript",
    )


@app.get("/")
def home():
    return FileResponse(os.path.join(frontend_dir, "index.html"))


@app.on_event("startup")
async def startup():
    # Initialize MySQL (creates users/fabrics/rooms tables if needed)
    await init_db()


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
