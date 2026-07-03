import logging
import os

from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from routers.preprocess_router import router as preprocess_router
from routers.roi_router import router as roi_router
from routers.color_makeup_router import router as color_makeup_router
from routers.diagnose_router import router as diagnose_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

load_dotenv()

# FastAPI App
app = FastAPI(
    title="Personal Color preprocessing Server",
    description="전처리, 분석, 메이크업 AI 통합 서버",
    version="1.0.0"
)

# Preprocess 파일 경로
app.mount(
    "/storage",
    StaticFiles(
        directory=os.getenv("IMAGE_PATH")
    ),
    name="storage"
)

# Makeup 결과 저장 경로
BASE_OUTPUT_DIR = "static/makeup_outputs"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

# Router 등록
app.include_router(preprocess_router)
app.include_router(roi_router)
app.include_router(color_makeup_router)
app.include_router(diagnose_router)

# Health Check API
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Preprocessing server is running"
    }