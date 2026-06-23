import logging
import os

from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from routers.preprocess_router import router as preprocess_router
from routers.roi_router import router as roi_router

from models.saved_file_log import SavedFileLog
from models.generated_file_log import GeneratedFileLog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

load_dotenv()

app = FastAPI(
    title="Personal Color preprocessing Server",
    description="이미지 전처리 및 ROI 생성 서버",
    version="1.0.0"
)

app.mount(
    "/storage",
    StaticFiles(
        directory=os.getenv("IMAGE_PATH")
    ),
    name="storage"
)

# Router 등록
app.include_router(preprocess_router)
app.include_router(roi_router)

# Health Check API
@app.get("/")
def heath_check():
    return {
        "status": "ok",
        "message": "Preprocessing server is running"
    }