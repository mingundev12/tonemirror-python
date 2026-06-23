import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import color_makeup_router

app = FastAPI(title="AIoT 딕셔너리 파라미터 기반 백엔드 서버 (Port 8000)")

OUTPUT_DIR = "static/makeup_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 분리한 라우터 모듈을 FastAPI 인스턴스에 등록합니다.
app.include_router(color_makeup_router.router)

@app.get("/")
def read_root():
    return {"message": "AIoT 백엔드 통합 제어 서버가 정상 작동 중입니다."}