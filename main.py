import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import color_makeup_router

app = FastAPI(title="AIoT 딕셔너리 파라미터 기반 백엔드 서버 (Port 8000)")

# 최상위 static 폴더망이 누락되지 않도록 초기 안전 자동화 생성 공정 반영
BASE_OUTPUT_DIR = "static/makeup_outputs"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# static 내부의 유저 ID 하위 디렉토리까지 전부 웹망으로 오픈합니다.
app.mount("/static", StaticFiles(directory="static"), name="static")

# 분리한 라우터 모듈을 FastAPI 인스턴스에 등록합니다.
app.include_router(color_makeup_router.router)

@app.get("/")
def read_root():
    return {"message": "AIoT 백엔드 통합 제어 서버가 정상 작동 중입니다."}