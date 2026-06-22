import os
import sys
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Request  # 👈 Request 추가
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List

# =========================================================================
# [MySQL 데이터베이스 연동 패키지]
# =========================================================================
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

# [내 핵심 비즈니스 모듈 바인딩]
from personal_color_analyzer import PersonalColorAnalyzer
from makeup import apply_full_foundation_stream

app = FastAPI(title="AIoT MySQL 기반 통합 백엔드 서버 (Port 8000)")

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# [MySQL 커넥션 설정] 팀의 스키마 환경에 맞게 계정/비밀번호/디비명 수정 필요
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
DATABASE_URL = "mysql+pymysql://root:1234@localhost:3306/aiot_db"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """ 요청마다 DB 세션을 열고 처리가 끝나면 자동으로 닫아주는 의존성 주입 함수 """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# [로컬 임시 저장 및 정적 서빙 설정]
# 프론트엔드가 결과물 이미지 URL로 직접 접근할 수 있도록 서빙 레이어를 올립니다.
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
OUTPUT_DIR = "static/makeup_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 내 AI 분석 엔진 전역 로드
color_analyzer = PersonalColorAnalyzer()


# =========================================================================
# [DTO 정의] 프론트엔드가 8000번 포트로 요청할 때 던져줄 데이터 규격
# =========================================================================
class ProcessRequest(BaseModel):
    user_id: int = Field(..., description="MySQL에서 다른 사람과 구분하여 데이터를 추출하기 위한 고유 고유 식별 번호")
    target_foundation_rgb: List[int] = Field([245, 222, 179], description="사용자가 UI에서 픽한 파운데이션 컬러")


def convert_mysql_blob_to_cv2(blob_data, is_gray=False):
    """ MySQL LONGBLOB 타입의 바이너리 데이터를 OpenCV 행렬(Numpy Array)로 역직렬화 """
    if not blob_data:
        return None
    nparr = np.frombuffer(blob_data, np.uint8)
    if is_gray:
        return cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def get_valid_mean_rgb(img_bgr):
    """ 마스킹 처리되어 잘려나간 ROI 이미지에서 검은 배경(0,0,0)을 제외한 유효 영역의 평균 RGB 계산 """
    if img_bgr is None: 
        return [128, 128, 128]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    idx = np.where(gray > 0)
    if len(idx[0]) == 0: 
        return [128, 128, 128]
    return [
        int(np.mean(img_bgr[:, :, 2][idx])), # R 채널 평균
        int(np.mean(img_bgr[:, :, 1][idx])), # G 채널 평균
        int(np.mean(img_bgr[:, :, 0][idx]))  # B 채널 평균
    ]


# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# [핵심 라우터] DB 연산부터 프론트엔드 반환까지 이어지는 데이터 파이프라인
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
@app.post("/ai/process")
def process_ai_pipeline_from_mysql(
    payload: ProcessRequest, 
    request: Request,               # 👈 프론트 도메인을 동적으로 가져오기 위해 주입
    db: Session = Depends(get_db)
):
    """
    [전체 데이터 흐름 확인]
    프론트엔드 요청 (user_id 수신) ──► MySQL 조회 (WHERE 절 분기) ──► 내 소스코드 연산 (personal/makeup) 
    ──► static 폴더 파일 저장 ──► 프론트엔드로 최종 결과 패킷 전송 (8000번 포트 응답)
    """
    try:
        # [Step 1] MySQL 테이블에서 넘겨받은 고유 user_id에 매칭되는 행만 정확하게 타겟팅해서 가져옵니다.
        query = text("""
            SELECT skin_region, skin_mask, forehead_region, left_cheek_region, 
                   right_cheek_region, iris_region, eyebrow_region, lip_region 
            FROM face_roi_table 
            WHERE user_id = :user_id
        """)
        
        result = db.execute(query, {"user_id": payload.user_id}).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="MySQL 데이터베이스에 해당 user_id의 얼굴 분석 파일이 존재하지 않습니다.")

        # [Step 2] DB 튜플 데이터들을 내 코드들이 다룰 수 있도록 OpenCV 행렬 객체로 원상복구합니다.
        src_img = convert_mysql_blob_to_cv2(result['skin_region'])
        skin_mask = convert_mysql_blob_to_cv2(result['skin_mask'], is_gray=True)
        
        forehead_img = convert_mysql_blob_to_cv2(result['forehead_region'])
        left_cheek_img = convert_mysql_blob_to_cv2(result['left_cheek_region'])
        right_cheek_img = convert_mysql_blob_to_cv2(result['right_cheek_region'])
        iris_img = convert_mysql_blob_to_cv2(result['iris_region'])
        eyebrow_img = convert_mysql_blob_to_cv2(result['eyebrow_region'])
        lip_img = convert_mysql_blob_to_cv2(result['lip_region'])

        if src_img is None or skin_mask is None:
            raise HTTPException(status_code=500, detail="MySQL 바이너리 데이터를 이미지 객체로 변환하는 과정에서 유실이 발생했습니다.")

        # [Step 3] 내 Personal Color 분석 알고리즘 가동
        left_cheek_rgb = get_valid_mean_rgb(left_cheek_img)
        right_cheek_rgb = get_valid_mean_rgb(right_cheek_img)
        cheek_rgb = [int((left_cheek_rgb[i] + right_cheek_rgb[i]) / 2) for i in range(3)]
        
        color_result = color_analyzer.diagnose(
            cheek_rgb=cheek_rgb,
            forehead_rgb=get_valid_mean_rgb(forehead_img),
            eye_rgb=get_valid_mean_rgb(iris_img),
            eyebrow_rgb=get_valid_mean_rgb(eyebrow_img),
            lip_rgb=get_valid_mean_rgb(lip_img)
        )

        # [Step 4] 내 가상 메이크업 렌더링 알고리즘 가동
        makeup_img = apply_full_foundation_stream(
            src_img=src_img,
            skin_mask=skin_mask,
            foundation_rgb=payload.target_foundation_rgb,
            alpha=0.22
        )

        # [Step 5] 동시성 충돌 방지를 위해 UUID를 활용한 고유 파일명 부여 후 물리적 저장 (.jpg)
        unique_filename = f"makeup_{uuid.uuid4().hex[:8]}.jpg"
        file_save_path = os.path.join(OUTPUT_DIR, unique_filename)
        cv2.imwrite(file_save_path, makeup_img)

        # 🌟 [Step 6] 현재 호출된 서버의 도메인(IP 및 포트)을 포함한 완전한 URL 생성
        # 예시: http://localhost:8000/static/makeup_outputs/makeup_a1b2c3d4.jpg
        base_url = str(request.base_url).rstrip("/")
        full_image_url = f"{base_url}/static/makeup_outputs/{unique_filename}"

        # 프론트엔드가 받아서 렌더링할 최종 JSON 결과물을 웹 응답으로 내보냅니다.
        return {
            "status": "success",
            "user_id": payload.user_id,
            "personal_color": color_result["personal_color"],
            "detected_skin_rgb": color_result["skin_rgb"],
            "makeup_image_url": full_image_url  # 👈 상대 경로 대신 완성형 URL 반환
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MySQL 통합 제어 파이프라인 연동 실패: {str(e)}")