import os
import uuid
import cv2
import numpy as np
import requests
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List

# [내 핵심 비즈니스 모듈 바인딩]
from personal_color_analyzer import PersonalColorAnalyzer
from makeup import apply_full_foundation_stream

router = APIRouter(prefix="/ai", tags=["Color & Makeup Processing Pipeline"])

# 최상위 기본 출력 디렉토리 정의
BASE_OUTPUT_DIR = "static/makeup_outputs"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

color_analyzer = PersonalColorAnalyzer()

# =========================================================================
# [DTO 정의] 팀원들과 합의된 100% 완전한 URL 기반 JSON 스펙
# =========================================================================
class FileItem(BaseModel):
    file_type: str = Field(..., description="파일 타입 용도 (예: skin_region, skin_mask 등)", example="skin_region")
    file_url: str = Field(..., description="해당 파일에 접근 가능한 웹 URL 주소", example="http://127.0.0.1:8000/storage/roi/xxx.png")

class ProcessRequest(BaseModel):
    user_id: int = Field(..., description="유저 고유 식별 일련번호", example=1)
    target_foundation_rgb: List[int] = Field([245, 222, 179], description="사용자가 UI에서 선택한 파운데이션 RGB 배열", example=[245, 222, 179])
    files: List[FileItem] = Field(..., description="ROIExporter가 생성한 이미지 URL 객체 리스트")


# =========================================================================
# [이미지 다운로드 유틸리티 함수]
# =========================================================================
def download_url_to_cv2(url_string: str) -> np.ndarray:
    """
    팀원의 피드백을 반영하여 requests.get과 raise_for_status()를 적용한 URL 이미지 변환 함수
    """
    try:
        response = requests.get(url_string, timeout=5)
        response.raise_for_status() 
        
        image_nparray = np.frombuffer(response.content, dtype=np.uint8)
        cv2_img = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
        return cv2_img
    except Exception as e:
        print(f" URL 다운로드 실패 ({url_string}): {e}")
        return None


# =========================================================================
# [핵심 파이프라인 엔드포인트]
# =========================================================================
@router.post("/process")
def process_ai_pipeline_from_urls(payload: ProcessRequest, request: Request):
    
    # 1. 이미지 딕셔너리 동적 조립
    regions_dictionary = {}
    
    for file_item in payload.files:
        f_type = file_item.file_type
        u_url = file_item.file_url
        
        cv2_img = download_url_to_cv2(u_url)
        if cv2_img is not None:
            regions_dictionary[f_type] = cv2_img

    # 2. 필수 가드레일 체크
    required_files = [
        "skin_region", "skin_mask", "forehead_region", 
        "left_cheek_region", "right_cheek_region", 
        "iris_region", "eyebrow_region", "lip_region"
    ]
    
    missing_files = [file for file in required_files if file not in regions_dictionary]
    if missing_files:
        raise HTTPException(
            status_code=400, 
            detail=f"분석에 필요한 필수 이미지 배달이 실패했습니다. 누락 항목: {missing_files}"
        )

    src_img = regions_dictionary["skin_region"]
    skin_mask = regions_dictionary["skin_mask"]

    # 3. Personal Color 진단 모듈 가동
    color_result = color_analyzer.diagnose_from_regions(regions_dictionary)

    # 4. 가상 메이크업 파운데이션 엔진 구동
    if len(skin_mask.shape) == 3:
        skin_mask = cv2.cvtColor(skin_mask, cv2.COLOR_BGR2GRAY)

    makeup_img = apply_full_foundation_stream(
        src_img=src_img,
        skin_mask=skin_mask,
        foundation_rgb=payload.target_foundation_rgb,
        alpha=0.22
    )

    # 5. 🌟 [디렉토리 구분 반영] 유저 ID별 전용 디렉토리 경로 빌드 및 동적 생성
    user_output_dir = os.path.join(BASE_OUTPUT_DIR, str(payload.user_id))
    os.makedirs(user_output_dir, exist_ok=True)

    # 6. 고유 파일명 생성 후 유저별 폴더에 물리적 저장
    unique_filename = f"makeup_{uuid.uuid4().hex[:8]}.jpg"
    file_save_path = os.path.join(user_output_dir, unique_filename)
    cv2.imwrite(file_save_path, makeup_img)

    # 7. 🌟 완전한 도메인 URL 생성 (경로 주소에 user_id 폴더 반영)
    base_url = str(request.base_url).rstrip("/")
    full_image_url = f"{base_url}/static/makeup_outputs/{payload.user_id}/{unique_filename}"

    # 8. 프론트 규격에 맞춰 딕셔너리 내부 값을 쪼개서 정확히 리턴
    return {
        "status": "success",
        "user_id": payload.user_id,
        "personal_color": color_result["personal_color"],
        "detected_skin_rgb": color_result["skin_rgb"],
        "makeup_image_url": full_image_url
    }