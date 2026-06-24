import os
import uuid
import cv2
import numpy as np
import requests 
from fastapi import APIRouter, HTTPException, Request
from typing import List

# [내 핵심 비즈니스 모듈 바인딩] 
from services.color_service import PersonalColorAnalyzer
from services.makeup_service import apply_full_foundation_stream

# [DTO 모듈 바인딩]
from schemas.color_makeup_schema import PersonalColorRequest, VirtualMakeupRequest, FileItem

router = APIRouter(prefix="/ai", tags=["Color & Makeup Processing Pipeline"])

# 중복 os.makedirs 제거하고 경로 상수만 통일성 있게 관리
BASE_OUTPUT_DIR = "static/makeup_outputs"

color_analyzer = PersonalColorAnalyzer()


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

# 이미지 다운로드 및 필수 가드레일 공통 검증 유틸
def fetch_and_validate_regions(files: List[FileItem]) -> dict:
    regions_dictionary = {}
    for file_item in files:
        cv2_img = download_url_to_cv2(file_item.file_url)
        if cv2_img is not None:
            regions_dictionary[file_item.file_type] = cv2_img

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
    return regions_dictionary


# =========================================================================
# [1️⃣ 퍼스널 컬러 진단 엔드포인트]
# =========================================================================
@router.post("/personal-color")
def process_personal_color_only(payload: PersonalColorRequest):
    """
    [기획서 명세 1] 컬러 진단명과 대표 피부색 RGB 전송
    """
    # 이미지 다운로드 및 가드레일 체크
    regions_dictionary = fetch_and_validate_regions(payload.files)

    # Personal Color 진단 모듈 가동
    color_result = color_analyzer.diagnose_from_regions(regions_dictionary)

    # 응답 규격 리턴
    return {
        "status": "success",
        "user_id": payload.user_id,
        "personal_color": color_result["personal_color"],
        "detected_skin_rgb": color_result["skin_rgb"]
    }


# =========================================================================
# [2️⃣ 가상 메이크업 합성 엔드포인트]
# =========================================================================
@router.post("/virtual-makeup")
def process_virtual_makeup_only(payload: VirtualMakeupRequest, request: Request):
    """
    [기획서 명세 2] 선택 파운데이션 기반 메이크업 합성 이미지 URL 생성
    """
    # 이미지 다운로드 및 가드레일 체크
    regions_dictionary = fetch_and_validate_regions(payload.files)
    src_img = regions_dictionary["skin_region"]
    skin_mask = regions_dictionary["skin_mask"]

    # 가상 메이크업 파운데이션 엔진 구동
    if len(skin_mask.shape) == 3:
        skin_mask = cv2.cvtColor(skin_mask, cv2.COLOR_BGR2GRAY)

    makeup_img = apply_full_foundation_stream(
        src_img=src_img,
        skin_mask=skin_mask,
        foundation_rgb=payload.target_foundation_rgb,
        alpha=0.22
    )

    # [디렉토리 구분 기능] 유저 ID별로 전용 폴더 동적 생성
    user_output_dir = os.path.join(BASE_OUTPUT_DIR, str(payload.user_id))
    os.makedirs(user_output_dir, exist_ok=True)

    # 고유한 파일명 매핑 및 유저 전용 디렉토리에 저장
    unique_filename = f"makeup_{uuid.uuid4().hex[:8]}.jpg"
    file_save_path = os.path.join(user_output_dir, unique_filename)
    cv2.imwrite(file_save_path, makeup_img)

    # 완전한 도메인 URL 생성 (경로에 user_id 반영)
    base_url = str(request.base_url).rstrip("/")
    full_image_url = f"{base_url}/static/makeup_outputs/{payload.user_id}/{unique_filename}"

    # 응답 규격 리턴
    return {
        "status": "success",
        "user_id": payload.user_id,
        "makeup_image_url": full_image_url
    }