import os
import uuid
import cv2
import numpy as np
import requests 
from fastapi import APIRouter, HTTPException, Request
from typing import List

from services.color_service import PersonalColorAnalyzer
from services.makeup_service import apply_full_foundation_stream
# 🛠️ PersonalColorRequest Import 제거 완료
from schemas.color_makeup_schema import VirtualMakeupRequest, FileItem

router = APIRouter(prefix="/ai", tags=["Color & Makeup Processing Pipeline"])

BASE_OUTPUT_DIR = "static/makeup_outputs"
color_analyzer = PersonalColorAnalyzer()


def download_url_to_cv2(url_string: str) -> np.ndarray:
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
# [1️⃣ 통합 검증 가드레일 - 8개 영역 전체 필수 및 최적화]
# =========================================================================
def fetch_and_validate_personal_color_regions(files: List[FileItem]) -> dict:
    """
    [존재 이유] 중복 다운로드로 인한 network 자원 낭비를 막고, 정확한 피부 톤 분석을 사전에 보장하여 메이크업 연산의 연쇄 에러를 방지하는 무결성 가드레일입니다.
    """
    regions_dictionary = {}
    for file_item in files:
        img = download_url_to_cv2(file_item.file_url)
        if img is not None:
            regions_dictionary[file_item.file_type] = img

    required_types = [
        "skin_region", "skin_mask", "forehead_region", 
        "left_cheek_region", "right_cheek_region", 
        "iris_region", "eyebrow_region", "lip_region"
    ]
    
    missing_elements = [r_type for r_type in required_types if r_type not in regions_dictionary]
    if missing_elements:
        raise HTTPException(
            status_code=422,
            detail=f"퍼스널 컬러 진단 및 메이크업에 필요한 파트가 누락되었습니다: {missing_elements}"
        )
        
    return regions_dictionary


# =========================================================================
# [2️⃣ 가상 메이크업 합성 엔드포인트 (내부 파라미터로 퍼컬 진단 통합)]
# =========================================================================
@router.post("/virtual-makeup")
def process_virtual_makeup_only(payload: VirtualMakeupRequest, request: Request):
    # 8개 영역 전체 검증 함수 호출하여 이미지를 단 한 번만 다운로드 및 검증 (연산 중복 제거)
    regions_dictionary = fetch_and_validate_personal_color_regions(payload.files)
    
    # 1. 내부 파라미터 연산: 다운로드된 데이터를 기반으로 퍼스널 컬러 진단 수행
    color_result = color_analyzer.diagnose_from_regions(regions_dictionary)
    skin_hex = f"#{color_result['skin_rgb'][0]:02X}{color_result['skin_rgb'][1]:02X}{color_result['skin_rgb'][2]:02X}"
    
    # 2. 메이크업 합성 연산 수행
    src_img = regions_dictionary["skin_region"]
    skin_mask = regions_dictionary["skin_mask"]

    # 헥스코드 변환 로직 (# 제거 후 RGB 파싱)
    hex_str = payload.target_foundation_hex.lstrip('#')
    target_rgb = [int(hex_str[i:i+2], 16) for i in (0, 2, 4)]

    if len(skin_mask.shape) == 3:
        skin_mask = cv2.cvtColor(skin_mask, cv2.COLOR_BGR2GRAY)

    makeup_img = apply_full_foundation_stream(
        src_img=src_img,
        skin_mask=skin_mask,
        foundation_rgb=target_rgb,
        alpha=0.22
    )

    # 원본 이미지 ID 단위로 임시 출력 디렉토리 분리 (휘발성 세션 데이터 꼬임 방지)
    user_output_dir = os.path.join(BASE_OUTPUT_DIR, str(payload.original_image_id))
    os.makedirs(user_output_dir, exist_ok=True)

    unique_filename = f"makeup_{uuid.uuid4().hex[:8]}.jpg"
    file_save_path = os.path.join(user_output_dir, unique_filename)
    cv2.imwrite(file_save_path, makeup_img)

    base_url = str(request.base_url).rstrip("/")
    full_image_url = f"{base_url}/static/makeup_outputs/{payload.original_image_id}/{unique_filename}"

    original_image_url = next(
        (file.file_url for file in payload.files if file.file_type == "skin_region"), 
        None
    )

    # 최종 결과 반환: 진단된 퍼스널 컬러와 메이크업 결과를 한 번에 묶어서 프론트에 리턴
    return {
        "status": "success",
        "original_image_id": payload.original_image_id,
        "personal_color": color_result["personal_color"],
        "detected_skin_hex": skin_hex,
        "selected_foundation_hex": payload.target_foundation_hex,
        "original_image_url": original_image_url,
        "makeup_image_url": full_image_url
    }