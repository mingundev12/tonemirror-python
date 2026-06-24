import os
import uuid
import cv2
import numpy as np
import requests 
from fastapi import APIRouter, HTTPException, Request
from typing import List

from services.color_service import PersonalColorAnalyzer
from services.makeup_service import apply_full_foundation_stream
from schemas.color_makeup_schema import PersonalColorRequest, VirtualMakeupRequest, FileItem

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
# [1️⃣ 퍼스널 컬러용 검증 가드레일 - 8개 영역 전체 필수]
# =========================================================================
def fetch_and_validate_personal_color_regions(files: List[FileItem]) -> dict:
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
            detail=f"퍼스널 컬러 분석에 필요한 필수 이미지 누락: {missing_files}"
        )
    return regions_dictionary


# =========================================================================
# [2️⃣ 가상 메이크업용 검증 가드레일 - 💡 필요한 2개 영역만 필터링]
# =========================================================================
def fetch_and_validate_makeup_regions(files: List[FileItem]) -> dict:
    regions_dictionary = {}
    for file_item in files:
        # 프론트가 8개를 통째로 보내더라도, 메이크업에 필요한 2개만 쏙 골려서 다운로드 연산 수행
        if file_item.file_type in ["skin_region", "skin_mask"]:
            cv2_img = download_url_to_cv2(file_item.file_url)
            if cv2_img is not None:
                regions_dictionary[file_item.file_type] = cv2_img

    # 메이크업 필수 요소만 타이트하게 체크 ⭕
    required_files = ["skin_region", "skin_mask"]
    missing_files = [file for file in required_files if file not in regions_dictionary]
    if missing_files:
        raise HTTPException(
            status_code=400, 
            detail=f"가상 메이크업 합성에 필요한 필수 이미지 누락: {missing_files}"
        )
    return regions_dictionary


# =========================================================================
# [3️⃣ 퍼스널 컬러 진단 엔드포인트] - 강사님 피드백 반영 완결본 ⭕
# =========================================================================
@router.post("/personal-color")
def process_personal_color_only(payload: PersonalColorRequest):
    # 8개 영역 전체 검증 함수 호출
    regions_dictionary = fetch_and_validate_personal_color_regions(payload.files)
    color_result = color_analyzer.diagnose_from_regions(regions_dictionary)

    skin_hex = f"#{color_result['skin_rgb'][0]:02X}{color_result['skin_rgb'][1]:02X}{color_result['skin_rgb'][2]:02X}"

    # 🔥 [강사님 피드백 반영] 프론트가 보낸 8개 파일 중, 차후 메이크업에 사용할 2개만 필터링하여 담아줍니다.
    makeup_inputs = [
        file for file in payload.files 
        if file.file_type in ["skin_region", "skin_mask"]
    ]

    return {
        "status": "success",
        "user_id": payload.user_id,
        "personal_color": color_result["personal_color"],
        "detected_skin_hex": skin_hex,
        # 🔥 스프링과 프론트엔드가 기억할 수 있도록 메이크업 입력 원본 소스를 응답에 추가하여 전송합니다.
        "makeup_inputs": makeup_inputs 
    }


# =========================================================================
# [4️⃣ 가상 메이크업 합성 엔드포인트]
# =========================================================================
@router.post("/virtual-makeup")
def process_virtual_makeup_only(payload: VirtualMakeupRequest, request: Request):
    # 메이크업 전용 가드레일 함수로 전격 교체! ⭕
    regions_dictionary = fetch_and_validate_makeup_regions(payload.files)
    
    src_img = regions_dictionary["skin_region"]
    skin_mask = regions_dictionary["skin_mask"]

    # 헥스코드 변환 로직
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

    user_output_dir = os.path.join(BASE_OUTPUT_DIR, str(payload.user_id))
    os.makedirs(user_output_dir, exist_ok=True)

    unique_filename = f"makeup_{uuid.uuid4().hex[:8]}.jpg"
    file_save_path = os.path.join(user_output_dir, unique_filename)
    cv2.imwrite(file_save_path, makeup_img)

    base_url = str(request.base_url).rstrip("/")
    full_image_url = f"{base_url}/static/makeup_outputs/{payload.user_id}/{unique_filename}"

    original_image_url = next(
        (file.file_url for file in payload.files if file.file_type == "skin_region"), 
        None
    )

    return {
        "status": "success",
        "user_id": payload.user_id,
        "selected_foundation_hex": payload.target_foundation_hex,
        "original_image_url": original_image_url,
        "makeup_image_url": full_image_url
    }