import os
import uuid
import cv2
import numpy as np
import requests
from fastapi import APIRouter, HTTPException
from typing import List

from services.color_service import PersonalColorAnalyzer
from services.makeup_service import apply_full_foundation_stream
from schemas.color_makeup_schema import VirtualMakeupRequest, FileItem

router = APIRouter(prefix="/ai", tags=["Color & Makeup Processing Pipeline"])

color_analyzer = PersonalColorAnalyzer()
IMAGE_PATH = os.getenv("IMAGE_PATH", "/app/storage")
BASE_URL = os.getenv("BASE_URL", "http://localhost:28282/ai").rstrip("/")


def load_image_from_url(url_string: str) -> np.ndarray:
    """공유 볼륨 로컬 경로 우선, HTTP는 fallback."""
    if "/storage/" in url_string:
        relative_path = url_string.split("/storage/", 1)[1]
        local_path = os.path.join(IMAGE_PATH, relative_path)
        if os.path.isfile(local_path):
            image = cv2.imread(local_path)
            if image is not None:
                return image

    if os.path.isfile(url_string):
        image = cv2.imread(url_string)
        if image is not None:
            return image

    try:
        response = requests.get(url_string, timeout=5)
        response.raise_for_status()
        image_nparray = np.frombuffer(response.content, dtype=np.uint8)
        return cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"이미지 로드 실패 ({url_string}): {e}")
        return None


def fetch_and_validate_personal_color_regions(files: List[FileItem]) -> dict:
    regions_dictionary = {}
    for file_item in files:
        img = load_image_from_url(file_item.file_url)
        if img is not None:
            regions_dictionary[file_item.file_type] = img

    required_types = [
        "original_image", "skin_region", "skin_mask", "forehead_region",
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


@router.post("/virtual-makeup")
def process_virtual_makeup_only(payload: VirtualMakeupRequest):
    regions_dictionary = fetch_and_validate_personal_color_regions(payload.files)

    color_result = color_analyzer.diagnose_from_regions(regions_dictionary)
    skin_hex = f"#{color_result['skin_rgb'][0]:02X}{color_result['skin_rgb'][1]:02X}{color_result['skin_rgb'][2]:02X}"

    src_img = regions_dictionary["original_image"]
    skin_mask = regions_dictionary["skin_mask"]

    hex_str = (payload.target_foundation_hex or "#F5DEB3").lstrip('#')
    target_rgb = [int(hex_str[i:i+2], 16) for i in (0, 2, 4)]

    if len(skin_mask.shape) == 3:
        skin_mask = cv2.cvtColor(skin_mask, cv2.COLOR_BGR2GRAY)

    makeup_img = apply_full_foundation_stream(
        src_img=src_img,
        skin_mask=skin_mask,
        foundation_rgb=target_rgb,
        alpha=0.22
    )

    output_dir = os.path.join(IMAGE_PATH, "makeup_outputs", str(payload.original_image_id))
    os.makedirs(output_dir, exist_ok=True)

    unique_filename = f"makeup_{uuid.uuid4().hex[:8]}.jpg"
    file_save_path = os.path.join(output_dir, unique_filename)
    cv2.imwrite(file_save_path, makeup_img)

    full_image_url = f"{BASE_URL}/storage/makeup_outputs/{payload.original_image_id}/{unique_filename}"

    original_image_url = next(
        (file.file_url for file in payload.files if file.file_type == "original_image"),
        None
    )

    return {
        "status": "success",
        "original_image_id": payload.original_image_id,
        "personal_color": color_result["personal_color"],
        "detected_skin_hex": skin_hex,
        "selected_foundation_hex": payload.target_foundation_hex,
        "original_image_url": original_image_url,
        "makeup_image_url": full_image_url
    }
