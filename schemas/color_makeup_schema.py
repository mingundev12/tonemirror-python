# schemas/color_makeup_schema.py
from pydantic import BaseModel, Field
from typing import List

class FileItem(BaseModel):
    file_type: str = Field(..., description="파일 타입 용도 (예: skin_region, skin_mask 등)", example="skin_region")
    file_url: str = Field(..., description="해당 파일에 접근 가능한 웹 URL 주소", example="http://127.0.0.1:8000/storage/roi/xxx.png")

# 1. 퍼스널 컬러 진단용 요청 DTO (전부 URL 리스트로 수신)
class PersonalColorRequest(BaseModel):
    original_image_id: str = Field(..., description="전처리 검수를 통과한 원본 이미지 고유 식별자 ID", example="img_87239aBc")
    files: List[FileItem] = Field(..., description="ROIExporter가 생성한 이미지 URL 객체 리스트")

# 2. 가상 메이크업 합성용 요청 DTO (전부 URL 리스트로 수신)
class VirtualMakeupRequest(BaseModel):
    original_image_id: str = Field(..., description="전처리 검수를 통과한 원본 이미지 고유 식별자 ID", example="img_87239aBc")
    target_foundation_hex: str = Field("#F5DEB3", description="사용자가 UI에서 선택한 파운데이션 헥스코드 (#포함)", example="#F5DEB3")
    files: List[FileItem] = Field(..., description="ROIExporter가 생성한 이미지 URL 객체 리스트")