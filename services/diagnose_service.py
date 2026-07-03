# services/diagnose_service.py

from fastapi import HTTPException

from services.preprocess_service import PreprocessService
from services.color_service import PersonalColorAnalyzer

class DiagnoseService:

    def __init__(self, db):
        self.preprocess_service = PreprocessService(db)
        self.color_analyzer = PersonalColorAnalyzer()

    def diagnose(self, request):

        # 1. 전처리
        preprocess_result = self.preprocess_service.preprocess(request)

        # 2. 퍼스널 컬러 분석
        color_result = self.color_analyzer.diagnose_from_regions(
            preprocess_result["regions"]
        )

        if not color_result:
            raise HTTPException(
                status_code=500,
                detail="퍼스널 컬러 분석에 실패했습니다."
            )
        
        skin_rgb = color_result.get("skin_rgb")
        personal_color = color_result.get("personal_color")

        if skin_rgb is None or personal_color is None:
            raise HTTPException(
                status_code=500,
                detail="퍼스널 컬러 분석 결과가 올바르지 않습니다."
            )

        # 3. RGB -> HEX
        r, g, b = color_result["skin_rgb"]

        skin_hex = f"#{r:02X}{g:02X}{b:02X}"

        # 4. 최종 응답
        return {
            "message": preprocess_result["message"],
            "files": preprocess_result["files"],
            "personal_color": color_result["personal_color"],
            "detected_skin_hex": skin_hex
        }