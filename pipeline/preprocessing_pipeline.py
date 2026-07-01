from validator.photo_quality_validator import PhotoQualityValidator
from validator.accessory_validator import AccessoryValidator

from extractor.region_extractor import RegionExtractor

class PreprocessingPipeline:

    def __init__(self):

        self.photo_validator = PhotoQualityValidator()
        self.accessory_validator = AccessoryValidator()
        self.region_extractor = RegionExtractor()

    def run(self, image):

        # 1. blur 검사
        result = self.photo_validator.validate_blur_only(image)

        if not result["passed"]:
            return result
        
        # 2. 안경/마스크 검사
        result = self.accessory_validator.validate(image)

        if not result["passed"]:
            return result

        # 3. 얼굴 품질 검사
        result = self.photo_validator.validate_face(image)

        if not result["passed"]:
            return result
        
        # 4. ROI 추출
        regions = self.region_extractor.extract(image)

        if regions is None:
            return {
                "passed": False,
                "message": "ROI 추출 실패"
            }
        
        # 5. 피부 노출 검사
        passed, message = self.photo_validator.validate_exposure(
            regions["skin_region"]
        )

        if not passed:
            return {
                "passed": False,
                "message": message
            }
        
        # 6. 좌우 그림자 검사
        passed, message = self.photo_validator.validate_shadow(
            regions["left_cheek_region"],
            regions["right_cheek_region"]
        )

        if not passed:
            return {
                "passed": False,
                "message": message
            }
        
        # 7. 최종 성공
        return {
            "passed": True,
            "message": "전처리 완료",
            "regions": regions
        }