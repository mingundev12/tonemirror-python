
from validator.face_mesh_validator import FaceMeshValidator
from extractor.skin_extractor import SkinExtractor
from extractor.iris_extractor import IrisExtractor
from extractor.forehead_extractor import ForeheadExtractor
from extractor.cheek_extractor import CheekExtractor


class RegionExtractor:

    def __init__(self):
        
        self.face_mesh_validator = FaceMeshValidator()
        self.skin_extractor = SkinExtractor()
        self.iris_extractor = IrisExtractor()
        self.forehead_extractor = ForeheadExtractor()
        self.cheek_extractor = CheekExtractor()

    
    # 전체 ROI 추출 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    
    def extract(self, image):

        height, width = image.shape[:2]

        face_landmarks = self.face_mesh_validator.get_landmarks(image)

        if face_landmarks is None:
            return None
        
        # Skin ROI
        skin_result = self.skin_extractor.extract(
            image,
            face_landmarks,
            width,
            height
        )

        skin_mask = skin_result["skin_mask"]

        # Iris ROI
        iris_result = self.iris_extractor.extract(
            image,
            face_landmarks,
            width,
            height
        )

        # Forehead ROI
        forehead_result = self.forehead_extractor.extract(
            image,
            face_landmarks,
            width,
            height,
            skin_mask
        )

        # Cheek ROI
        cheek_result = self.cheek_extractor.extract(
            image,
            face_landmarks,
            width,
            height,
            skin_mask
        )

        # 전체 region ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

        return {
            **skin_result,
            **iris_result,
            **forehead_result,
            **cheek_result
        }