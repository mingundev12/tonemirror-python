# preprocess_service.py
import cv2
import logging

from fastapi import HTTPException

from pipeline.preprocessing_pipeline import PreprocessingPipeline
from exporter.roi_exporter import ROIExporter


logger = logging.getLogger(__name__)


class PreprocessService:

    def __init__(self, db):
        self.db = db
        self.pipeline = PreprocessingPipeline()
        self.exporter = ROIExporter(db)

    def preprocess(self, request):

        try:
            logger.info(
                f"Preprocess started file_id={request.file_id}"
            )

            logger.info(
                f"Image Path = {request.file_url}"
            )

            # 1. 공유 Volume에 저장된 이미지 읽기
            image = cv2.imread(request.file_url)

            if image is None:
                raise HTTPException(
                    status_code=400,
                    detail="이미지를 읽을 수 없습니다."
                )

            # 2. 전처리 Pipeline 실행
            result = self.pipeline.run(image)

            if not result["passed"]:
                raise HTTPException(
                    status_code=400,
                    detail=result["message"]
                )

            # 3. ROI 저장 및 DB 기록
            generated_files = self.exporter.export(
                parent_file_id=request.file_id,
                roi_images=result["regions"]
            )

            logger.info(
                f"Preprocess completed file_id={request.file_id}"
            )

            return {
                "message": result["message"],
                "files": generated_files,
                "regions":result["regions"]
            }

        # 우리가 직접 발생시킨 예외는 그대로 전달
        except HTTPException:
            raise

        # 예상하지 못한 오류
        except Exception as e:
            logger.exception(
                f"Preprocess Error: {e}"
            )

            raise HTTPException(
                status_code=500,
                detail="전처리 중 내부 서버 오류가 발생했습니다."
            )