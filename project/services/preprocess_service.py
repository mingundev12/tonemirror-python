import cv2
import numpy as np
import requests
import logging

from models.saved_file_log import SavedFileLog
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

            # 1. 원본 이미지 존재 여부 확인
            saved_file = (
                self.db.query(SavedFileLog)
                .filter(
                    SavedFileLog.file_id == request.file_id,
                    SavedFileLog.is_deleted == 0
                )
                .first()
            )

            if saved_file is None:
                raise HTTPException(
                    status_code=404,
                    detail="원본 이미지가 존재하지 않습니다."
                )

            # 2. 이미지 다운로드
            try:
                response = requests.get(
                    request.file_url,
                    timeout=5
                )
                response.raise_for_status()

            except requests.exceptions.Timeout:
                raise HTTPException(
                    status_code=408,
                    detail="이미지 다운로드 시간이 초과되었습니다."
                )

            except requests.exceptions.ConnectionError:
                raise HTTPException(
                    status_code=503,
                    detail="이미지 서버에 연결할 수 없습니다."
                )

            except requests.exceptions.RequestException:
                raise HTTPException(
                    status_code=400,
                    detail="올바르지 않은 이미지 URL입니다."
                )

            # 3. OpenCV 이미지 변환
            image_array = np.frombuffer(
                response.content,
                np.uint8
            )

            image = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )

            if image is None:
                raise HTTPException(
                    status_code=400,
                    detail="올바르지 않은 이미지 파일입니다."
                )

            # 4. 전처리 Pipeline 실행
            result = self.pipeline.run(image)

            if not result["passed"]:
                raise HTTPException(
                    status_code=400,
                    detail=result["message"]
                )

            # 5. ROI 저장 및 DB 기록
            generated_files = self.exporter.export(
                parent_file_id=request.file_id,
                roi_images=result["regions"]
            )

            # 6. 결과 반환
            logger.info(
                f"Preprocess completed file_id={request.file_id}"
            )

            return {
                "message": result["message"],
                "files": generated_files
            }

        # 우리가 직접 만든 예외는 그대로 전달
        except HTTPException:
            raise

        # 예상하지 못한 모든 오류 처리
        except Exception as e:
            logger.exception(
                f"Preprocess Error: {e}"
            )

            raise HTTPException(
                status_code=500,
                detail="전처리 중 내부 서버 오류가 발생했습니다."
            )