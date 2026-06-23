import os
import logging

from dotenv import load_dotenv

from datetime import datetime
from fastapi import HTTPException

from models.generated_file_log import GeneratedFileLog


logger = logging.getLogger(__name__)


class ROIService:

    def __init__(self, db):
        self.db = db

        load_dotenv()

        self.image_path = os.getenv("IMAGE_PATH")

    def delete_roi(self, parent_file_id: int):

        try:
            logger.info(
                f"ROI delete started parent_file_id={parent_file_id}"
            )

            # 1. 해당 원본 이미지의 ROI 전체 조회
            roi_files = (
                self.db.query(GeneratedFileLog)
                .filter(
                    GeneratedFileLog.parent_file_id == parent_file_id,
                    GeneratedFileLog.is_deleted == 0
                )
                .all()
            )

            if not roi_files:
                raise HTTPException(
                    status_code=404,
                    detail="삭제할 ROI 파일이 존재하지 않습니다."
                )

            # 2. ROI 파일 전체 삭제 및 DB Soft Delete
            for roi in roi_files:

                # URL에서 storage 이후 경로 추출
                relative_path = roi.file_url.split(
                    "/storage/"
                )[1]

                # 실제 파일 경로 생성
                file_path = os.path.join(
                    self.image_path,
                    "storage",
                    relative_path
                )

                # PNG 파일 삭제
                if os.path.exists(file_path):
                    os.remove(file_path)

                    logger.info(
                        f"ROI file deleted path={file_path}"
                    )

                # DB Soft Delete 처리
                roi.is_deleted = 1
                roi.deleted_at = datetime.now()

            # 3. 변경사항 저장
            self.db.commit()

            logger.info(
                f"ROI delete completed parent_file_id={parent_file_id}, count={len(roi_files)}"
            )

            return {
                "message": "ROI 파일이 전체 삭제되었습니다."
            }

        except HTTPException:
            raise

        except Exception as e:
            logger.exception(
                f"ROI delete error: {e}"
            )

            raise HTTPException(
                status_code=500,
                detail="ROI 삭제 중 내부 서버 오류가 발생했습니다."
            )