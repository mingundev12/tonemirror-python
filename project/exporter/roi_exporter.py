import cv2
import os
import logging

from dotenv import load_dotenv
from datetime import datetime

from models.generated_file_log import GeneratedFileLog
from schemas.generated_file_schema import ROIResponse


logger = logging.getLogger(__name__)


class ROIExporter:

    def __init__(self, db):
        self.db = db

        load_dotenv()

        # 이미지 루트 경로
        self.image_path = os.getenv("IMAGE_PATH")

        # 실제 ROI 저장 경로
        self.base_path = os.path.join(
            self.image_path,
            "roi"
        )

        # 외부에서 접근하는 URL
        self.base_url = os.getenv("BASE_URL")

        os.makedirs(
            self.base_path,
            exist_ok=True
        )

    def export(self, parent_file_id, roi_images):

        try:
            logger.info(
                f"ROI export started parent_file_id={parent_file_id}"
            )

            results = []

            # 파일명 중복 방지를 위한 timestamp
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")

            for file_type, image in roi_images.items():

                # 파일명 생성
                filename = (
                    f"{parent_file_id}_{timestamp}_{file_type}.png"
                )

                file_path = os.path.join(
                    self.base_path,
                    filename
                )

                # PNG 저장
                success = cv2.imwrite(
                    file_path,
                    image
                )

                if not success:
                    raise Exception(
                        f"{file_type} PNG 저장 실패"
                    )

                logger.info(
                    f"ROI PNG saved path={file_path}"
                )

                # 접근 URL 생성
                file_url = (
                    f"{self.base_url}/storage/roi/{filename}"
                )

                # DB Entity 생성
                generated_file = GeneratedFileLog(
                    parent_file_id=parent_file_id,
                    file_url=file_url,
                    file_type=file_type,
                    created_at=datetime.now(),
                    deleted_at=None,
                    is_deleted=0
                )

                self.db.add(generated_file)
                self.db.flush()

                logger.info(
                    f"ROI DB saved file_id={generated_file.file_id}, file_type={file_type}"
                )

                # Response 생성
                response = ROIResponse(
                    file_url=generated_file.file_url,
                    file_type=generated_file.file_type
                )

                results.append(response)

            self.db.commit()

            logger.info(
                f"ROI export completed parent_file_id={parent_file_id}, count={len(results)}"
            )

            return results

        except Exception as e:
            self.db.rollback()

            logger.exception(
                f"ROI export error: {e}"
            )

            raise