from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from datetime import datetime

from database.db import Base

class GeneratedFileLog(Base):

    __tablename__ = "generated_file_log"

    # 생성 파일 ID (PK)
    file_id = Column(Integer, primary_key=True, index=True)

    # 원본 이미지 ID (saved_file_log FK)
    parent_file_id = Column(Integer, ForeignKey("saved_file_log.file_id"))

    # 생성된 파일 경로
    file_url = Column(String(255), nullable=False)

    # 파일 종류
    # ex) skin, forehead, lip, iris, eyebrow, makeup
    file_type = Column(String(20), nullable=False)

    # 생성 날짜
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    # 삭제 날짜
    deleted_at = Column(DateTime, nullable=True)

    # 삭제 여부
    # 0 : 정상, 1 : 삭제됨
    is_deleted = Column(Integer, default=0, nullable=False)