from sqlalchemy import Column, Integer, String, DateTime
from database.db import Base

class SavedFileLog(Base):
    __tablename__ = "saved_file_log"

    file_id = Column(Integer, primary_key=True)

    file_url = Column(String(255), nullable=False)

    created_at = Column(DateTime, nullable=True)

    deleted_at = Column(DateTime, nullable=True)

    is_deleted = Column(Integer, nullable=True)
