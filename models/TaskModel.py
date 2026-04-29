# -*- coding: utf-8 -*-

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, String, Text

from database import Base


class Task(Base):
    __tablename__ = "tasks"

    task_id = Column(String(36), primary_key=True, index=True)
    ip = Column(String(64), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default="pending")  # pending / processing / done / failed
    original_filename = Column(String(255), nullable=True)
    file_dir = Column(String(512), nullable=True)
    use_doc_preprocessor = Column(Boolean, nullable=False, default=False)
    ocr_result = Column(Text, nullable=True)   # JSON 字符串
    error_msg = Column(Text, nullable=True)
