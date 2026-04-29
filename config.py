# -*- coding: utf-8 -*-

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./ocr_tasks.db")
UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "./uploads"))
RATE_LIMIT: str = os.getenv("RATE_LIMIT", "10/minute")
# 同时处理的 OCR 任务上限；其余任务排入 queued 等待队列
MAX_CONCURRENT_OCR: int = int(os.getenv("MAX_CONCURRENT_OCR", "1"))
