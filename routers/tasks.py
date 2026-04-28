# -*- coding: utf-8 -*-

import json
import mimetypes
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config import RATE_LIMIT, UPLOAD_DIR
from database import get_db
from limiter import get_client_ip, limiter
from models.TaskModel import Task
from routers.ocr import _run_ocr

router = APIRouter(prefix="/ocr", tags=["Tasks"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ── Response schemas ──────────────────────────────────────────────────────────

class TaskCreateResponse(BaseModel):
    task_id: str
    status: str


class TaskDetailResponse(BaseModel):
    task_id: str
    ip: str
    created_at: datetime
    status: str
    original_filename: Optional[str]
    ocr_result: Optional[Any]
    error_msg: Optional[str]


class TaskListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[TaskDetailResponse]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitize_ip(ip: str) -> str:
    """将 IP 地址转换为安全的文件夹名称片段（替换非字母数字字符为下划线）。"""
    return re.sub(r"[^a-zA-Z0-9]", "_", ip)


def _task_dir(ip: str, task_id: str) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    folder = UPLOAD_DIR / f"{_sanitize_ip(ip)}_{task_id}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _process_ocr(task_id: str, image_path: Path, db_url: str):
    """在后台线程中执行 OCR 并更新数据库。"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    engine = create_engine(db_url, connect_args=connect_args)
    LocalSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = LocalSession()

    try:
        task: Task = db.query(Task).filter(Task.task_id == task_id).first()
        if not task:
            return

        task.status = "processing"
        db.commit()

        result = _run_ocr(str(image_path))
        result_json = json.dumps(result, ensure_ascii=False)

        result_file = image_path.parent / "result.json"
        result_file.write_text(result_json, encoding="utf-8")

        task.ocr_result = result_json
        task.status = "done"
        db.commit()

    except Exception as exc:
        db.rollback()
        task = db.query(Task).filter(Task.task_id == task_id).first()
        if task:
            task.status = "failed"
            task.error_msg = str(exc)
            db.commit()
    finally:
        db.close()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/tasks",
    response_model=TaskCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="创建 OCR 任务（异步）",
)
@limiter.limit(RATE_LIMIT)
async def create_task(
    request: Request,
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件格式，请上传: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    ip = get_client_ip(request)
    task_id = str(uuid.uuid4())
    task_dir = _task_dir(ip, task_id)

    image_path = task_dir / f"original{suffix}"
    contents = await file.read()
    image_path.write_bytes(contents)

    from config import DATABASE_URL
    task = Task(
        task_id=task_id,
        ip=ip,
        created_at=datetime.utcnow(),
        status="pending",
        original_filename=file.filename,
        file_dir=str(task_dir),
    )
    db.add(task)
    db.commit()

    background_tasks.add_task(_process_ocr, task_id, image_path, DATABASE_URL)

    return TaskCreateResponse(task_id=task_id, status="pending")


@router.get(
    "/tasks/{task_id}",
    response_model=TaskDetailResponse,
    summary="查询 OCR 任务状态与结果",
)
def get_task(task_id: str, db: Session = Depends(get_db)):
    task: Task = db.query(Task).filter(Task.task_id == task_id).first()
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    ocr_result = None
    if task.ocr_result:
        try:
            ocr_result = json.loads(task.ocr_result)
        except json.JSONDecodeError:
            ocr_result = task.ocr_result

    return TaskDetailResponse(
        task_id=task.task_id,
        ip=task.ip,
        created_at=task.created_at,
        status=task.status,
        original_filename=task.original_filename,
        ocr_result=ocr_result,
        error_msg=task.error_msg,
    )


@router.get(
    "/tasks/{task_id}/image",
    summary="获取任务原始图片",
)
def get_task_image(task_id: str, db: Session = Depends(get_db)):
    task: Task = db.query(Task).filter(Task.task_id == task_id).first()
    if not task or not task.file_dir:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    task_dir = Path(task.file_dir)
    # Find the original image file (original.*)
    candidates = list(task_dir.glob("original.*"))
    if not candidates:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="原始图片不存在")

    image_file = candidates[0]
    media_type, _ = mimetypes.guess_type(image_file.name)
    return FileResponse(path=str(image_file), media_type=media_type or "image/jpeg")


@router.get(
    "/tasks",
    response_model=TaskListResponse,
    summary="查询当前 IP 的历史任务列表",
)
def list_tasks(
    request: Request,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
):
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 100:
        page_size = 20

    ip = get_client_ip(request)
    query = db.query(Task).filter(Task.ip == ip).order_by(Task.created_at.desc())
    total = query.count()
    tasks = query.offset((page - 1) * page_size).limit(page_size).all()

    items = []
    for task in tasks:
        ocr_result = None
        if task.ocr_result:
            try:
                ocr_result = json.loads(task.ocr_result)
            except json.JSONDecodeError:
                ocr_result = task.ocr_result
        items.append(
            TaskDetailResponse(
                task_id=task.task_id,
                ip=task.ip,
                created_at=task.created_at,
                status=task.status,
                original_filename=task.original_filename,
                ocr_result=ocr_result,
                error_msg=task.error_msg,
            )
        )

    return TaskListResponse(total=total, page=page, page_size=page_size, items=items)
