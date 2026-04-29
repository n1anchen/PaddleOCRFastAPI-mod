# -*- coding: utf-8 -*-

import asyncio
import json
import mimetypes
import re
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config import RATE_LIMIT, UPLOAD_DIR
from database import SessionLocal, get_db
from limiter import get_client_ip, limiter
from models.TaskModel import Task
from utils.ocr_worker import run_ocr_file

# 进程池：max_workers=1 保证同一时间只有一个 OCR 进程，
# 可根据 CPU/GPU 资源适当调大。
_ocr_pool = ProcessPoolExecutor(max_workers=1)

router = APIRouter(prefix="/ocr", tags=["Tasks"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ── Response schemas ──────────────────────────────────────────────────────────

class TaskCreateResponse(BaseModel):
    task_id: str
    status: str


class TaskImageVariants(BaseModel):
    original: Optional[str] = None
    corrected: Optional[str] = None


class TaskDetailResponse(BaseModel):
    task_id: str
    ip: str
    created_at: datetime
    status: str
    original_filename: Optional[str]
    use_doc_preprocessor: bool = False
    image_variants: TaskImageVariants
    default_image_variant: str = "original"
    ocr_image_variant: str = "original"
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


def _task_image_files(task_dir: Path) -> dict[str, Optional[Path]]:
    original_candidates = list(task_dir.glob("original.*"))
    corrected_path = task_dir / "corrected.png"
    return {
        "original": original_candidates[0] if original_candidates else None,
        "corrected": corrected_path if corrected_path.exists() else None,
    }


def _build_image_variants(task_id: str, task_dir: Path) -> TaskImageVariants:
    files = _task_image_files(task_dir)
    return TaskImageVariants(
        original=(
            f"/ocr/tasks/{task_id}/image?variant=original"
            if files["original"] is not None
            else None
        ),
        corrected=(
            f"/ocr/tasks/{task_id}/image?variant=corrected"
            if files["corrected"] is not None
            else None
        ),
    )


def _resolve_ocr_image_variant(task: Task, ocr_result: Optional[Any]) -> str:
    if isinstance(ocr_result, list) and ocr_result:
        first_item = ocr_result[0]
        if isinstance(first_item, dict):
            variant = first_item.get("ocr_image_variant")
            if isinstance(variant, str) and variant:
                return variant
    return "corrected" if bool(task.use_doc_preprocessor) else "original"


def _build_task_response(task: Task, ocr_result: Optional[Any]) -> TaskDetailResponse:
    task_dir = Path(task.file_dir) if task.file_dir else None
    image_variants = (
        _build_image_variants(task.task_id, task_dir)
        if task_dir is not None
        else TaskImageVariants()
    )
    ocr_image_variant = _resolve_ocr_image_variant(task, ocr_result)
    default_image_variant = (
        "corrected"
        if bool(task.use_doc_preprocessor) and image_variants.corrected
        else "original"
    )

    return TaskDetailResponse(
        task_id=task.task_id,
        ip=task.ip,
        created_at=task.created_at,
        status=task.status,
        original_filename=task.original_filename,
        use_doc_preprocessor=bool(task.use_doc_preprocessor),
        image_variants=image_variants,
        default_image_variant=default_image_variant,
        ocr_image_variant=ocr_image_variant,
        ocr_result=ocr_result,
        error_msg=task.error_msg,
    )


async def _process_ocr_async(task_id: str, image_path: Path) -> None:
    """异步后台任务：DB 操作在主进程，OCR 推理在独立子进程（不阻塞事件循环）。"""
    db = SessionLocal()
    try:
        task: Task = db.query(Task).filter(Task.task_id == task_id).first()
        if not task:
            return

        use_doc_preprocessor = bool(task.use_doc_preprocessor)

        task.status = "processing"
        db.commit()

        # 将 CPU 密集的 OCR 推理送入独立进程，事件循环不阻塞
        loop = asyncio.get_running_loop()
        worker_payload = await loop.run_in_executor(
            _ocr_pool,
            run_ocr_file,
            str(image_path),
            use_doc_preprocessor,
        )

        result_json = json.dumps(worker_payload["ocr_result"], ensure_ascii=False)
        (image_path.parent / "result.json").write_text(result_json, encoding="utf-8")

        task = db.query(Task).filter(Task.task_id == task_id).first()
        if task:
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
    use_doc_preprocessor: bool = Form(False),
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

    task = Task(
        task_id=task_id,
        ip=ip,
        created_at=datetime.utcnow(),
        status="pending",
        original_filename=file.filename,
        file_dir=str(task_dir),
        use_doc_preprocessor=use_doc_preprocessor,
    )
    db.add(task)
    db.commit()

    background_tasks.add_task(_process_ocr_async, task_id, image_path)

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

    return _build_task_response(task, ocr_result)


@router.get(
    "/tasks/{task_id}/image",
    summary="获取任务图片",
)
def get_task_image(
    task_id: str,
    variant: str = "original",
    db: Session = Depends(get_db),
):
    task: Task = db.query(Task).filter(Task.task_id == task_id).first()
    if not task or not task.file_dir:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    task_dir = Path(task.file_dir)
    image_files = _task_image_files(task_dir)
    if variant not in image_files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="不支持的图片变体")
    image_file = image_files[variant]
    if image_file is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="图片不存在")

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
            _build_task_response(task, ocr_result)
        )

    return TaskListResponse(total=total, page=page, page_size=page_size, items=items)
