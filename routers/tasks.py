# -*- coding: utf-8 -*-

import asyncio
import json
import logging
import mimetypes
import re
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config import MAX_CONCURRENT_OCR, RATE_LIMIT, UPLOAD_DIR
from database import SessionLocal, get_db
from limiter import get_client_ip, limiter
from models.TaskModel import Task
from utils.ocr_worker import run_ocr_file

logger = logging.getLogger(__name__)

# 进程池：max_workers 与 MAX_CONCURRENT_OCR 保持一致
_ocr_pool = ProcessPoolExecutor(max_workers=MAX_CONCURRENT_OCR)

# 应用层任务队列：存放 (task_id, image_path) 元组
_task_queue: asyncio.Queue[tuple[str, Path]] = asyncio.Queue()
# 长运行 worker 协程的 Task 句柄，用于 lifespan 关闭时取消
_worker_tasks: list[asyncio.Task] = []

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
    queue_position: Optional[int] = None  # queued 时的排队位置（1-based）


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


def _build_task_response(
    task: Task,
    ocr_result: Optional[Any],
    queue_position: Optional[int] = None,
) -> TaskDetailResponse:
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
        queue_position=queue_position,
    )


# ── Queue workers ─────────────────────────────────────────────────────────────

async def _ocr_queue_worker() -> None:
    """长运行协程：从队列中取任务并依次处理。"""
    while True:
        task_id, image_path = await _task_queue.get()
        try:
            await _process_ocr_async(task_id, image_path)
        except Exception:
            logger.exception("OCR worker 遇到未捕获异常，task_id=%s", task_id)
        finally:
            _task_queue.task_done()


async def start_workers() -> None:
    """启动 MAX_CONCURRENT_OCR 个 worker 协程，并将 DB 中残留的 queued 任务重新入队。"""
    for _ in range(MAX_CONCURRENT_OCR):
        t = asyncio.create_task(_ocr_queue_worker())
        _worker_tasks.append(t)

    # 崩溃恢复：将上次运行中未完成的 queued/processing 任务重新入队
    db = SessionLocal()
    try:
        stuck = (
            db.query(Task)
            .filter(Task.status.in_(["queued", "processing"]))
            .order_by(Task.created_at)
            .all()
        )
        for task in stuck:
            if task.file_dir:
                task_dir = Path(task.file_dir)
                candidates = list(task_dir.glob("original.*"))
                if candidates:
                    task.status = "queued"  # 统一重置为 queued
                    db.commit()
                    await _task_queue.put((task.task_id, candidates[0]))
                    logger.info("崩溃恢复：重新入队 task_id=%s", task.task_id)
    finally:
        db.close()


async def stop_workers() -> None:
    """取消所有 worker 协程，等待队列排空。"""
    for t in _worker_tasks:
        t.cancel()
    await asyncio.gather(*_worker_tasks, return_exceptions=True)
    _worker_tasks.clear()


async def _process_ocr_async(task_id: str, image_path: Path) -> None:
    """异步协程：将 queued 任务转为 processing 再送入进程池推理。"""
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
        status="queued",
        original_filename=file.filename,
        file_dir=str(task_dir),
        use_doc_preprocessor=use_doc_preprocessor,
    )
    db.add(task)
    db.commit()

    await _task_queue.put((task_id, image_path))

    return TaskCreateResponse(task_id=task_id, status="queued")


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

    # 计算排队位置：比本任务更早入队（created_at 更小）且仍在 queued 的任务数 + 1
    queue_position: Optional[int] = None
    if task.status == "queued":
        ahead = (
            db.query(Task)
            .filter(Task.status == "queued", Task.created_at < task.created_at)
            .count()
        )
        queue_position = ahead + 1

    return _build_task_response(task, ocr_result, queue_position)


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
