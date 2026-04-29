# -*- coding: utf-8 -*-
"""
OCR worker 函数，专供 ProcessPoolExecutor 使用。
每个 worker 进程在第一次调用时懒加载自己的 OCR 模型实例，
后续任务直接复用，避免重复初始化开销。
"""

import os
from pathlib import Path

_ocr_instances = {}


def _get_ocr(use_doc_preprocessor: bool = False):
    global _ocr_instances
    cache_key = bool(use_doc_preprocessor)
    if cache_key in _ocr_instances:
        return _ocr_instances[cache_key]

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PROJECT_ROOT / ".paddlex"))
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")

    from paddleocr import PaddleOCR

    OCR_LANGUAGE = os.environ.get("OCR_LANGUAGE", "ch")
    OCR_MODEL_DIR = (
        Path(os.environ.get("OCR_MODEL_DIR", str(PROJECT_ROOT / ".paddleocr")))
        .expanduser()
        .resolve()
    )

    def _resolve(env_name: str, default: Path) -> Path:
        v = os.environ.get(env_name)
        return Path(v).expanduser().resolve() if v else default

    model_dirs = {
        "text_detection_model_dir": _resolve(
            "OCR_TEXT_DETECTION_MODEL_DIR",
            OCR_MODEL_DIR / "ch_PP-OCRv4_det_infer",
        ),
        "text_recognition_model_dir": _resolve(
            "OCR_TEXT_RECOGNITION_MODEL_DIR",
            OCR_MODEL_DIR / "ch_PP-OCRv4_rec_infer",
        ),
        "textline_orientation_model_dir": _resolve(
            "OCR_TEXTLINE_ORIENTATION_MODEL_DIR",
            OCR_MODEL_DIR / "ch_ppocr_mobile_v2.0_cls_infer",
        ),
    }

    has_explicit = any(
        os.environ.get(e)
        for e in (
            "OCR_TEXT_DETECTION_MODEL_DIR",
            "OCR_TEXT_RECOGNITION_MODEL_DIR",
            "OCR_TEXTLINE_ORIENTATION_MODEL_DIR",
        )
    )
    has_local_ch = OCR_LANGUAGE == "ch" and all(
        d.exists() for d in model_dirs.values()
    )

    preprocess_kwargs = {
        "use_doc_orientation_classify": cache_key,
        "use_doc_unwarping": cache_key,
    }

    if has_explicit or has_local_ch:
        ocr = PaddleOCR(
            use_textline_orientation=True,
            **preprocess_kwargs,
            text_detection_model_dir=str(model_dirs["text_detection_model_dir"]),
            text_recognition_model_dir=str(model_dirs["text_recognition_model_dir"]),
            textline_orientation_model_dir=str(
                model_dirs["textline_orientation_model_dir"]
            ),
        )
    else:
        ocr = PaddleOCR(
            use_angle_cls=True,
            **preprocess_kwargs,
            lang=OCR_LANGUAGE,
        )

    _ocr_instances[cache_key] = ocr
    return ocr


def run_ocr_file(image_path: str, use_doc_preprocessor: bool = False) -> list:
    """在 worker 进程中执行 OCR 识别，结果为可序列化的 list。
    此函数必须是顶层函数以支持跨进程 pickle。"""
    ocr = _get_ocr(use_doc_preprocessor=use_doc_preprocessor)
    results = ocr.ocr(image_path)
    return [r.json.get("res", r.json) for r in results]
