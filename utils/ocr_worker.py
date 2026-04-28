# -*- coding: utf-8 -*-
"""
OCR worker 函数，专供 ProcessPoolExecutor 使用。
每个 worker 进程在第一次调用时懒加载自己的 OCR 模型实例，
后续任务直接复用，避免重复初始化开销。
"""

import os
from pathlib import Path

_ocr_instance = None


def _get_ocr():
    global _ocr_instance
    if _ocr_instance is not None:
        return _ocr_instance

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

    if has_explicit or has_local_ch:
        _ocr_instance = PaddleOCR(
            use_textline_orientation=True,
            text_detection_model_dir=str(model_dirs["text_detection_model_dir"]),
            text_recognition_model_dir=str(model_dirs["text_recognition_model_dir"]),
            textline_orientation_model_dir=str(
                model_dirs["textline_orientation_model_dir"]
            ),
        )
    else:
        _ocr_instance = PaddleOCR(use_angle_cls=True, lang=OCR_LANGUAGE)

    return _ocr_instance


def run_ocr_file(image_path: str) -> list:
    """在 worker 进程中执行 OCR 识别，结果为可序列化的 list。
    此函数必须是顶层函数以支持跨进程 pickle。"""
    ocr = _get_ocr()
    results = ocr.ocr(image_path)
    return [r.json.get("res", r.json) for r in results]
