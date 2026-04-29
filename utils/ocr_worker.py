# -*- coding: utf-8 -*-
"""
OCR worker 函数，专供 ProcessPoolExecutor 使用。
每个 worker 进程在第一次调用时懒加载自己的 OCR / DocPreprocessor 实例，
后续任务直接复用，避免重复初始化开销。
"""

import os
from pathlib import Path
from typing import Any

import cv2

_ocr_instances = {}
_doc_preprocessor = None


def _configure_runtime() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(project_root / ".paddlex"))
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
    return project_root


def _get_ocr(use_doc_preprocessor: bool = False):
    global _ocr_instances
    cache_key = bool(use_doc_preprocessor)
    if cache_key in _ocr_instances:
        return _ocr_instances[cache_key]

    project_root = _configure_runtime()

    from paddleocr import PaddleOCR

    ocr_language = os.environ.get("OCR_LANGUAGE", "ch")
    ocr_model_dir = (
        Path(os.environ.get("OCR_MODEL_DIR", str(project_root / ".paddleocr")))
        .expanduser()
        .resolve()
    )

    def _resolve(env_name: str, default: Path) -> Path:
        value = os.environ.get(env_name)
        return Path(value).expanduser().resolve() if value else default

    model_dirs = {
        "text_detection_model_dir": _resolve(
            "OCR_TEXT_DETECTION_MODEL_DIR",
            ocr_model_dir / "ch_PP-OCRv4_det_infer",
        ),
        "text_recognition_model_dir": _resolve(
            "OCR_TEXT_RECOGNITION_MODEL_DIR",
            ocr_model_dir / "ch_PP-OCRv4_rec_infer",
        ),
        "textline_orientation_model_dir": _resolve(
            "OCR_TEXTLINE_ORIENTATION_MODEL_DIR",
            ocr_model_dir / "ch_ppocr_mobile_v2.0_cls_infer",
        ),
    }

    has_explicit = any(
        os.environ.get(env_name)
        for env_name in (
            "OCR_TEXT_DETECTION_MODEL_DIR",
            "OCR_TEXT_RECOGNITION_MODEL_DIR",
            "OCR_TEXTLINE_ORIENTATION_MODEL_DIR",
        )
    )
    has_local_ch = ocr_language == "ch" and all(
        model_dir.exists() for model_dir in model_dirs.values()
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
            lang=ocr_language,
        )

    _ocr_instances[cache_key] = ocr
    return ocr


def _get_doc_preprocessor():
    global _doc_preprocessor
    if _doc_preprocessor is not None:
        return _doc_preprocessor

    _configure_runtime()

    from paddleocr import DocPreprocessor

    _doc_preprocessor = DocPreprocessor(
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
    )
    return _doc_preprocessor


def _save_png(image: Any, output_path: Path) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("保存矫正后图片失败")
    output_path.write_bytes(encoded.tobytes())


def _run_doc_preprocessor(image_path: Path) -> tuple[Path, dict[str, Any]]:
    result = _get_doc_preprocessor().predict(
        str(image_path),
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
    )
    if not result:
        raise RuntimeError("文档矫正未返回任何结果")

    doc_result = result[0]
    output_img = doc_result.get("output_img")
    if output_img is None:
        raise RuntimeError("文档矫正结果缺少 output_img")

    corrected_path = image_path.parent / "corrected.png"
    _save_png(output_img, corrected_path)

    doc_meta = {
        "input_path": None,
        "page_index": doc_result.get("page_index"),
        "model_settings": doc_result.get("model_settings", {}),
        "angle": int(doc_result.get("angle", -1)),
    }
    return corrected_path, doc_meta


def run_ocr_file(image_path: str, use_doc_preprocessor: bool = False) -> dict[str, Any]:
    """在 worker 进程中执行 OCR 识别，结果为可序列化的 dict。
    此函数必须是顶层函数以支持跨进程 pickle。"""
    source_path = Path(image_path)
    ocr_input = str(source_path)
    ocr_image_variant = "original"
    doc_preprocessor_meta = None
    corrected_image_path = None

    if use_doc_preprocessor:
        corrected_path, doc_preprocessor_meta = _run_doc_preprocessor(source_path)
        ocr_input = str(corrected_path)
        ocr_image_variant = "corrected"
        corrected_image_path = str(corrected_path)

    # 显式预处理后，OCR 始终对当前输入图做识别，避免再次进入内部文档矫正链路。
    ocr = _get_ocr(use_doc_preprocessor=False)
    results = ocr.ocr(ocr_input)
    serialized = [result.json.get("res", result.json) for result in results]

    for item in serialized:
        model_settings = item.setdefault("model_settings", {})
        model_settings["use_doc_preprocessor"] = use_doc_preprocessor
        item["ocr_image_variant"] = ocr_image_variant
        if doc_preprocessor_meta is not None:
            item["doc_preprocessor_res"] = doc_preprocessor_meta

    return {
        "ocr_result": serialized,
        "ocr_image_variant": ocr_image_variant,
        "corrected_image_path": corrected_image_path,
    }
