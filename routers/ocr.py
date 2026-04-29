# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Any

import requests
from fastapi import APIRouter, Form, HTTPException, UploadFile, status

from models.OCRModel import *
from models.RestfulModel import *
from utils.ImageHelper import base64_to_ndarray, bytes_to_ndarray

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDX_CACHE_HOME = PROJECT_ROOT / ".paddlex"
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(DEFAULT_PDX_CACHE_HOME))
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")

from paddleocr import DocPreprocessor, PaddleOCR

OCR_LANGUAGE = os.environ.get("OCR_LANGUAGE", "ch")
OCR_MODEL_DIR = Path(
    os.environ.get("OCR_MODEL_DIR", str(PROJECT_ROOT / ".paddleocr"))
).expanduser().resolve()

DEFAULT_MODEL_DIRS = {
    "text_detection_model_dir": OCR_MODEL_DIR / "ch_PP-OCRv4_det_infer",
    "text_recognition_model_dir": OCR_MODEL_DIR / "ch_PP-OCRv4_rec_infer",
    "textline_orientation_model_dir": OCR_MODEL_DIR / "ch_ppocr_mobile_v2.0_cls_infer",
}

router = APIRouter(prefix="/ocr", tags=["OCR"])

_ocr_instance: PaddleOCR | None = None
_doc_preprocessor: DocPreprocessor | None = None


def _resolve_model_dir(env_name: str, default_dir: Path) -> Path:
    configured_dir = os.environ.get(env_name)
    if configured_dir:
        return Path(configured_dir).expanduser().resolve()
    return default_dir


def _build_ocr() -> PaddleOCR:
    OCR_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_dirs = {
        name: _resolve_model_dir(env_name, default_dir)
        for name, env_name, default_dir in (
            (
                "text_detection_model_dir",
                "OCR_TEXT_DETECTION_MODEL_DIR",
                DEFAULT_MODEL_DIRS["text_detection_model_dir"],
            ),
            (
                "text_recognition_model_dir",
                "OCR_TEXT_RECOGNITION_MODEL_DIR",
                DEFAULT_MODEL_DIRS["text_recognition_model_dir"],
            ),
            (
                "textline_orientation_model_dir",
                "OCR_TEXTLINE_ORIENTATION_MODEL_DIR",
                DEFAULT_MODEL_DIRS["textline_orientation_model_dir"],
            ),
        )
    }

    has_explicit_model_dirs = any(
        os.environ.get(env_name)
        for env_name in (
            "OCR_TEXT_DETECTION_MODEL_DIR",
            "OCR_TEXT_RECOGNITION_MODEL_DIR",
            "OCR_TEXTLINE_ORIENTATION_MODEL_DIR",
        )
    )
    has_local_chinese_models = OCR_LANGUAGE == "ch" and all(
        model_dir.exists() for model_dir in model_dirs.values()
    )

    if has_explicit_model_dirs or has_local_chinese_models:
        return PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            text_detection_model_dir=str(model_dirs["text_detection_model_dir"]),
            text_recognition_model_dir=str(model_dirs["text_recognition_model_dir"]),
            textline_orientation_model_dir=str(model_dirs["textline_orientation_model_dir"]),
        )

    return PaddleOCR(
        use_angle_cls=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        lang=OCR_LANGUAGE,
    )


def _get_ocr() -> PaddleOCR:
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = _build_ocr()
    return _ocr_instance


def _get_doc_preprocessor() -> DocPreprocessor:
    global _doc_preprocessor
    if _doc_preprocessor is None:
        _doc_preprocessor = DocPreprocessor(
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
        )
    return _doc_preprocessor


def _preprocess_image(image: Any) -> tuple[Any, dict[str, Any]]:
    result = _get_doc_preprocessor().predict(
        image,
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
    )
    if not result:
        raise RuntimeError("文档矫正未返回任何结果")

    doc_result = result[0]
    output_img = doc_result.get("output_img")
    if output_img is None:
        raise RuntimeError("文档矫正结果缺少 output_img")

    doc_meta = {
        "input_path": None,
        "page_index": doc_result.get("page_index"),
        "model_settings": doc_result.get("model_settings", {}),
        "angle": int(doc_result.get("angle", -1)),
    }
    return output_img, doc_meta


def _run_ocr(image: Any, use_doc_preprocessor: bool = False):
    ocr_input = image
    ocr_image_variant = "original"
    doc_preprocessor_meta = None

    if use_doc_preprocessor:
        ocr_input, doc_preprocessor_meta = _preprocess_image(image)
        ocr_image_variant = "corrected"

    results = _get_ocr().ocr(ocr_input)
    serialized = [result.json.get("res", result.json) for result in results]

    for item in serialized:
        model_settings = item.setdefault("model_settings", {})
        model_settings["use_doc_preprocessor"] = use_doc_preprocessor
        item["ocr_image_variant"] = ocr_image_variant
        if doc_preprocessor_meta is not None:
            item["doc_preprocessor_res"] = doc_preprocessor_meta

    return serialized


@router.get('/predict-by-path', response_model=RestfulModel, summary="识别本地图片")
def predict_by_path(image_path: str, use_doc_preprocessor: bool = False):
    result = _run_ocr(image_path, use_doc_preprocessor=use_doc_preprocessor)
    restfulModel = RestfulModel(
        resultcode=200, message="Success", data=result, cls=OCRModel)
    return restfulModel


@router.post('/predict-by-base64', response_model=RestfulModel, summary="识别 Base64 数据")
def predict_by_base64(base64model: Base64PostModel, use_doc_preprocessor: bool = False):
    img = base64_to_ndarray(base64model.base64_str)
    result = _run_ocr(img, use_doc_preprocessor=use_doc_preprocessor)
    restfulModel = RestfulModel(
        resultcode=200, message="Success", data=result, cls=OCRModel)
    return restfulModel


@router.post('/predict-by-file', response_model=RestfulModel, summary="识别上传文件")
async def predict_by_file(file: UploadFile, use_doc_preprocessor: bool = Form(False)):
    restfulModel: RestfulModel = RestfulModel()
    if file.filename.endswith((".jpg", ".png")):
        restfulModel.resultcode = 200
        restfulModel.message = file.filename
        file_bytes = file.file.read()
        img = bytes_to_ndarray(file_bytes)
        result = _run_ocr(img, use_doc_preprocessor=use_doc_preprocessor)
        restfulModel.data = result
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请上传 .jpg 或 .png 格式图片"
        )
    return restfulModel


@router.get('/predict-by-url', response_model=RestfulModel, summary="识别图片 URL")
async def predict_by_url(imageUrl: str, use_doc_preprocessor: bool = False):
    restfulModel: RestfulModel = RestfulModel()
    response = requests.get(imageUrl)
    image_bytes = response.content
    if image_bytes.startswith(b"\xff\xd8\xff") or image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        restfulModel.resultcode = 200
        img = bytes_to_ndarray(image_bytes)
        result = _run_ocr(img, use_doc_preprocessor=use_doc_preprocessor)
        restfulModel.data = result
        restfulModel.message = "Success"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请上传 .jpg 或 .png 格式图片"
        )
    return restfulModel