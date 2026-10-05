"""MRI image endpoints (research prototype), mounted by app.py as a separate router.

    GET  /mri/info              classes, served model, metrics, dataset, disclaimer
    POST /predict-mri           multipart field "file": one 2D axial brain MRI slice (JPG/PNG/BMP/TIFF)
    POST /predict-mri/explain   same, plus a Grad-CAM overlay as base64 PNG (deep model only)

The model is loaded lazily on first use, so the tabular API starts fast and keeps working
when no image model or no torch/OpenCV is installed (these endpoints then return JSON 503).
The served model is described by src/imgProcessing/output/models/served/served.json
(written by src/imgProcessing/serve_model.py).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import threading
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

IMG_DIR = Path(__file__).resolve().parent / "imgProcessing"
if str(IMG_DIR) not in sys.path:
    sys.path.insert(0, str(IMG_DIR))

logger = logging.getLogger("alzheimer_api.mri")
router = APIRouter(tags=["MRI (research prototype)"])

DISCLAIMER = "Research prototype; not a medical diagnosis. Trained on a public, augmented 2D MRI dataset without clinical validation."
MAX_UPLOAD_BYTES = int(os.environ.get("MRI_MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/bmp", "image/x-ms-bmp", "image/tiff", "application/octet-stream", ""}
MIN_SIDE, MAX_SIDE = 32, 4096

_lock = threading.Lock()
_state: dict = {"loaded": False, "error": None}


def _served_dir() -> Path:
    import mri_config as C
    return C.SERVED_MODEL_DIR


def _load() -> dict:
    """Load the served model once. Returns the state dict; never raises."""
    with _lock:
        if _state["loaded"] or _state["error"]:
            return _state
        try:
            served = json.loads((_served_dir() / "served.json").read_text(encoding="utf-8"))
        except FileNotFoundError:  # not cached: a model served later is picked up without a restart
            return {"loaded": False, "error": ("no_model", "No MRI model has been trained/served yet. "
                                               "Run: python src/imgProcessing/serve_model.py")}
        except Exception as err:
            _state["error"] = ("bad_model", f"served.json unreadable: {err}")
            return _state
        try:
            from mri_preprocess import PreprocessSpec
            _state["served"] = served
            _state["spec"] = PreprocessSpec.from_dict(served["preprocess"])
            if served["kind"] == "deep":
                import torch
                from mri_dl_trainer import build_model
                dev = torch.device(os.environ.get("MRI_API_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
                model = build_model(served["arch"], len(served["class_names"]), served["config"]["dropout"], pretrained=False)
                model.load_state_dict(torch.load(_served_dir() / served["weights_file"], map_location=dev))
                _state.update(model=model.to(dev).eval(), device=dev)
            else:
                import joblib
                _state["model"] = joblib.load(_served_dir() / served["weights_file"])
            _state["loaded"] = True
            logger.info("MRI model loaded: %s", served.get("model_name"))
        except ImportError as err:
            _state["error"] = ("missing_dependency", f"Image dependencies missing ({err}). pip install -r requirements-image.txt")
        except Exception as err:
            logger.exception("MRI model failed to load")
            _state["error"] = ("load_failed", f"MRI model failed to load: {err}")
        return _state


def _unavailable(st: dict) -> JSONResponse:
    code, msg = st["error"]
    return JSONResponse(status_code=503, content={"error": msg, "code": code, "disclaimer": DISCLAIMER})


async def _read_image(file: UploadFile):
    """Validate type/size/decodability. Returns (gray uint8, None) or (None, JSONResponse)."""
    ctype = (file.content_type or "").lower()
    if ctype not in ALLOWED_TYPES:
        return None, JSONResponse(status_code=415, content={"error": f"Unsupported content type '{ctype}'. Send a JPG/PNG/BMP/TIFF image."})
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) == 0:
        return None, JSONResponse(status_code=400, content={"error": "Empty file."})
    if len(data) > MAX_UPLOAD_BYTES:
        return None, JSONResponse(status_code=413, content={"error": f"File larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."})
    try:
        from mri_preprocess import decode_gray
        gray = decode_gray(data)
    except ImportError as err:
        return None, JSONResponse(status_code=503, content={"error": f"Image dependencies missing ({err})."})
    except ValueError:
        return None, JSONResponse(status_code=400, content={"error": "File is not a decodable image."})
    h, w = gray.shape[:2]
    if min(h, w) < MIN_SIDE or max(h, w) > MAX_SIDE:
        return None, JSONResponse(status_code=400, content={"error": f"Image size {w}x{h} outside {MIN_SIDE}..{MAX_SIDE} px."})
    return gray, None


def _predict(st: dict, gray, explain: bool = False) -> dict:
    import numpy as np

    from mri_preprocess import preprocess_gray, to_tensor
    served = st["served"]
    img = preprocess_gray(gray, st["spec"])
    out = {"input_quality_warning": None}
    if float((img > 10).mean()) < 0.05:
        out["input_quality_warning"] = "Almost no foreground found; the image may not be a skull-stripped axial brain slice."
    if served["kind"] == "deep":
        import torch
        import torch.nn.functional as F
        x = to_tensor(img).unsqueeze(0).to(st["device"])
        size = served.get("image_size", 224)
        if size != img.shape[0]:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        with _lock:
            if explain:
                from mri_gradcam import gradcam, overlay_png
                cam, _, probs = gradcam(st["model"], served["arch"], x)
                vis = img if size == img.shape[0] else np.asarray(
                    F.interpolate(torch.from_numpy(img)[None, None].float(), size=(size, size), mode="bilinear")[0, 0].clamp(0, 255).byte())
                out["gradcam_png_base64"] = base64.b64encode(overlay_png(vis, cam)).decode("ascii")
            else:
                with torch.no_grad():
                    probs = torch.softmax(st["model"](x).float(), 1)[0].cpu().numpy()
    else:
        import cv2
        from mri_model_trainer import predict_proba
        art = st["model"]
        fs = art["feature_size"]
        feat = (cv2.resize(img, (fs, fs), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0).reshape(1, -1)
        probs = predict_proba(art["pipeline"], feat)[0]
    names = served["class_names"]
    k = int(np.argmax(probs))
    out.update({
        "predicted_class": names[k],
        "confidence": round(float(probs[k]), 4),
        "probabilities": {n: round(float(p), 4) for n, p in zip(names, probs)},
        "model": {"name": served.get("model_name"), "version": served.get("version"), "kind": served["kind"]},
        "disclaimer": DISCLAIMER,
    })
    return out


@router.get("/mri/info")
def mri_info():
    st = _load()
    if st["error"]:
        return {"model_available": False, "reason": st["error"][1], "disclaimer": DISCLAIMER}
    s = st["served"]
    return {"model_available": True, "class_names": s["class_names"],
            "model": {"name": s.get("model_name"), "version": s.get("version"), "kind": s["kind"], "arch": s.get("arch")},
            "metrics": s.get("metrics"), "dataset": s.get("dataset"), "evaluation": s.get("evaluation"),
            "input": "One 2D axial brain MRI slice, skull-stripped like the training data (JPG/PNG/BMP/TIFF, max "
                     f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
            "disclaimer": DISCLAIMER}


@router.post("/predict-mri")
async def predict_mri(file: UploadFile = File(...)):
    gray, err = await _read_image(file)
    if err:
        return err
    st = _load()
    if st["error"]:
        return _unavailable(st)
    try:
        return _predict(st, gray)
    except Exception as e:
        logger.exception("predict-mri failed")
        return JSONResponse(status_code=500, content={"error": f"Prediction failed: {e}"})


@router.post("/predict-mri/explain")
async def predict_mri_explain(file: UploadFile = File(...)):
    gray, err = await _read_image(file)
    if err:
        return err
    st = _load()
    if st["error"]:
        return _unavailable(st)
    if st["served"]["kind"] != "deep":
        return JSONResponse(status_code=409, content={"error": "Grad-CAM needs the deep model; the served model is classical."})
    try:
        return _predict(st, gray, explain=True)
    except Exception as e:
        logger.exception("predict-mri/explain failed")
        return JSONResponse(status_code=500, content={"error": f"Explanation failed: {e}"})
