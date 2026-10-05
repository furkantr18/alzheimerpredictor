"""API tests for the MRI endpoints (stdlib only, synthetic images). Server must be running:
    _backend\\START_BACKEND.bat        (or: .venv\\Scripts\\python.exe -m uvicorn src.app:app --port 8000)
    .venv\\Scripts\\python.exe tests\\test_mri_api.py [--expect-model | --expect-no-model]

Checks: /mri/info, valid PNG/JPG upload, CORS header, explain overlay, and the error paths
(empty, not an image, wrong content type, too large, tiny image), plus that the tabular
handler still answers.
"""

from __future__ import annotations

import base64
import json
import sys
import urllib.error
import urllib.request
import uuid

import numpy as np

BASE = "http://127.0.0.1:8000"


def synthetic_png(h=208, w=176, fmt=".png") -> bytes:
    import cv2
    img = np.zeros((h, w), np.uint8)
    cv2.ellipse(img, (w // 2, h // 2), (max(w // 2 - 12, 2), max(h // 2 - 10, 2)), 0, 0, 360, 170, -1)
    cv2.ellipse(img, (w // 2, h // 2), (max(w // 10, 1), max(h // 7, 1)), 0, 0, 360, 30, -1)  # "ventricles"
    noise = np.random.default_rng(0).normal(0, 12, img.shape)
    img = np.clip(img + noise * (img > 0), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(fmt, img)
    return buf.tobytes()


def post_file(path: str, data: bytes, ctype: str = "image/png", filename: str = "slice.png"):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(BASE + path, data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}", "Origin": "https://oracleapex.com"})
    return _send(req)


def get(path: str):
    return _send(urllib.request.Request(BASE + path, headers={"Origin": "https://oracleapex.com"}))


def _send(req):
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read() or b"{}"), dict(r.headers)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw), dict(e.headers)
        except ValueError:
            return e.code, {"raw": raw[:200].decode(errors="replace")}, dict(e.headers)


def main() -> int:
    expect = "model" if "--expect-model" in sys.argv else ("none" if "--expect-no-model" in sys.argv else "any")
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))

    s, j, h = get("/patients_data_handler")
    check("tabular handler still answers", s == 200 and "working" in json.dumps(j), s)

    s, info, _ = get("/mri/info")
    check("/mri/info returns JSON 200 with disclaimer", s == 200 and "disclaimer" in info, s)
    has_model = bool(info.get("model_available"))
    if expect == "model":
        check("model is available", has_model, info.get("reason", ""))
    if expect == "none":
        check("no model reported cleanly", not has_model, info.get("reason", ""))

    s, j, h = post_file("/predict-mri", synthetic_png())
    if has_model:
        probs = j.get("probabilities", {})
        check("/predict-mri PNG -> 200 + class + probabilities", s == 200 and j.get("predicted_class") in probs and len(probs) == 4,
              f"{s} {j.get('predicted_class')}")
        check("probabilities sum to ~1", abs(sum(probs.values()) - 1) < 1e-3, round(sum(probs.values()), 4))
        check("disclaimer present", "not a medical diagnosis" in j.get("disclaimer", ""))
        check("model name/version present", bool(j.get("model", {}).get("name")) and bool(j.get("model", {}).get("version")))
        hl = {k.lower(): v for k, v in h.items()}
        # same CORS behaviour as the tabular endpoints (Starlette echoes the Origin when credentials are allowed)
        check("CORS + private-network headers on response", hl.get("access-control-allow-origin") in ("*", "https://oracleapex.com")
              and hl.get("access-control-allow-private-network") == "true", hl.get("access-control-allow-origin"))
        s2, j2, _ = post_file("/predict-mri", synthetic_png(fmt=".jpg"), "image/jpeg", "slice.jpg")
        check("/predict-mri JPEG -> 200", s2 == 200, s2)
        s3, j3, _ = post_file("/predict-mri/explain", synthetic_png())
        if s3 == 200:
            png = base64.b64decode(j3.get("gradcam_png_base64", ""))
            check("/predict-mri/explain -> base64 PNG overlay", png[:8] == b"\x89PNG\r\n\x1a\n", len(png))
        else:
            check("/predict-mri/explain -> 409 for classical model", s3 == 409, s3)
    else:
        check("/predict-mri without model -> JSON 503", s == 503 and "error" in j, s)

    s, j, _ = post_file("/predict-mri", b"")
    check("empty file -> 400", s == 400 and "error" in j, s)
    s, j, _ = post_file("/predict-mri", b"this is not an image at all", "image/png")
    check("undecodable -> 400", s == 400 and "error" in j, s)
    s, j, _ = post_file("/predict-mri", synthetic_png(), "text/plain", "x.txt")
    check("wrong content type -> 415", s == 415 and "error" in j, s)
    s, j, _ = post_file("/predict-mri", b"\x89PNG" + b"0" * (11 * 1024 * 1024))
    check("over 10 MB -> 413", s == 413 and "error" in j, s)
    s, j, _ = post_file("/predict-mri", synthetic_png(16, 16))
    check("tiny 16x16 image -> 400", s == 400 and "error" in j, s)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
