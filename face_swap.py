"""
Face swap via insightface inswapper_128.
Uses the model already downloaded for roop at /home/work/facefusion-local/models/inswapper_128.onnx.
"""
import os
from io import BytesIO
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

INSWAPPER  = "/home/work/facefusion-local/models/inswapper_128.onnx"
ANTELOPE_DIR = "/home/work/ComfyUI/models/insightface"   # parent of /models/antelopev2

_app = None
_swapper = None


def _ensure():
    global _app, _swapper
    if _app is None:
        _app = FaceAnalysis(name="antelopev2", root=ANTELOPE_DIR,
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        _app.prepare(ctx_id=0, det_size=(640, 640))
    if _swapper is None:
        _swapper = get_model(INSWAPPER, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


def swap(source_bytes: bytes, target_bytes: bytes) -> bytes:
    """Take source face → paste onto every face in target. Returns PNG bytes."""
    _ensure()
    src = cv2.imdecode(np.frombuffer(source_bytes, np.uint8), cv2.IMREAD_COLOR)
    tgt = cv2.imdecode(np.frombuffer(target_bytes, np.uint8), cv2.IMREAD_COLOR)
    if src is None or tgt is None:
        raise ValueError("Could not decode one of the images")

    src_faces = _app.get(src)
    if not src_faces:
        raise ValueError("No face found in source image")
    src_face = sorted(src_faces, key=lambda f: f.bbox[2]*f.bbox[3], reverse=True)[0]

    tgt_faces = _app.get(tgt)
    if not tgt_faces:
        raise ValueError("No face found in target image")

    out = tgt.copy()
    for f in tgt_faces:
        out = _swapper.get(out, f, src_face, paste_back=True)

    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()
