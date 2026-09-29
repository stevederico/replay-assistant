"""One-time offline export of YOLO26n-seg to ONNX plus a reference dump.

This is the ONLY Python in yolo-server, and it never runs in the service or in
Docker. Run it in a throwaway venv, commit the outputs, delete the venv:

    python3 -m venv /tmp/yolo-export && /tmp/yolo-export/bin/pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu ultralytics onnx onnxslim onnxruntime
    cd /tmp/yolo-export-work            # any empty directory holding the fixtures
    /tmp/yolo-export/bin/python <repo>/scripts/export_model.py frame14.jpg frame20.jpg
    /tmp/yolo-export/bin/python <repo>/scripts/export_model.py --reference-only frame14.jpg frame20.jpg
    rm -rf /tmp/yolo-export

Outputs (relative to the working directory):
    yolo26n-seg.onnx            opset 12, static 1x3x640x640 input (skipped with --reference-only)
    ultralytics_expected.json   what ultralytics itself detects on each fixture
                                image, used by tests/model.rs to prove the Rust
                                decode matches

The reference runs the exported ONNX file through ultralytics (so it uses the
same static 640x640 letterbox as the service; the .pt path letterboxes to a
640x384 rectangle for 16:9 frames and gives slightly different numbers), on
pixels decoded by ffmpeg (the decoder the service uses; OpenCV's JPEG decoder
differs by a few gray levels). Polygons use the largest contour, which is what
the service returns.

Ultralytics YOLO26 weights are AGPL-3.0. See README.md.
"""
import json
import subprocess
import sys

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import ops

CONF = 0.25
IOU = 0.7
ONNX = "yolo26n-seg.onnx"


def export():
    model = YOLO("yolo26n-seg.pt")
    model.export(format="onnx", imgsz=640, opset=12, dynamic=False, simplify=True)


def ffmpeg_pixels(path):
    """Decode an image with ffmpeg exactly as the service does: rgb24 -> BGR array."""
    height, width = cv2.imread(path).shape[:2]
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True,
        capture_output=True,
    ).stdout
    rgb = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    return np.ascontiguousarray(rgb[..., ::-1])


def reference(paths):
    model = YOLO(ONNX, task="segment")
    expected = {"conf": CONF, "iou": IOU, "images": {}}
    for path in paths:
        result = model(ffmpeg_pixels(path), verbose=False, conf=CONF, iou=IOU)[0]
        segments = ops.masks2segments(result.masks.data, strategy="largest") if result.masks is not None else []
        detections = []
        for i, (box, cls, score) in enumerate(zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf)):
            polygon = ops.scale_coords(result.masks.data.shape[1:], segments[i].copy(), result.orig_shape)
            detections.append({
                "class": int(cls),
                "score": round(float(score), 4),
                "box": [round(v, 2) for v in box.tolist()],
                "polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in polygon],
            })
        expected["images"][path] = {
            "width": result.orig_shape[1],
            "height": result.orig_shape[0],
            "detections": detections,
        }
    with open("ultralytics_expected.json", "w") as f:
        json.dump(expected, f)


if __name__ == "__main__":
    args = sys.argv[1:]
    reference_only = "--reference-only" in args
    images = [a for a in args if not a.startswith("--")]
    if not reference_only:
        export()
    reference(images)
