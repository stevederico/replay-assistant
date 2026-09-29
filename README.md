# yolo-server

A small, stateless object-detection service: a zero-crate Rust binary that runs
**YOLO26n-seg** on the CPU. It is a drop-in replacement for a Python/Flask/ultralytics
detection service.

- **Zero crates.** `[dependencies]` is empty (`cargo tree` prints only `yolo-server`).
  Inference is the system `libonnxruntime` over its C API (`#[link(name = "onnxruntime")]`).
  Frames come from the `ffmpeg` command line as piped raw RGB. HTTP, JSON,
  letterbox, box decode, NMS, mask decode and contour tracing are hand written.
- **Stateless.** Nothing is stored between requests. The upload lives in a temp
  file while it is processed and is deleted afterwards. The only in-memory state
  is a bounded map of progress for `GET /progress/{id}`.
- **Sleeps when idle.** It starts in about 0.3 s (model load plus a warm-up run),
  so Railway serverless can stop it between uses.

## API

### `GET /health`

`{"status":"ok","model":"yolo26n-seg","onnxruntime":"1.30.0","inflight":0}`

### `POST /detect`

Body: the raw bytes of a video or an image (`Content-Length` required, no
multipart, no chunked upload). Query parameters, all optional:

| Parameter | Default | Meaning |
|---|---|---|
| `every_n` | `1` (`YOLO_EVERY_N`) | Keep every N-th frame, 1 to 10000 |
| `conf` | `0.25` | Score threshold, 0.001 to 1 (a score must be strictly greater) |
| `iou` | `0.7` | NMS IoU threshold, above 0 and at most 1 |
| `classes` | all | Comma-separated COCO ids, e.g. `0,32` (person, sports ball) |
| `max_frames` | `3000` (`YOLO_MAX_FRAMES`) | Cap on sampled frames for this request |
| `masks` | `1` | `0` skips polygons (no mask work at all) |
| `simplify` | `1` | Polygon tolerance in model pixels (Douglas-Peucker), 0 to 20; `0` keeps every contour point |
| `detection_id` | none | Lets a second request poll `GET /progress/{id}` |

Response:

```json
{
  "model": "yolo26n-seg", "width": 1280, "height": 720, "fps": 29.97,
  "totalFrames": 900, "everyN": 3, "conf": 0.25, "iou": 0.7,
  "frames": [
    {"frame": 0, "detections": [
      {"class": 0, "name": "person", "score": 0.6651,
       "box": [1048.08, 361.54, 1112.74, 481.43],
       "polygon": [[1050.1, 400.2], [1049.5, 410.0]]}
    ]}
  ],
  "sampled": 300,
  "timing": {"totalMs": 18999.4, "letterboxMs": 1205.1, "inferMs": 16674.6,
             "postMs": 692.8, "msPerFrame": 63.33, "inferMsPerFrame": 55.58}
}
```

- `frame` is the source frame number (`k * every_n`). Boxes and polygons are in
  source pixels, after any phone rotation is applied (`width`/`height` are the
  decoded size).
- Boxes are `[x1, y1, x2, y2]`. Polygons are the outline of the largest blob of
  each mask; a detection whose mask is empty keeps its box and gets `[]`.
- Errors are `{"error": "plain-language message"}`.

| Status | When |
|---|---|
| 400 | Bad query parameter (names it), empty body, upload ended early |
| 411 | No `Content-Length`, or a chunked upload |
| 413 | Body over `YOLO_MAX_BODY_MB` |
| 415 | `multipart/*` body |
| 422 | Not a decodable video or image, too many frames after sampling, result over the size cap |
| 503 + `Retry-After: 5` | Already at `YOLO_MAX_INFLIGHT` requests |
| 504 | Over `YOLO_REQUEST_TIMEOUT_SECS`; the job is cancelled |
| 500 | Server-side failure (details go to the log, not the response) |

### `GET /progress/{id}`

`{"frame": 210, "total": 900, "status": "processing"}`. Status is `processing`,
`completed` or `failed`. The last 64 ids are kept. 404 for an unknown id.

### Old routes

`/video_feed`, `/infer`, `/upload_video`, `/analyze_frames`, `/analyze_release` and
`/detection_progress/*` answer **501** with a message pointing at `/detect`. There
is no MJPEG live feed (a CPU cannot stream it) and no pose model.

## Configuration

| Variable | Default | |
|---|---|---|
| `PORT` | `5001` | Listen port |
| `YOLO_MODEL` | `models/yolo26n-seg.onnx` | Model path |
| `YOLO_THREADS` | `min(cores, 4)` | onnxruntime threads (4 beat 8 on a 4-core/8-thread CPU) |
| `YOLO_MAX_BODY_MB` | `512` | Largest upload |
| `YOLO_MAX_FRAMES` | `3000` | Most sampled frames per request |
| `YOLO_MAX_INFLIGHT` | `3` | Requests uploading, queued or running at once (one runs, the rest queue) |
| `YOLO_REQUEST_TIMEOUT_SECS` | `600` | Whole request, queue wait included |
| `YOLO_UPLOAD_TIMEOUT_SECS` | `300` | Time allowed to receive the upload |
| `YOLO_MAX_RESPONSE_MB` | `256` | Largest response built in memory |
| `YOLO_TMP_DIR` | `$TMPDIR/yolo-server` | Where uploads wait; stale files are removed at boot |
| `FFMPEG`, `FFPROBE` | `ffmpeg`, `ffprobe` | Executables |

One inference thread runs at a time. Memory is bounded: uploads stream to disk in
64 KB chunks, one frame buffer is reused, the response is capped, and at most 32
connections are open.

## The model

`models/yolo26n-seg.onnx`: YOLO26n-seg, exported once with ultralytics 8.4.165
(`scripts/export_model.py`), opset 12, static `1x3x640x640` input, outputs
`(1,116,8400)` and `(1,32,160,160)`. The export is not byte-reproducible, so the
hash below is of the committed file.

| | |
|---|---|
| Source weights | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt |
| `yolo26n-seg.pt` sha256 | `361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f` |
| `yolo26n-seg.onnx` sha256 | `f0dbe8b1007423d90ad67a506023be922ff9fe3b172deef3365d9734e2381d00` |

### License: AGPL-3.0

This repository is licensed **AGPL-3.0** (see `LICENSE`). The bundled model is
Ultralytics YOLO26n-seg, whose code and pretrained weights are AGPL-3.0 (or a paid
Ultralytics Enterprise license). Serving the weights from a network service is the
case AGPL section 13 is about, so if you run a modified copy as a service, offer its
source to your users. (Not legal advice.)

To use this with a different license, swap the model. The service only needs an ONNX
detector with the same output layout (or a small change in `src/yolo.rs`). Some
permissively licensed options (check each repo's LICENSE and the
weights' terms, COCO-trained weights also carry dataset terms, before adopting one):

| Model | License | Notes |
|---|---|---|
| YOLOX (Megvii) | Apache-2.0 | Boxes only, no masks |
| RT-DETR (PaddleDetection) | Apache-2.0 | Boxes only |
| D-FINE, DEIM | Apache-2.0 | Boxes only, recent |
| RTMDet-Ins (MMDetection) | Apache-2.0 | Instance segmentation |
| Mask R-CNN (torchvision, Detectron2) | BSD-3-Clause, Apache-2.0 | Instance segmentation, heavier and slower on CPU |
| RF-DETR (Roboflow) | Apache-2.0 for most sizes | Has a segmentation variant; check the checkpoint |
| SAM / SAM 2 / MobileSAM | Apache-2.0 | Promptable segmentation, not a class detector |
| YOLOv5, v8, v10, 11, 12, 26 | AGPL-3.0 | Same family |
| YOLOv7, YOLOv9 | GPL-3.0 | Copyleft |

If you only need boxes, a boxes-only Apache-2.0 detector plus `masks=0` is a real option.

## How closely it matches ultralytics

`tests/fixtures/ultralytics_expected.json` (not shipped, see below) holds what
ultralytics itself detects when it runs the same ONNX file on the same
ffmpeg-decoded pixels (`scripts/export_model.py --reference-only`). `tests/model.rs` requires, per
detection, boxes within 1.5 px, scores within 0.02 and polygon areas within 3 %
(8 % for scores under 0.3, whose masks are marginal), and the same detection
count. Measured on the fixtures: boxes within 0.4 px, scores within 0.002,
polygon area within 5.4 % (the worst is a marginal mask).

Two behaviours worth knowing:

- Ultralytics runs `.pt` models with a rectangular letterbox (640x384 for 16:9)
  but ONNX models with a square 640x640 one. This service is the ONNX case.
- Ultralytics 8.4 upsamples masks, thresholds, then crops to the box; older
  versions cropped first, which inflated mask borders. This service follows 8.4.

## Speed

Measured on the dev OptiPlex (Intel i7-7700, 4 cores / 8 threads, no GPU),
release build, the 30 s 1280x720 clip in `../backend/yolo/assets/video.mov`,
end to end through `POST /detect` (ffmpeg decode, letterbox, model, decode, JSON):

| Run | ms per frame | Notes |
|---|---|---|
| 4 threads, masks on, `every_n=3`, `conf=0.1` | **63** | 300 frames in 19.0 s; model 56, letterbox 4, decode+masks 2 |
| 4 threads, masks off, every frame | 63 | 900 frames in 56.6 s |
| 8 threads, masks on, `every_n=3` | 91 | Hyperthreads and ffmpeg fight for cores |

The rows above were measured with YOLO11n-seg on a quiet machine, before the swap
to YOLO26n-seg. Back to back on the same busy machine (load average 6 to 9),
`every_n=3`, `conf=0.1`: YOLO26n 101 and 106 ms per frame, YOLO11n 107 and 110
ms, so YOLO26n is about 4 % faster. Absolute times on a quiet machine were not
re-measured for YOLO26n.

Cold start is about 0.3 s to a healthy `/health` (0.6 s on the very first run
with a cold disk cache). Idle memory is 113 MB; peak was 178 MB while processing
the 13 MB clip. The old service defaulted to `yolo11x-seg`, which is far larger
and far more accurate on small objects such as balls: expect the nano model to
miss more of them.

## Tests

```bash
scripts/install-onnxruntime.sh     # once: pinned tarball to ~/.local/opt, sha256 checked, no sudo
scripts/test.sh                    # sets LIBRARY_PATH and LD_LIBRARY_PATH, runs cargo test --locked
cargo test --locked --no-default-features   # pure-logic tests only, no libonnxruntime needed
```

- Unit tests (in `src/`): letterbox, IoU, NMS, box decode, mask decode, contour
  tracing, polygon simplification, JSON writer and parser, request parsing, query
  validation, ffprobe parsing, and the HTTP server over a real socket with a fake
  detector (limits, timeouts, 503 when busy, 501 old routes, cleanup).
- `tests/model.rs`: real model on the fixture images, the ultralytics comparison,
  frame sampling and rotated clips, limits, garbage input, progress and
  cancellation, and an HTTP round trip. Without the `onnxruntime` feature it
  prints `SKIP:` and passes; without ffmpeg each test prints `SKIP:`.
- The comparison fixtures are not shipped (the original two frames came from
  broadcast footage that cannot be redistributed). Put two 1280x720 frames named
  `frame14.jpg` and `frame20.jpg` in `tests/fixtures/`, run
  `scripts/export_model.py --reference-only frame14.jpg frame20.jpg`, and the four
  fixture tests run. Without them they print `SKIP:` and pass. The two "Measured on
  the fixtures" numbers above were taken on the original frames.
  The tests that build a synthetic clip with ffmpeg need no fixtures.

## Build and run

```bash
. scripts/ort-env.sh
cargo build --release --locked
PORT=5001 ./target/release/yolo-server
curl -X POST --data-binary @clip.mp4 'localhost:5001/detect?every_n=3&conf=0.1&classes=0,32'
```

`Dockerfile` and `railway.json` are for Railway (`sleepApplication: true`); the
image downloads the pinned onnxruntime release and verifies its sha256, installs
ffmpeg, runs as a non-root user and has a `HEALTHCHECK` (`yolo-server healthcheck`,
no curl needed).

To deploy to Railway: create a service whose root directory is this
directory, set `PORT=5001`, leave Serverless on, then set
`YOLO_SERVICE_URL=http://<service>.railway.internal:5001` on the calling service.
