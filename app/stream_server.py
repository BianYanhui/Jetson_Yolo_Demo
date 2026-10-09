"""Realtime YOLOv6 MJPEG stream server for LAN viewing."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template_string

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.detector import YOLOv6Detector  # noqa: E402

INDEX_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Jetson YOLOv6 Live</title>
  <style>
    :root {
      --bg: #0f1419;
      --panel: #1a2332;
      --text: #e7ecf3;
      --muted: #8b9bb4;
      --accent: #3dd68c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "SF Pro Text", "Segoe UI", sans-serif;
      background:
        radial-gradient(1200px 600px at 10% -10%, #1e3a5f 0%, transparent 55%),
        radial-gradient(900px 500px at 100% 0%, #163828 0%, transparent 50%),
        var(--bg);
      color: var(--text);
    }
    main {
      max-width: 1100px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: clamp(1.4rem, 2.5vw, 2rem);
      letter-spacing: 0.02em;
    }
    .meta {
      color: var(--muted);
      margin-bottom: 18px;
      font-size: 0.95rem;
    }
    .stage {
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 20px 50px rgba(0,0,0,0.35);
    }
    img {
      display: block;
      width: 100%;
      height: auto;
      background: #000;
      min-height: 240px;
    }
    .bar {
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      padding: 12px 16px;
      font-size: 0.9rem;
      color: var(--muted);
      border-top: 1px solid rgba(255,255,255,0.06);
    }
    .bar strong { color: var(--accent); font-weight: 600; }
  </style>
</head>
<body>
  <main>
    <h1>Jetson YOLOv6 Live</h1>
    <p class="meta">CUDA realtime detection · 720p MJPEG · refresh automatically</p>
    <div class="stage">
      <img src="/video_feed" alt="YOLOv6 stream" />
      <div class="bar">
        <div>Status: <strong id="status">connecting…</strong></div>
        <div>Infer FPS: <strong id="fps">-</strong></div>
        <div>Device: <strong id="device">-</strong></div>
      </div>
    </div>
  </main>
  <script>
    async function poll() {
      try {
        const r = await fetch('/api/stats');
        const j = await r.json();
        document.getElementById('status').textContent = j.running ? 'live' : 'stopped';
        document.getElementById('fps').textContent = j.fps.toFixed(1);
        document.getElementById('device').textContent = j.device;
      } catch (e) {
        document.getElementById('status').textContent = 'offline';
      }
    }
    poll();
    setInterval(poll, 1000);
  </script>
</body>
</html>
"""


class CameraWorker:
    def __init__(
        self,
        camera: int | str,
        detector: YOLOv6Detector,
        width: int = 1280,
        height: int = 720,
        jpeg_quality: int = 70,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.width = width
        self.height = height
        self.jpeg_quality = jpeg_quality
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self.frame_count = 0
        self.last_error = ""

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.camera, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self) -> None:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        while self._running:
            cap = None
            try:
                cap = self._open_capture()
                if not cap.isOpened():
                    self.last_error = f"cannot open camera {self.camera}"
                    time.sleep(1.0)
                    continue
                self.last_error = ""
                while self._running:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        self.last_error = "camera read failed"
                        break
                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = cv2.resize(frame, (self.width, self.height))
                    annotated = self.detector.infer(frame)
                    ok, buf = cv2.imencode(".jpg", annotated, encode_param)
                    if not ok:
                        continue
                    with self._lock:
                        self._jpeg = buf.tobytes()
                    self.frame_count += 1
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                time.sleep(0.5)
            finally:
                if cap is not None:
                    cap.release()
            time.sleep(0.2)


def build_app(worker: CameraWorker) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template_string(INDEX_HTML)

    @app.get("/video_feed")
    def video_feed():
        def generate():
            boundary = b"--frame"
            while True:
                jpeg = worker.get_jpeg()
                if jpeg is None:
                    time.sleep(0.05)
                    continue
                yield (
                    boundary
                    + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode()
                    + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
                time.sleep(0.02)

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    @app.get("/api/stats")
    def stats():
        return jsonify(
            {
                "running": worker._running,
                "fps": worker.detector.last_fps,
                "device": str(worker.detector.device),
                "frames": worker.frame_count,
                "error": worker.last_error,
            }
        )

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "frames": worker.frame_count})

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Jetson YOLOv6 realtime MJPEG server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--camera", default="0", help="camera index or /dev/videoX")
    p.add_argument("--weights", default=str(ROOT / "weights" / "yolov6n.pt"))
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--device", default="0")
    p.add_argument("--no-half", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    camera: int | str
    if str(args.camera).isdigit():
        camera = int(args.camera)
    else:
        camera = args.camera

    print(f"[boot] loading weights: {args.weights}", flush=True)
    detector = YOLOv6Detector(
        weights=args.weights,
        device=args.device,
        img_size=args.img_size,
        conf_thres=args.conf,
        iou_thres=args.iou,
        half=not args.no_half,
    )
    print(f"[boot] device={detector.device} half={detector.half}", flush=True)

    worker = CameraWorker(
        camera=camera,
        detector=detector,
        width=args.width,
        height=args.height,
        jpeg_quality=args.jpeg_quality,
    )
    worker.start()

    app = build_app(worker)

    def _shutdown(*_args):
        worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[boot] serving on http://{args.host}:{args.port}/", flush=True)
    # threaded=True so /video_feed does not block / and /api/stats
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    # Avoid OpenCV using too many CPU threads on Jetson
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
