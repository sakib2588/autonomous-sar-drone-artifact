#!/usr/bin/env python3
"""Headless raw camera preview, no model. Plain video transmission over the LAN.

Runs ON the Raspberry Pi. Copy of 15_pi_live.py must sit next to it -- reuses
its open_camera() only, for the same MJPG/buffersize settings as the field
pipeline, so what you see matches what the recorder would capture.

    python3 21_pi_raw_stream.py

Then on another device on the same network, open:  http://<pi-ip>:8080/

This does NOT run the model and does NOT record. It exists to check the
camera/framing/focus after a flight without spending inference time or
writing an mp4. Stop it before running 15_pi_live.py -- two processes cannot
share one camera.

NETWORK EXPOSURE: unauthenticated. Fine on the field hotspot, not on a public
network.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load_open_camera():
    path = HERE / "15_pi_live.py"
    if not path.exists():
        raise SystemExit(
            f"[FAIL] {path} not found.\n"
            "       This reuses the field pipeline's open_camera() so the capture settings\n"
            "       (MJPG, buffersize) match. Copy 15_pi_live.py next to this script and re-run."
        )
    spec = importlib.util.spec_from_file_location("pi_live", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "open_camera"):
        raise SystemExit("[FAIL] 15_pi_live.py has no open_camera() -- version mismatch, re-copy it from the repo.")
    return mod.open_camera


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class FrameSlot:
    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg = None
        self._seq = 0
        self.stats = {}

    def publish(self, jpeg: bytes, stats: dict) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self.stats = stats
            self._cond.notify_all()

    def wait_next(self, last_seq: int, timeout: float = 5.0):
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._jpeg, self._seq


SLOT = FrameSlot()
STOP = threading.Event()

INDEX = b"""<!doctype html><meta charset=utf-8><title>SAR Pi raw preview</title>
<style>
 body{background:#111;color:#ddd;font:14px system-ui,sans-serif;margin:0;padding:16px}
 h1{font-size:16px;margin:0 0 12px;font-weight:600}
 img{max-width:100%;border:1px solid #333;background:#000;display:block}
 #s{margin-top:10px;font-family:ui-monospace,monospace;font-size:13px;line-height:1.6;color:#9fe}
 .warn{color:#fc6}
</style>
<h1>SAR Pi raw camera preview (no model) <span class=warn>(unauthenticated LAN stream)</span></h1>
<img src="/stream.mjpg" alt="live">
<div id=s>connecting...</div>
<script>
async function poll(){
 try{const r=await fetch('/stats.json',{cache:'no-store'});const d=await r.json();
  document.getElementById('s').textContent = `capture ${d.capture_fps} FPS | clients ${d.clients}`;
 }catch(e){document.getElementById('s').textContent='stats unavailable';}
 setTimeout(poll,1000);}
poll();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    clients = 0
    _lock = threading.Lock()

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", INDEX)
        elif self.path == "/stats.json":
            body = json.dumps(dict(SLOT.stats, clients=Handler.clients)).encode()
            self._send(200, "application/json", body)
        elif self.path == "/snapshot.jpg":
            jpeg, _ = SLOT.wait_next(-1, timeout=5.0)
            if jpeg is None:
                self._send(503, "text/plain", b"no frame yet")
            else:
                self._send(200, "image/jpeg", jpeg)
        elif self.path == "/stream.mjpg":
            self._stream()
        else:
            self._send(404, "text/plain", b"not found")

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream(self) -> None:
        with Handler._lock:
            Handler.clients += 1
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        seq = -1
        try:
            while not STOP.is_set():
                jpeg, seq = SLOT.wait_next(seq)
                if jpeg is None:
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with Handler._lock:
                Handler.clients -= 1


def capture_loop(cli, open_camera) -> None:
    import cv2

    cap = open_camera(cli.camera, cli.width, cli.height, cli.fps, cli.buffersize)
    if not cap.isOpened():
        print(f"[FAIL] could not open {cli.camera}", flush=True)
        STOP.set()
        return
    print(f"[camera] {cli.camera} buffersize={cap.get(cv2.CAP_PROP_BUFFERSIZE)}", flush=True)

    enc = [int(cv2.IMWRITE_JPEG_QUALITY), cli.quality]
    cap_times: list = []

    while not STOP.is_set():
        ok, frame = cap.read()
        if not ok:
            print("[camera] read failed, reopening", flush=True)
            cap.release()
            cap = open_camera(cli.camera, cli.width, cli.height, cli.fps, cli.buffersize)
            continue

        now = time.time()
        cap_times.append(now)
        if len(cap_times) > 30:
            cap_times.pop(0)

        shown = frame if cli.scale == 1.0 else cv2.resize(frame, None, fx=cli.scale, fy=cli.scale,
                                                          interpolation=cv2.INTER_AREA)
        okj, buf = cv2.imencode(".jpg", shown, enc)
        if not okj:
            continue

        def rate(ts):
            return round((len(ts) - 1) / (ts[-1] - ts[0]), 2) if len(ts) > 1 and ts[-1] > ts[0] else 0.0

        SLOT.publish(buf.tobytes(), {"capture_fps": rate(cap_times)})

    cap.release()
    print("[capture] stopped", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="/dev/video0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--buffersize", type=int, default=2)
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--quality", type=int, default=70)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    cli = ap.parse_args()

    open_camera = load_open_camera()

    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())

    worker = threading.Thread(target=capture_loop, args=(cli, open_camera), daemon=True)
    worker.start()

    server = ThreadingHTTPServer((cli.host, cli.port), Handler)
    server.daemon_threads = True
    url = f"http://{lan_ip()}:{cli.port}/"
    print(f"\n[serve] open this on your laptop:  {url}", flush=True)
    print("[serve] unauthenticated -- anyone on this network can watch. Ctrl-C to stop.\n", flush=True)

    t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True)
    t.start()
    try:
        while not STOP.is_set():
            STOP.wait(0.5)
    except KeyboardInterrupt:
        STOP.set()
    finally:
        STOP.set()
        server.shutdown()
        worker.join(timeout=3.0)
        print("[stop] clean shutdown", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
