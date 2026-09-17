#!/usr/bin/env python3
"""Headless live preview: watch the Pi's detections in a browser, no monitor.

Runs ON the Raspberry Pi. Copies of 15_pi_live.py, decode.py and sysmon.py must
sit next to it -- the Pi has no checkout of the repo, same convention as
13_pi_bench.py.

    python3 20_pi_stream_preview.py --model models/best_full_integer_quant.tflite

Then on the laptop, open:  http://<pi-ip>:8080/

Why this exists: 16_pi_live_preview.py pipes raw BGR into ffplay on display :0,
which needs a monitor physically attached to the Pi. The Pi is headless and
SSH-only (nine services disabled, see the pi4-field-deployment note), so that
tool cannot be used to watch a flight rehearsal from a laptop. cv2.imshow is
equally unavailable -- the deployed OpenCV is the headless build with no GUI
backend. An MJPEG HTTP stream needs neither: any browser on the LAN renders it,
and it survives the client disconnecting and reconnecting.

Raw BGR over the network was rejected on bandwidth: 1280x720x3 at 24 FPS is
~66 MB/s, far past what the Pi's WiFi carries. JPEG-encoding each frame first
brings a 720p preview to roughly 1-3 MB/s depending on --quality.

PARITY IS THE POINT. This imports preprocess, draw_detections and held_boxes
from 15_pi_live.py rather than reimplementing them, so what you watch is what
the field pipeline actually does -- including the held-box behaviour between
inferences. A preview that quietly differs from the pipeline is worse than no
preview, because it builds confidence in something that was never tested.

This does NOT record. 15_pi_live.py is the recorder; running both at once means
two processes fighting over one camera. Use this to watch and aim, then stop it
and run the recorder.

NETWORK EXPOSURE: the stream is unauthenticated. Anyone who can reach the Pi on
port 8080 can watch the camera. That is fine on a home LAN and not fine on a
shared or public network. Bind to a specific interface with --host, or leave it
stopped when not in use.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from decode import decode_yolo_output  # noqa: E402


def load_pipeline_module():
    """Import 15_pi_live.py for its preprocessing and drawing.

    Loaded by path because the module name starts with a digit and cannot be a
    normal import. Failing loudly here is deliberate: the entire value of this
    tool is showing the same thing the field pipeline shows.
    """
    path = HERE / "15_pi_live.py"
    if not path.exists():
        raise SystemExit(
            f"[FAIL] {path} not found.\n"
            "       This preview reuses the field pipeline's preprocessing and drawing so the\n"
            "       two cannot drift. Copy 15_pi_live.py next to this script and re-run."
        )
    spec = importlib.util.spec_from_file_location("pi_live", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("preprocess", "draw_detections", "held_boxes", "open_camera", "infer_tiled"):
        if not hasattr(mod, fn):
            raise SystemExit(f"[FAIL] 15_pi_live.py has no {fn}() -- version mismatch, re-copy it from the repo.")
    return mod


def vcgencmd(*args: str) -> str:
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return ""


def soc_temp_c():
    raw = vcgencmd("measure_temp")
    if "=" not in raw:
        return None
    try:
        return float(raw.split("=")[1].split("'")[0])
    except (IndexError, ValueError):
        return None


def lan_ip() -> str:
    """Best-effort own address, for printing a URL the user can actually click."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packet is sent; this only picks a route
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class FrameSlot:
    """Latest-frame-wins hand-off from the capture thread to HTTP clients.

    Deliberately NOT a queue. A slow or stalled browser must never make the
    capture loop fall behind -- it just misses frames, which is the correct
    failure mode for a live preview.
    """

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

INDEX = b"""<!doctype html><meta charset=utf-8><title>SAR Pi live preview</title>
<style>
 body{background:#111;color:#ddd;font:14px system-ui,sans-serif;margin:0;padding:16px}
 h1{font-size:16px;margin:0 0 12px;font-weight:600}
 img{max-width:100%;border:1px solid #333;background:#000;display:block}
 #s{margin-top:10px;font-family:ui-monospace,monospace;font-size:13px;line-height:1.6;color:#9fe}
 .warn{color:#fc6}
</style>
<h1>SAR Pi live preview <span class=warn>(unauthenticated LAN stream)</span></h1>
<img src="/stream.mjpg" alt="live">
<div id=s>connecting...</div>
<script>
async function poll(){
 try{const r=await fetch('/stats.json',{cache:'no-store'});const d=await r.json();
  document.getElementById('s').textContent =
   `capture ${d.capture_fps} FPS | inference ${d.infer_fps} FPS (${d.infer_ms} ms) | `+
   `detections ${d.n_det} | held ${d.n_held} | SoC ${d.temp_c} C | throttle ${d.throttled} | clients ${d.clients}`;
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
        pass  # the access log would drown the operational prints

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
            pass  # browser tab closed; entirely normal
        finally:
            with Handler._lock:
                Handler.clients -= 1


def capture_loop(cli, pipeline) -> None:
    import cv2

    from tflite_runtime.interpreter import Interpreter

    interp = Interpreter(model_path=cli.model, num_threads=cli.threads)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    model_w = int(inp["shape"][1])
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    print(f"[model] {cli.model}  input {model_w}x{model_w} {inp['dtype'].__name__}", flush=True)

    cap = pipeline.open_camera(cli.camera, cli.width, cli.height, cli.fps, cli.buffersize)
    if not cap.isOpened():
        print(f"[FAIL] could not open {cli.camera}", flush=True)
        STOP.set()
        return
    print(f"[camera] {cli.camera} buffersize={cap.get(cv2.CAP_PROP_BUFFERSIZE)}", flush=True)

    hold = cli.box_hold_frames if cli.box_hold_frames is not None else cli.infer_every_n
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), cli.quality]
    frame_idx = 0
    last_dets: list = []
    last_infer_frame = -(10 ** 9)
    infer_ms = 0.0
    cap_times: list = []
    inf_times: list = []
    t_temp = 0.0
    temp = soc_temp_c()
    throttled = vcgencmd("get_throttled")

    while not STOP.is_set():
        ok, frame = cap.read()
        if not ok:
            print("[camera] read failed, reopening", flush=True)
            cap.release()
            cap = pipeline.open_camera(cli.camera, cli.width, cli.height, cli.fps, cli.buffersize)
            continue

        now = time.time()
        cap_times.append(now)
        if len(cap_times) > 30:
            cap_times.pop(0)

        fresh = frame_idx % cli.infer_every_n == 0
        dets: list = []
        if fresh:
            t0 = time.perf_counter()
            if cli.tile:
                dets = pipeline.infer_tiled(interp, inp, out, frame, model_w, in_scale, in_zero,
                                            out_scale, out_zero, cli.conf_thres, cli.iou_thres,
                                            cli.tile_overlap)
            else:
                x = pipeline.preprocess(frame, model_w, in_scale, in_zero, inp["dtype"])
                interp.set_tensor(inp["index"], x)
                interp.invoke()
                y = interp.get_tensor(out["index"])
                dets = decode_yolo_output(y, out_scale, out_zero, conf_thres=cli.conf_thres,
                                          iou_thres=cli.iou_thres, img_size=model_w)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            last_dets = dets
            last_infer_frame = frame_idx
            inf_times.append(now)
            if len(inf_times) > 15:
                inf_times.pop(0)

        held = pipeline.held_boxes(frame_idx, last_infer_frame, last_dets, hold)
        shown = frame if cli.scale == 1.0 else cv2.resize(frame, None, fx=cli.scale, fy=cli.scale,
                                                          interpolation=cv2.INTER_AREA)
        drawn = dets if fresh else held
        if cli.tile and cli.scale != 1.0:
            # Tiled boxes are full-frame pixels at CAPTURE resolution; the streamed
            # image is smaller, so they must follow it down or they land off-target.
            drawn = [dict(d, box_xyxy=[v * cli.scale for v in d["box_xyxy"]]) for d in drawn]
        pipeline.draw_detections(shown, drawn, model_w, fresh=fresh, frame_coords=cli.tile)

        okj, buf = cv2.imencode(".jpg", shown, enc)
        if not okj:
            continue

        if now - t_temp > 5.0:  # vcgencmd is a subprocess spawn; not every frame
            temp = soc_temp_c()
            throttled = vcgencmd("get_throttled")
            t_temp = now

        def rate(ts):
            return round((len(ts) - 1) / (ts[-1] - ts[0]), 2) if len(ts) > 1 and ts[-1] > ts[0] else 0.0

        SLOT.publish(buf.tobytes(), {
            "capture_fps": rate(cap_times),
            "infer_fps": rate(inf_times),
            "infer_ms": round(infer_ms, 1),
            "n_det": len(dets if fresh else held),
            "n_held": 0 if fresh else len(held),
            "temp_c": temp,
            "throttled": throttled,
        })
        frame_idx += 1

    cap.release()
    print("[capture] stopped", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to the .tflite artifact")
    ap.add_argument("--camera", default="/dev/video0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--buffersize", type=int, default=2,
                    help="V4L2 queue depth. 2 measured optimal on this Pi; 1 HALVES capture rate here")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--infer-every-n", type=int, default=5)
    ap.add_argument("--conf-thres", type=float, default=0.25)
    ap.add_argument("--iou-thres", type=float, default=0.45)
    ap.add_argument("--box-hold-frames", type=int, default=None)
    ap.add_argument("--tile", action="store_true",
                    help="tiled inference, matching 15_pi_live.py --tile and how the model was trained. "
                         "Measured on this Pi: 640 model 2223 ms over 6 tiles, 416 model 1190 ms over 8")
    ap.add_argument("--tile-overlap", type=float, default=0.2)
    ap.add_argument("--scale", type=float, default=0.75, help="downscale the STREAMED image; inference is unaffected")
    ap.add_argument("--quality", type=int, default=70, help="JPEG quality 1-100; lower it if WiFi struggles")
    ap.add_argument("--host", default="0.0.0.0", help="bind address. 0.0.0.0 exposes the stream to the whole LAN")
    ap.add_argument("--port", type=int, default=8080)
    cli = ap.parse_args()

    pipeline = load_pipeline_module()

    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())

    worker = threading.Thread(target=capture_loop, args=(cli, pipeline), daemon=True)
    worker.start()

    server = ThreadingHTTPServer((cli.host, cli.port), Handler)
    server.daemon_threads = True
    url = f"http://{lan_ip()}:{cli.port}/"
    print(f"\n[serve] open this on your laptop:  {url}", flush=True)
    print(f"[serve] single frame: {url}snapshot.jpg   stats: {url}stats.json", flush=True)
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
