import cv2
import time
import threading
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn
import webview

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = YOLO("yolov8n.pt")
cameras = {}
alerts = []

class CameraStream:
    def __init__(self, source):
        src = int(source) if str(source).isdigit() else source
        self.cap = cv2.VideoCapture(src)
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                else:
                    time.sleep(0.5)
            time.sleep(0.02)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.cap.release()

HTML_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8"><title>CCTV AI Sentinel</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: sans-serif; }
    body { background: #0f1117; color: #fff; display: flex; height: 100vh; }
    .side { width: 300px; background: #181b24; padding: 16px; display: flex; flex-direction: column; gap: 10px; border-right: 1px solid #272c3d; }
    .main { flex: 1; display: flex; flex-direction: column; padding: 16px; }
    .alerts { width: 260px; background: #181b24; padding: 16px; border-left: 1px solid #272c3d; overflow-y: auto; }
    input, button { padding: 8px; border-radius: 4px; border: 1px solid #374151; background: #0f1117; color: #fff; }
    button { background: #2563eb; cursor: pointer; border: none; font-weight: bold; }
    .danger { background: #dc2626; }
    .video-wrap { position: relative; flex: 1; display: flex; align-items: center; justify-content: center; background: #000; }
    #feed { max-width: 100%; max-height: 100%; }
    #zone-canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: crosshair; }
    .alert-card { background: rgba(220,38,38,0.2); border-left: 4px solid #dc2626; padding: 8px; margin-bottom: 8px; font-size: 0.85rem; }
  </style>
</head>
<body>
  <div class="side">
    <h3>Add RTSP Stream</h3>
    <input id="cname" placeholder="Camera Name" />
    <input id="curl" placeholder="RTSP URL (or 0 for webcam)" />
    <button onclick="addCam()">+ Add Camera</button>
    <h3 style="margin-top:20px;">Cameras</h3>
    <div id="cam-list" style="display:flex; flex-direction:column; gap:6px;"></div>
  </div>
  <div class="main">
    <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
      <h3 id="cam-title">Monitoring Feed</h3>
      <div>
        <button id="draw-btn" onclick="toggleDraw()">Draw Restricted Zone</button>
        <button onclick="clearZone()">Clear Zone</button>
        <button class="danger" onclick="deleteCam()">Remove Camera</button>
      </div>
    </div>
    <div class="video-wrap">
      <img id="feed" src="/video_feed/cam_main" />
      <canvas id="zone-canvas"></canvas>
    </div>
  </div>
  <div class="alerts">
    <h3>Intrusion Alerts</h3>
    <div id="alert-feed" style="margin-top:10px;"></div>
  </div>
  <script>
    let activeCam = "cam_main", isDrawing = false, pts = [];
    const canvas = document.getElementById("zone-canvas"), ctx = canvas.getContext("2d"), feed = document.getElementById("feed");
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    function beep() {
      const osc = audioCtx.createOscillator(), gain = audioCtx.createGain();
      osc.connect(gain); gain.connect(audioCtx.destination);
      osc.frequency.value = 880; gain.gain.value = 0.2; osc.start(); osc.stop(audioCtx.currentTime + 0.2);
    }
    function sync() { canvas.width = feed.clientWidth; canvas.height = feed.clientHeight; canvas.style.top = feed.offsetTop + "px"; canvas.style.left = feed.offsetLeft + "px"; }
    feed.onload = sync; window.onresize = sync;

    async function loadCams() {
      const res = await fetch("/api/cameras");
      const list = await res.json();
      const el = document.getElementById("cam-list");
      el.innerHTML = "";
      list.forEach(c => {
        const d = document.createElement("div");
        d.style.cssText = "padding:8px; background:#202534; cursor:pointer; border-radius:4px;";
        d.innerHTML = `<strong>${c.name}</strong><br><small style="color:#aaa">${c.url}</small>`;
        d.onclick = () => { activeCam = c.id; feed.src = '/video_feed/' + c.id; document.getElementById('cam-title').innerText = c.name; pts = []; ctx.clearRect(0,0,canvas.width,canvas.height); loadCams(); };
        el.appendChild(d);
      });
    }
    async function addCam() {
      const name = document.getElementById("cname").value || "Camera";
      const url = document.getElementById("curl").value || "0";
      await fetch("/api/cameras", { method: "POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({name, url}) });
      document.getElementById("cname").value = ""; document.getElementById("curl").value = "";
      loadCams();
    }
    async function deleteCam() {
      if(!confirm("Remove camera?")) return;
      await fetch('/api/cameras/' + activeCam, {method:"DELETE"});
      location.reload();
    }
    function toggleDraw() {
      isDrawing = !isDrawing;
      const b = document.getElementById("draw-btn");
      if(isDrawing) { b.innerText = "Save Zone"; b.style.background = "#10b981"; pts = []; }
      else { b.innerText = "Draw Restricted Zone"; b.style.background = "#2563eb"; saveZone(); }
    }
    canvas.onclick = (e) => {
      if(!isDrawing) return;
      const rect = canvas.getBoundingClientRect();
      pts.push([(e.clientX - rect.left)/canvas.width, (e.clientY - rect.top)/canvas.height]);
      ctx.clearRect(0,0,canvas.width,canvas.height);
      ctx.strokeStyle = "#eab308"; ctx.lineWidth = 2; ctx.beginPath();
      pts.forEach((p, i) => { const x = p[0]*canvas.width, y = p[1]*canvas.height; if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
      ctx.stroke();
    };
    async function saveZone() {
      await fetch('/api/cameras/' + activeCam + '/zone', { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({points: pts}) });
      ctx.clearRect(0,0,canvas.width,canvas.height);
    }
    async function clearZone() {
      pts = []; ctx.clearRect(0,0,canvas.width,canvas.height);
      await fetch('/api/cameras/' + activeCam + '/zone', { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({points: []}) });
    }
    let lastLen = 0;
    setInterval(async () => {
      const res = await fetch("/api/alerts");
      const al = await res.json();
      const b = document.getElementById("alert-feed");
      b.innerHTML = "";
      al.slice().reverse().forEach(a => {
        b.innerHTML += `<div class="alert-card"><strong>${a.cam} • ${a.time}</strong><br>${a.text}</div>`;
      });
      if(al.length > lastLen) { beep(); lastLen = al.length; }
    }, 1500);
    loadCams();
  </script>
</body>
</html>
"""

@app.get("/")
def home():
    return HTMLResponse(HTML_DASHBOARD)

@app.post("/api/cameras")
async def add_cam_api(req: Request):
    data = await req.json()
    cid = f"cam_{int(time.time())}"
    cameras[cid] = {
        "id": cid,
        "name": data.get("name", "Camera"),
        "url": data.get("url", 0),
        "zone": [],
        "stream": CameraStream(data.get("url", 0))
    }
    return {"status": "ok", "id": cid}

@app.delete("/api/cameras/{cid}")
def del_cam_api(cid: str):
    if cid in cameras:
        cameras[cid]["stream"].release()
        del cameras[cid]
    return {"status": "ok"}

@app.get("/api/cameras")
def list_cams_api():
    return [{"id": k, "name": v["name"], "url": v["url"]} for k, v in cameras.items()]

@app.post("/api/cameras/{cid}/zone")
async def set_zone_api(cid: str, req: Request):
    if cid in cameras:
        data = await req.json()
        cameras[cid]["zone"] = data.get("points", [])
    return {"status": "ok"}

@app.get("/api/alerts")
def alerts_api():
    return alerts[-10:]

def generate_frames(cid):
    global alerts
    while cid in cameras:
        frame = cameras[cid]["stream"].read()
        if frame is None:
            time.sleep(0.05)
            continue
        h, w = frame.shape[:2]
        if w > 1024:
            scale = 1024 / w
            frame = cv2.resize(frame, (1024, int(h * scale)))
            h, w = frame.shape[:2]

        zone = cameras[cid]["zone"]
        pts_norm = []
        if len(zone) >= 3:
            pts_norm = np.array([[int(p[0] * w), int(p[1] * h)] for p in zone], np.int32)
            cv2.polylines(frame, [pts_norm], True, (0, 255, 255), 2)

        results = model.predict(frame, classes=[0], conf=0.45, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bottom_center = (int((x1 + x2) / 2), y2)
            in_zone = False
            if len(pts_norm) >= 3:
                in_zone = (cv2.pointPolygonTest(pts_norm, bottom_center, False) >= 0)

            if in_zone:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                if not alerts or (time.time() - alerts[-1].get("_ts", 0)) > 2:
                    alerts.append({
                        "cam": cameras[cid]["name"],
                        "time": time.strftime("%H:%M:%S"),
                        "text": "Zone Intrusion Detected!",
                        "_ts": time.time()
                    })
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/video_feed/{cid}")
def feed_api(cid: str):
    if cid not in cameras:
        return "Not found", 404
    return StreamingResponse(generate_frames(cid), media_type="multipart/x-mixed-replace; boundary=frame")

def start_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

if __name__ == "__main__":
    cameras["cam_main"] = {
        "id": "cam_main",
        "name": "Default Camera",
        "url": 0,
        "zone": [],
        "stream": CameraStream(0)
    }
    t = threading.Thread(target=start_server, daemon=True)
    t.start()
    time.sleep(1.5)
    webview.create_window("CCTV AI Sentinel Admin", "http://127.0.0.1:8000", width=1280, height=800)
    webview.start()

