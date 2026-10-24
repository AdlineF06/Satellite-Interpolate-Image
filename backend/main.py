import base64, os, time, numpy as np, cv2, torch
from datetime import datetime, timezone
from functools import lru_cache
from urllib.parse import urlencode
from urllib.request import Request as URLRequest, urlopen
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from synthetic import make_sequence
from model import RefineNet, make_input

app = FastAPI(title="SatInterp API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

REGIONS = [
    dict(id="bob", name="Bay of Bengal", lat=15, lng=88, event="Cyclone", satellite="INSAT-3DR", seed=11),
    dict(id="arb", name="Arabian Sea", lat=17, lng=66, event="Cyclone", satellite="INSAT-3D", seed=22),
    dict(id="wpac", name="West Pacific", lat=18, lng=135, event="Typhoon", satellite="Himawari-9", seed=33),
    dict(id="gom", name="Gulf of Mexico", lat=25, lng=-90, event="Hurricane", satellite="GOES-16", seed=44),
    dict(id="nati", name="North Atlantic", lat=24, lng=-55, event="Hurricane", satellite="GOES-16", seed=55),
    dict(id="sati", name="South Atlantic", lat=-18, lng=-25, event="Cyclone", satellite="Meteosat-11", seed=66),
    dict(id="sind", name="South Indian Ocean", lat=-18, lng=75, event="Cyclone", satellite="Meteosat-9", seed=77),
    dict(id="swpac", name="Southwest Pacific", lat=-20, lng=165, event="Cyclone", satellite="Himawari-9", seed=88),
    dict(id="coral", name="Coral Sea", lat=-17, lng=152, event="Cyclone", satellite="Himawari-9", seed=99),
    dict(id="carib", name="Caribbean Sea", lat=16, lng=-74, event="Hurricane", satellite="GOES-16", seed=111),
    dict(id="eaf", name="East Africa Coast", lat=-12, lng=45, event="Cyclone", satellite="Meteosat-9", seed=122),
    dict(id="mad", name="Madagascar Channel", lat=-18, lng=44, event="Cyclone", satellite="Meteosat-9", seed=133),
    dict(id="med", name="Mediterranean Sea", lat=35, lng=20, event="Storm", satellite="Meteosat-11", seed=144),
    dict(id="scs", name="South China Sea", lat=13, lng=115, event="Typhoon", satellite="Himawari-9", seed=155),
    dict(id="eas", name="East China Sea", lat=30, lng=128, event="Typhoon", satellite="Himawari-9", seed=166),
    dict(id="nind", name="North Indian Ocean", lat=12, lng=78, event="Cyclone", satellite="INSAT-3DR", seed=177),
    dict(id="biscay", name="Bay of Biscay", lat=45, lng=-5, event="Atlantic storm", satellite="Meteosat-11", seed=188),
    dict(id="bering", name="Bering Sea", lat=58, lng=178, event="Arctic storm", satellite="Himawari-9", seed=199),
    dict(id="tasman", name="Tasman Sea", lat=-40, lng=160, event="Cyclone", satellite="Himawari-9", seed=211),
    dict(id="pacse", name="Southeast Pacific", lat=-20, lng=-100, event="Pacific storm", satellite="GOES-18", seed=222),
    dict(id="pacne", name="Northeast Pacific", lat=20, lng=-125, event="Hurricane", satellite="GOES-18", seed=233),
]
TIMES = ["10:00", "10:10", "10:20", "10:30", "10:40"]
_cache = {}
net, MODEL_LOADED = RefineNet().eval(), False
if os.path.exists("refine.pt"):
    net.load_state_dict(torch.load("refine.pt", map_location="cpu")); MODEL_LOADED = True

def region(rid):
    r = next((x for x in REGIONS if x["id"] == rid), None)
    if not r: raise HTTPException(404, "Unknown region")
    if rid not in _cache: _cache[rid] = make_sequence(r["seed"])
    return _cache[rid]

def b64(img):
    ok, buf = cv2.imencode(".png", (np.clip(img, 0, 1) * 255).astype(np.uint8))
    return "data:image/png;base64," + base64.b64encode(buf).decode()

def metrics(p, g):
    # Identical images have infinite PSNR; JSON cannot encode Infinity.
    psnr_value = float(psnr(g, p, data_range=1))
    if not np.isfinite(psnr_value):
        psnr_value = 100.0
    return dict(psnr=round(psnr_value, 2),
                ssim=round(float(ssim(g, p, data_range=1)), 4),
                mae=round(float(np.abs(g - p).mean()), 4))

@app.get("/api/regions")
def regions(): return [{k: v for k, v in r.items() if k != "seed"} for r in REGIONS]

@app.get("/api/frames/{rid}")
def frames(rid: str):
    seq = region(rid)
    return [dict(idx=i, time=TIMES[i], image=b64(f)) for i, f in enumerate(seq)]

class Req(BaseModel):
    region: str
    start: int = 0

class TimedReq(BaseModel):
    region: str
    before: datetime
    target: datetime
    after: datetime
    lat: float | None = None
    lng: float | None = None

def as_utc(value: datetime):
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)

def gibs_layer(lng):
    if -155 <= lng < -15: return "GOES-East_ABI_Band13_Clean_Infrared"
    if lng < -155 or lng >= 170: return "GOES-West_ABI_Band13_Clean_Infrared"
    if 60 <= lng < 170: return "Himawari_AHI_Band13_Clean_Infrared"
    raise HTTPException(422, "NASA GIBS has no supported geostationary layer for this longitude. The current app supports GOES coverage in the Americas and Himawari coverage in Asia-Pacific; use Demo sequence for this area.")

@lru_cache(maxsize=24)
def fetch_gibs_image(layer: str, when: str, lat: float, lng: float):
    half_lat = 2.0
    half_lng = min(8.0, 2.5 / max(abs(np.cos(np.deg2rad(lat))), 0.25))
    bbox = (max(-180, lng - half_lng), max(-90, lat - half_lat),
            min(180, lng + half_lng), min(90, lat + half_lat))
    params = urlencode({
        "SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap",
        "LAYERS": layer, "STYLES": "", "SRS": "EPSG:4326",
        "BBOX": ",".join(f"{v:.6f}" for v in bbox),
        "WIDTH": "256", "HEIGHT": "256", "FORMAT": "image/png",
        "TRANSPARENT": "FALSE", "TIME": when,
    })
    url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?" + params
    try:
        request = URLRequest(url, headers={"User-Agent": "SatInterp/1.0"})
        with urlopen(request, timeout=45) as response:
            payload = response.read()
    except Exception as exc:
        raise HTTPException(502, f"NASA imagery request failed for {when}: {exc}") from exc
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        message = payload[:250].decode("utf-8", errors="replace")
        raise HTTPException(404, f"NASA has no usable {layer} image for {when}. {message}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0

def satellite_frame(region_data, moment):
    stamp = as_utc(moment).replace(second=0, microsecond=0)
    iso = stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    layer = gibs_layer(region_data["lng"])
    return fetch_gibs_image(layer, iso, float(region_data["lat"]), float(region_data["lng"])), iso

@app.post("/api/interpolate")
def interpolate(req: Req):
    seq = region(req.region)
    if not 0 <= req.start <= len(seq) - 3: raise HTTPException(400, "start must be 0..2")
    f0, gt, f1 = seq[req.start], seq[req.start + 1], seq[req.start + 2]
    t0 = time.time()
    flow, w0, w1, x = make_input(f0, f1)
    t_flow = time.time() - t0
    avg, warp = 0.5 * (f0 + f1), 0.5 * (w0 + w1)
    with torch.no_grad(): out = net(x)[0, 0].numpy()
    S = 16; h, w = f0.shape
    arrows = [dict(x=int(x_), y=int(y_), dx=float(flow[y_, x_, 0]), dy=float(flow[y_, x_, 1]))
              for y_ in range(S // 2, h, h // S) for x_ in range(S // 2, w, w // S)]
    mag = float(np.linalg.norm(flow, axis=-1).mean())
    return dict(
        times=dict(a=TIMES[req.start], mid=TIMES[req.start + 1], b=TIMES[req.start + 2]),
        frame_a=b64(f0), frame_b=b64(f1), generated=b64(out), ground_truth=b64(gt),
        arrows=arrows, size=[w, h], mean_motion_px=round(mag, 2), model_trained=MODEL_LOADED,
        metrics=dict(averaging=metrics(avg, gt), flow_warp=metrics(warp, gt), warp_refine=metrics(out, gt)),
        timing_ms=dict(flow=round(t_flow * 1000), total=round((time.time() - t0) * 1000)))

@app.post("/api/interpolate-times")
def interpolate_times(req: TimedReq):
    r = next((item for item in REGIONS if item["id"] == req.region), None)
    if not r: raise HTTPException(404, "Unknown region")
    if (req.lat is None) != (req.lng is None):
        raise HTTPException(400, "Send both latitude and longitude for a custom map point.")
    lat = float(r["lat"] if req.lat is None else req.lat)
    lng = float(r["lng"] if req.lng is None else req.lng)
    if not -90 <= lat <= 90 or not -180 <= lng <= 180:
        raise HTTPException(400, "Latitude must be -90..90 and longitude must be -180..180.")
    image_area = {**r, "lat": lat, "lng": lng}
    before, target, after = map(as_utc, (req.before, req.target, req.after))
    if not before < target < after:
        raise HTTPException(400, "Choose times in order: before frame, target time, after frame.")
    gap_minutes = (after - before).total_seconds() / 60
    if gap_minutes > 120:
        raise HTTPException(400, "Choose source frames no more than 120 minutes apart; long gaps make motion interpolation unreliable.")
    alpha = (target - before).total_seconds() / (after - before).total_seconds()
    f0, before_iso = satellite_frame(image_area, before)
    gt, target_iso = satellite_frame(image_area, target)
    f1, after_iso = satellite_frame(image_area, after)

    started = time.time()
    flow, w0, w1, x = make_input(f0, f1, alpha)
    flow_ms = time.time() - started
    avg = (1 - alpha) * f0 + alpha * f1
    warp = (1 - alpha) * w0 + alpha * w1
    with torch.no_grad(): out = net(x, alpha)[0, 0].numpy()
    residual = np.abs(out - gt)
    histogram_max = max(float(np.quantile(residual, 0.995)), 1e-5)
    hist_counts, hist_edges = np.histogram(residual, bins=20, range=(0, histogram_max))
    histogram_outliers = int((residual > histogram_max).sum())
    # Adapt thresholds to this frame's observed residual scale so the curve is
    # readable even when nearly all pixel errors are very small.
    tolerances = np.linspace(0, histogram_max, 20)
    similarity_curve = [round(float((residual <= threshold).mean()), 5) for threshold in tolerances]
    S = 16; h, w = f0.shape
    arrows = [dict(x=int(px), y=int(py), dx=float(flow[py, px, 0]), dy=float(flow[py, px, 1]))
              for py in range(S // 2, h, h // S) for px in range(S // 2, w, w // S)]
    mag = float(np.linalg.norm(flow, axis=-1).mean())
    return dict(
        times=dict(a=before_iso, mid=target_iso, b=after_iso),
        location=dict(lat=lat, lng=lng),
        diagnostics=dict(
            pixel_error_bins=[round(float(x), 4) for x in hist_edges[1:]],
            pixel_error_counts=[int(x) for x in hist_counts],
            histogram_max=round(histogram_max, 6), histogram_outliers=histogram_outliers,
            similarity_tolerances=[round(float(x), 4) for x in tolerances],
            similarity_curve=similarity_curve,
            loss_curve=[round(float(1 - value), 5) for value in similarity_curve],
            curve_note="Validation pixel similarity at error thresholds; not training epochs or classification accuracy.",
        ),
        frame_a=b64(f0), frame_b=b64(f1), generated=b64(out), ground_truth=b64(gt),
        arrows=arrows, size=[w, h], mean_motion_px=round(mag, 2), model_trained=MODEL_LOADED,
        metrics=dict(averaging=metrics(avg, gt), flow_warp=metrics(warp, gt), warp_refine=metrics(out, gt)),
        data_source="NASA GIBS", observation_note="NASA returns the nearest available satellite observation to each requested UTC time.",
        timing_ms=dict(flow=round(flow_ms * 1000), total=round((time.time() - started) * 1000)))

# python -m uvicorn main:app --reload --port 8001
# npm run dev
