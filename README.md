# Orbit Interpolator – Smart Satellite Frame Interpolation

React + react-globe.gl frontend, FastAPI + PyTorch/OpenCV backend.
Pipeline: preprocess → optical flow (Farneback) → warp both frames to t+0.5 → U-Net refine → PSNR/SSIM/MAE vs ground truth.

## Run
Backend (terminal 1):
    cd backend
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python train.py                 # optional, creates refine.pt (a few minutes on CPU)
    uvicorn main:app --reload --port 8000

Frontend (terminal 2):
    cd frontend
    npm install
    npm run dev                     # open http://localhost:5173

## Use real satellite data
Select an area and choose **NASA satellite imagery**. Enter the before, target, and after timestamps in your local time; the backend converts them to UTC and fetches infrared imagery for that region from NASA GIBS. Choose **Demo sequence** to use the offline procedural examples instead.

NASA GIBS uses the nearest available observation for each requested timestamp, so the source images are not guaranteed to match a minute exactly. Geostationary coverage is typically at about 10-minute intervals and the archive is limited (generally around 90 days). Source frames must be within 120 minutes of each other. The selected target timestamp is fetched independently and used to compare the generated frame with the observed image. GIBS access requires an internet connection; no extra account token is required for these public imagery requests. This app currently routes the Americas to GOES and Asia-Pacific to Himawari; locations outside those feeds will show a coverage message and can still use the synthetic demo mode.

For model training, `train.py` still uses synthetic midpoint examples. A trained model may not perform well on real infrared satellite imagery without fine-tuning on appropriately prepared real satellite frames.
