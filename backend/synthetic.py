"""Procedural cyclone sequences so the project runs without downloading data.
Replace with real Himawari/GOES/INSAT frames (see README)."""
import numpy as np, cv2

def _norm(a): return (a - a.min()) / (a.max() - a.min() + 1e-8)

def make_sequence(seed, n=5, S=256):
    rng = np.random.default_rng(seed)
    b1 = _norm(cv2.GaussianBlur(rng.random((S, S)).astype(np.float32), (0, 0), 3))
    b2 = _norm(cv2.GaussianBlur(rng.random((S, S)).astype(np.float32), (0, 0), 9))
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    spin = 0.22 + 0.1 * rng.random()
    frames = []
    for t in range(n):
        cx, cy = S / 2 + 3 * t, S / 2 - 2 * t
        dx, dy = xx - cx, yy - cy
        r = np.sqrt(dx ** 2 + dy ** 2) + 1e-3
        th = np.arctan2(dy, dx)
        ths = th - spin * t * np.exp(-r / 110)          # differential rotation
        mx = (S / 2 + r * np.cos(ths)).astype(np.float32)
        my = (S / 2 + r * np.sin(ths)).astype(np.float32)
        n1 = cv2.remap(b1, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        n2 = cv2.remap(b2, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        spiral = 0.5 + 0.5 * np.cos(2 * ths + 0.06 * r)
        env = np.exp(-(r / 105) ** 2)
        eye = 1 - 0.9 * np.exp(-(r / 12) ** 2)
        img = eye * (0.55 * env * spiral + 0.3 * n1 + 0.2 * n2)
        frames.append(np.clip(_norm(img), 0, 1).astype(np.float32))
    return frames
