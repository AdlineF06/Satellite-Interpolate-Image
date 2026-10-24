"""Warp-and-Refine: optical flow warps both frames to t=0.5, a small U-Net refines."""
import numpy as np, cv2, torch, torch.nn as nn

def flow_01(f0, f1):
    a, b = (f0 * 255).astype(np.uint8), (f1 * 255).astype(np.uint8)
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 3, 7, 1.5, 0)

def _remap(img, flow, s):
    h, w = img.shape
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    return cv2.remap(img, (gx + s * flow[..., 0]).astype(np.float32),
                     (gy + s * flow[..., 1]).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def warp_to_time(f0, f1, flow, alpha=0.5):
    """Backward warp both endpoints toward a fractional time in the interval."""
    return _remap(f0, flow, -alpha), _remap(f1, flow, 1 - alpha)

def _blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.PReLU(),
                                     nn.Conv2d(o, o, 3, padding=1), nn.PReLU())

class RefineNet(nn.Module):
    """Input: [w0, w1, |flow|]  ->  residual added to mean(w0, w1)."""
    def __init__(s, c=32):
        super().__init__()
        s.e1, s.e2, s.e3 = _blk(3, c), _blk(c, 2 * c), _blk(2 * c, 4 * c)
        s.pool = nn.MaxPool2d(2)
        s.u2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, 2); s.d2 = _blk(4 * c, 2 * c)
        s.u1 = nn.ConvTranspose2d(2 * c, c, 2, 2);     s.d1 = _blk(2 * c, c)
        s.out = nn.Conv2d(c, 1, 3, padding=1)
        nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)   # starts as plain blend
    def forward(s, x, alpha=0.5):
        e1 = s.e1(x); e2 = s.e2(s.pool(e1)); e3 = s.e3(s.pool(e2))
        d2 = s.d2(torch.cat([s.u2(e3), e2], 1)); d1 = s.d1(torch.cat([s.u1(d2), e1], 1))
        return ((1 - alpha) * x[:, :1] + alpha * x[:, 1:2] + s.out(d1)).clamp(0, 1)

def make_input(f0, f1, alpha=0.5):
    flow = flow_01(f0, f1)
    w0, w1 = warp_to_time(f0, f1, flow, alpha)
    mag = np.linalg.norm(flow, axis=-1) / 10.0
    x = np.stack([w0, w1, mag])[None].astype(np.float32)
    return flow, w0, w1, torch.from_numpy(x)
