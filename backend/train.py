"""Train RefineNet. Default = synthetic triplets; point to real triplets by editing get_triplets()."""
import torch, numpy as np, torch.nn.functional as F
from model import RefineNet, make_input
from synthetic import make_sequence

def get_triplets(n_seq=60):
    for seed in range(1000, 1000 + n_seq):
        seq = make_sequence(seed)
        for i in range(3): yield seq[i], seq[i + 1], seq[i + 2]   # (t, t+10, t+20)

dev = "cuda" if torch.cuda.is_available() else "cpu"
net = RefineNet().to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
data = [(make_input(a, c)[3], torch.from_numpy(b)[None, None]) for a, b, c in get_triplets()]
for ep in range(15):
    tot = 0
    for x, y in data:
        x, y = x.to(dev), y.to(dev)
        p = net(x)
        gx = lambda t: t[..., :, 1:] - t[..., :, :-1]; gy = lambda t: t[..., 1:, :] - t[..., :-1, :]
        loss = F.l1_loss(p, y) + 0.5 * (F.l1_loss(gx(p), gx(y)) + F.l1_loss(gy(p), gy(y)))
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    print(f"epoch {ep+1} loss {tot/len(data):.5f}")
torch.save(net.state_dict(), "refine.pt")
