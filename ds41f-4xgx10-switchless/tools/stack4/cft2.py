#!/usr/bin/env python3
"""cft2.py NAME FEATS HID: an engine-sized confidence MLP: one evaluation per anchor, input = chosen scalars of all 5 depths
(+ the target's log p of the bonus), output 5 logits; logit_j = a_j * shipped_j + out_j. Trained on the conditional rows
(a label exists and all earlier drafts hit) of tr0-2, selected on the last 5 %, scored on held; writes Q_NAME.npy [N, 5] and W_NAME.npz.
FEATS: comma list of c (shipped logit), v (top logit), m (margin), l (log p1), p (p2), e (entropy), t (target log p(bonus)), T (all 3 target scalars)"""
import sys, os, numpy as np, torch, json
L3 = os.path.expanduser("~/.hermes/cache/scratch/l3/data/")
NAME, FE, HID = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
col = dict(c=0, v=1, m=2, l=3, p=4, e=5)
def load(d):
    S, T, H = np.load(L3 + d + "/S.npy"), np.load(L3 + d + "/T.npy"), np.load(L3 + d + "/H.npy")
    X = [S[:, :, col[f]] for f in FE if f in col]
    if "t" in FE: X.append(T[:, :1])
    if "T" in FE: X.append(T[:, :3])
    X = np.concatenate(X, 1).astype(np.float32)
    cond = np.concatenate([np.ones((len(H), 1), bool), np.cumprod(H[:, :-1] == 1, 1).astype(bool)], 1) & (H >= 0)
    return X, S[:, :, 0].astype(np.float32), (H == 1).astype(np.float32), cond.astype(np.float32)
tr = [load("cft/tr%d" % i) for i in range(3)]
Xt, Ct, Yt, Mt = (np.concatenate([t[k] for t in tr]) for k in range(4))
Xh, Ch, Yh, Mh = load("cfh")
n = len(Xt); nv = n // 20
mu, sd = Xt[:-nv].mean(0), Xt[:-nv].std(0) + 1e-6
torch.manual_seed(1)
class Net(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.l1 = torch.nn.Linear(Xt.shape[1], HID); s.l2 = torch.nn.Linear(HID, 5); s.a = torch.nn.Parameter(torch.ones(5))
        torch.nn.init.zeros_(s.l2.weight); torch.nn.init.zeros_(s.l2.bias)
    def forward(s, x, c): return s.a * c + s.l2(torch.relu(s.l1(x)))
net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.01)
T = lambda a: torch.from_numpy(a)
Xn = T((Xt - mu) / sd); Ct_, Yt_, Mt_ = T(Ct), T(Yt), T(Mt)
Xhn = T((Xh - mu) / sd); Ch_, Yh_, Mh_ = T(Ch), T(Yh), T(Mh)
def loss(x, c, y, m): return (torch.nn.functional.binary_cross_entropy_with_logits(net(x, c), y, reduction="none") * m).sum() / m.sum()
best = (9, None); EP = 10; sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=EP * ((n - nv) // 4096 + 1))
for ep in range(EP):
    perm = torch.randperm(n - nv)
    for i in range(0, n - nv, 4096):
        b = perm[i:i + 4096]; l = loss(Xn[b], Ct_[b], Yt_[b], Mt_[b]); opt.zero_grad(); l.backward(); opt.step(); sched.step()
    with torch.no_grad(): v = loss(Xn[-nv:], Ct_[-nv:], Yt_[-nv:], Mt_[-nv:]).item()
    if v < best[0]: best = (v, {k: t.clone() for k, t in net.state_dict().items()})
net.load_state_dict(best[1])
from sklearn.metrics import roc_auc_score
with torch.no_grad():
    Q = net(Xhn, Ch_).numpy(); lh = loss(Xhn, Ch_, Yh_, Mh_).item()
    base = (torch.nn.functional.binary_cross_entropy_with_logits(Ch_, Yh_, reduction="none") * Mh_).sum().item() / Mh.sum()
m = Mh.astype(bool)
print(f"{NAME}: feats {FE} hid {HID}: val {best[0]:.4f}; held log-loss {lh:.4f} (shipped {base:.4f}), AUC {roc_auc_score(Yh[m], Q[m]):.4f} (shipped {roc_auc_score(Yh[m], Ch[m]):.4f})")
np.save("Q_%s.npy" % NAME, Q)
sdct = net.state_dict()
np.savez("W_%s.npz" % NAME, mu=mu, sd=sd, w1=sdct["l1.weight"].numpy(), b1=sdct["l1.bias"].numpy(), w2=sdct["l2.weight"].numpy(), b2=sdct["l2.bias"].numpy(), a=sdct["a"].numpy(), feats=np.array(FE))
