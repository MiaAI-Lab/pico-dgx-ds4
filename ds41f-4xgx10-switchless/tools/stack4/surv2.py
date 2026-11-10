#!/usr/bin/env python3
"""surv2.py: certified head with ONE global threshold per row (cross-rank max of the lower bounds), coarse codes of 5..8 bits.
minifloat(M, E): sign, M mantissa bits, exponent offset below the largest exponent of the row's group of 32 values (2^E - 1
binades, one code for zero). Survivors = rows whose upper bound reaches the threshold; fallback = more than CAP 8-row tiles."""
import numpy as np, torch, json, os, sys, time
HD = os.path.expanduser("~/.hermes/cache/scratch/s4/hd/"); V, K, RK = 129280, 5120, 32320
X = torch.from_numpy(np.load(HD + "X.npy")); rows = json.load(open(HD + "rows.json")); N = X.shape[0]; BL = 16
dev = "mps"
Wc = torch.from_numpy(np.load(HD + "W.npy"))
nw = torch.from_numpy(np.load(HD + "norms.npy"))[0].float()
def minifloat(w, M, E):
    n = w.shape[0]; g = w.view(n, K // 32, 32); a = g.abs()
    emax = torch.floor(torch.log2(a.amax(dim=2, keepdim=True).clamp(min=1e-30)))
    e = torch.floor(torch.log2(a.clamp(min=1e-30)))
    nb = 2 ** E - 1                                   # binades kept
    emin = emax - (nb - 1)
    ec = torch.maximum(e, emin)                       # values below the lowest binade use its grid (down to 0)
    step = torch.pow(2.0, ec - M)
    q = torch.round(a / step) * step                  # may round up into the next binade: representable there (or the group max + 1 ulp: clamp)
    top = torch.pow(2.0, emax) * (2 - 2.0 ** -M)
    q = torch.minimum(q, top)
    lowest = torch.pow(2.0, emin)
    q = torch.where(a < lowest, torch.where(a >= lowest / 2, lowest, torch.zeros_like(a)), q)   # below the lowest binade: zero or the least value
    return (q * g.sign()).view(n, K)
fmts = [("e4m3 FP8 copy (8 bit)", "W8.npy", None, 8.0 + 8 / 1024), ("minifloat m3e3 (7 bit)", None, (3, 3), 7.16), ("minifloat m2e3 (6 bit)", None, (2, 3), 6.16), ("minifloat m2e2 (5 bit)", None, (2, 2), 5.16), ("minifloat m1e3 (5 bit)", None, (1, 3), 5.16), ("minifloat m1e2 (4 bit)", None, (1, 2), 4.16)]
xn = X.double().norm(dim=1).float(); U = 2.0 ** -22; BS = 128
EXACT_BITS = 12.2
W = Wc.to(dev)
out = []
for name, f, me, bits in fmts:
    t0 = time.time()
    if f: Wq = torch.from_numpy(np.load(HD + f))
    else:
        Wq = torch.empty_like(Wc)
        for r0 in range(0, V, 3232): Wq[r0:r0 + 3232] = minifloat(Wc[r0:r0 + 3232], *me)
    rho = torch.cat([(Wc[r0:r0 + 3232].double() - Wq[r0:r0 + 3232].double()).norm(dim=1) for r0 in range(0, V, 3232)]).float()
    nq = torch.cat([Wq[r0:r0 + 3232].double().norm(dim=1) for r0 in range(0, V, 3232)]).float()
    bound = ((rho + K * U * (nw + nq)) * 1.0001).to(dev); Wq = Wq.to(dev)
    T8 = torch.empty(N, V // 8, dtype=torch.bool); cnt = []; wrong = 0
    for i in range(0, N, BS):
        x = X[i:i + BS].to(dev); n = x.shape[0]
        L = x @ W.t(); Cq = x @ Wq.t(); am = L.argmax(1)
        B = xn[i:i + BS].to(dev)[:, None] * bound[None, :]
        sg = (Cq + B) >= (Cq - B).max(1).values[:, None]
        assert bool(sg[torch.arange(n), am].all()), "certificate excluded the exact argmax"
        cnt.append(sg.sum(1).cpu()); T8[i:i + n] = sg.view(n, V // 8, 8).any(2).cpu(); wrong += (Cq.argmax(1) != am).sum().item()
    del Wq
    cnt = torch.cat(cnt).float(); t8 = T8.float().sum(1)
    line = f"{name}: residual {100*(rho/nw).mean():.2f} % of ||w||, bound +-{(xn.mean()*bound.mean().cpu()):.1f} logits; coarse argmax wrong {100*wrong/N:.2f} %; survivors/row median {cnt.median():.0f} mean {cnt.mean():.0f} p90 {cnt.quantile(.9):.0f} p99 {cnt.quantile(.99):.0f} max {cnt.max():.0f}"
    res = {}
    for R in (1, 2, 3, 4, 6):
        u = []
        for s in range(0, N - BL + 1, BL):
            for o in range(0, BL - R + 1, R): u.append(T8[s + o:s + o + R].any(0).float().sum().item())
        u = np.array(u); tiles = V // 8
        # bytes relative to the coded exact head: coarse (all rows) + exact re-read of surviving tiles; fallback (> 25 % of tiles) reads everything
        fb = u > 0.25 * tiles
        rel = bits / EXACT_BITS + np.where(fb, 1.0, u / tiles)
        res[R] = (float(np.mean(u / tiles)), float(fb.mean()), float(rel.mean()))
    line += " | head bytes vs today (coarse + survivors, fallback = full): " + ", ".join(f"R={R}: {100*v[2]:.1f} % (tiles {100*v[0]:.2f} %, fallback {100*v[1]:.2f} %)" for R, v in res.items())
    print(line, f"[{time.time()-t0:.0f}s]", flush=True)
