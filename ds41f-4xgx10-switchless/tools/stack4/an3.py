#!/usr/bin/env python3
"""an3.py STEPS.jsonl [W.npz]: engine-side confidence features (step log "cf": shipped logits, margins, log p1 of the 5 depths)
with labels from the next step's verdict -> X.npy, Y.npy, M.npy; compares with the torch-mirror training distribution and scores
the Mac-trained MLP and the shipped head on them."""
import sys, json, numpy as np
steps = []; reqs = []
for l in open(sys.argv[1]):
    try: r = json.loads(l)
    except Exception: continue
    if "req" in r: reqs.append([])
    elif reqs: reqs[-1].append(r)
X = []; Y = []; M = []; RQ = []
for qi, st in enumerate(reqs):
    for a, b in zip(st[:-1], st[1:]):
        if "cf" not in a or b["in"][1:] != a["d"][:b["R"] - 1]: continue
        k = b["R"] - 1; acc = b["n"] - 1
        y = [0] * 5; m = [0] * 5
        for j in range(k):
            if j < acc: y[j] = 1; m[j] = 1
            elif j == acc: m[j] = 1
        X.append(a["cf"]); Y.append(y); M.append(m); RQ.append(qi)
X = np.array(X, np.float32); Y = np.array(Y, np.float32); M = np.array(M, np.float32); RQ = np.array(RQ)
out = sys.argv[1].rsplit("/", 1)[-1].replace(".jsonl", "")
np.savez(out + "_xy.npz", X=X, Y=Y, M=M, RQ=RQ)
print("anchors", len(X), "labelled rows", int(M.sum()), "per depth", M.sum(0).astype(int), "hit rate per depth", (Y.sum(0) / np.maximum(M.sum(0), 1)).round(3))
W = np.load(sys.argv[2] if len(sys.argv) > 2 else "W_cml.npz")
print("feature means engine  :", X.mean(0).round(2)); print("feature means training:", W["mu"].round(2)); print("feature sd engine  :", X.std(0).round(2)); print("feature sd training:", W["sd"].round(2))
def bce(lg, y, m): return float((m * (np.logaddexp(0, lg) - y * lg)).sum() / m.sum())
c = X[:, :5]
h = np.maximum(((X - W["mu"]) / W["sd"]) @ W["w1"].T + W["b1"], 0); q = W["a"] * c + h @ W["w2"].T + W["b2"]
from sklearn.metrics import roc_auc_score
mm = M.astype(bool)
print("log-loss on engine rows: shipped %.4f, Mac-trained MLP %.4f | AUC shipped %.4f, MLP %.4f" % (bce(c, Y, M), bce(q, Y, M), roc_auc_score(Y[mm], c[mm]), roc_auc_score(Y[mm], q[mm])))
