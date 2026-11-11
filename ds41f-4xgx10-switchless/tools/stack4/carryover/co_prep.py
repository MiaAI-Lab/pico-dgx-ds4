#!/usr/bin/env python3
"""co_prep.py DUMPDIR OUTDIR: join the decode-step dump (mh.bin: every step row's main_hidden, rejected rows too), the step log
(steps.jsonl) and the prefill feature dump (feat/c*.a*.bin) into one file per request:
  q%05d.npz: toks [T], mh [T, 15360] u16 (every position's final main_hidden: prompt rows from the prefill, the rest from the step
  that accepted them), P (prompt tokens), an [n, 3] = (anchor a = the last final position of a step, first index into rej, count),
  rej_mh [m, 15360] u16, rej_pos [m] (the position each rejected row was computed at), rej_tok [m] (its wrong token)."""
import sys, os, json, struct, glob
import numpy as np
D, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
MAXQ = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
K = 15360
# prefill feature files -> requests (a file whose name has .a0. starts a request)
feats = []
for f in sorted(glob.glob(D + "/feat/c*.bin")):
    a = int(f.rsplit(".a", 1)[1].split(".")[0])
    if a == 0: feats.append([f])
    elif feats: feats[-1].append(f)
def read_feat(files):
    toks, rows = [], []
    for f in files:
        b = np.fromfile(f, dtype=np.uint8)
        h = struct.unpack("<8i", b[:32].tobytes()); M = h[2]
        toks.append(np.frombuffer(b[32:32 + 4 * M].tobytes(), dtype=np.int32))
        rows.append(np.frombuffer(b[32 + 4 * M:].tobytes(), dtype=np.uint16).reshape(M, 20480)[:, :K])
    return np.concatenate(toks), np.concatenate(rows)
# step log -> requests
reqs = []
for l in open(D + "/steps.jsonl"):
    try: r = json.loads(l)
    except Exception: continue
    if "req" in r: reqs.append(dict(hdr=r, steps=[]))
    elif reqs: reqs[-1]["steps"].append(r)
mf = open(D + "/mh.bin", "rb")
def mh_requests():
    """per request of mh.bin: list of (i, pos0, R, n, rows u16 [R, K])"""
    cur = None
    while True:
        h = mf.read(16)
        if len(h) < 16:
            if cur is not None: yield cur
            return
        i, pos0, R, n = struct.unpack("<4i", h)
        if i == -1:
            if cur is not None: yield cur
            cur = []
        else:
            cur.append((i, pos0, R, n, np.frombuffer(mf.read(R * K * 2), dtype=np.uint16).reshape(R, K)))
print("requests: step log", len(reqs), "feature groups", len(feats))
nq = 0; stat = dict(ok=0, skip=0, anchors=0, rej_anchors=0, rej_rows=0)
def feat_len(files):
    n = 0
    for f in files:
        with open(f, "rb") as fh: n += struct.unpack("<8i", fh.read(32))[2]
    return n
FL = [feat_len(g) for g in feats]; fi = 0
for qi, (q, recs) in enumerate(zip(reqs, mh_requests())):
    if nq >= MAXQ: break
    st = q["steps"]
    if len(st) != len(recs) or not st: stat["skip"] += 1; continue
    P = st[0]["p"]
    j = next((x for x in range(fi, min(fi + 4, len(FL))) if FL[x] == P), None)   # the request's prefill features: the next group of P rows
    if j is None: stat["skip"] += 1; continue
    fi = j + 1
    ptoks, pmh = read_feat(feats[j])
    T = P + sum(s["n"] for s in st) + 1
    toks = np.zeros(T, np.int32); mh = np.zeros((T, K), np.uint16); toks[:P] = ptoks; mh[:P] = pmh
    toks[P] = st[0]["in"][0]
    an = []; rmh = []; rpos = []; rtok = []; ok = True
    for s, (i, pos0, R, n, rows) in zip(st, recs):
        if s["p"] != pos0 or s["R"] != R or s["n"] != n: ok = False; break
        for j in range(n):                       # rows 0 .. n-1 are final (row 0: the bonus; then the accepted drafts)
            mh[pos0 + j] = rows[j]; toks[pos0 + j] = s["in"][j]
        toks[pos0 + n] = s["out"][n - 1]         # the token after them (the next bonus)
        a = pos0 + n - 1; k0 = len(rpos)
        for j in range(n, R):                    # rejected rows: computed at pos0 + j on a wrong token
            rmh.append(rows[j]); rpos.append(pos0 + j); rtok.append(s["in"][j])
        an.append((a, k0, R - n))
    if not ok: stat["skip"] += 1; continue
    T2 = st[-1]["p"] + st[-1]["n"] + 1
    np.savez(OUT + "/q%05d.npz" % nq, toks=toks[:T2], mh=mh[:T2], P=P, an=np.array(an, np.int32).reshape(-1, 3),
             rej_mh=np.array(rmh, np.uint16).reshape(-1, K), rej_pos=np.array(rpos, np.int32), rej_tok=np.array(rtok, np.int32), qi=qi)
    nq += 1; stat["ok"] += 1; stat["anchors"] += len(an); stat["rej_anchors"] += sum(1 for x in an if x[2]); stat["rej_rows"] += len(rpos)
print(stat)
