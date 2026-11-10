#!/usr/bin/env python3
"""stepms.py RANK0LOG [SKIP_FIRST_N_REQS] [LAST_N]: per-request decode stats from the engine log: tokens, verify steps, GPU ms by rows -> tok/s by GPU time and wall."""
import re, sys, collections
log = open(sys.argv[1], errors="replace").read()
dr = re.findall(r"req (\d+) drafts: (\d+) verify steps, ([\d.]+) drafts verified and ([\d.]+) accepted per step, steps verifying 0\.\.5: ([\d ]+); gate [^;]*; GPU ms by rows: ([^\n]*)", log)
ss = {int(m[0]): m for m in re.findall(r"req (\d+) session \S+ prompt (\d+), reused \d+ \((?:cold|warm)[^)]*\), prefill ([\d.]+) ms \([\d.]+ tok/s\), (\d+) tokens in ([\d.]+) ms", log)}
a = int(sys.argv[2]) if len(sys.argv) > 2 else 0; b = int(sys.argv[3]) if len(sys.argv) > 3 else None
dr = dr[a:] if b is None else dr[a:a + b]
tot = collections.Counter(); byR = collections.defaultdict(lambda: [0, 0.0])
for m in dr:
    q = int(m[0]); vs = int(m[1]); tot["vs"] += vs; tot["ver"] += vs * float(m[2]); tot["acc"] += vs * float(m[3])
    for R, t, n in re.findall(r"(\d+): ([\d.]+) \((\d+)\)", m[5]):
        byR[int(R)][0] += int(n); byR[int(R)][1] += float(t) * int(n)
    if q in ss: tot["tok"] += int(ss[q][3]); tot["wall"] += float(ss[q][4]); tot["n"] += 1
ng = sum(v[0] for v in byR.values()); gm = sum(v[1] for v in byR.values())
print(f"reqs {len(dr)} ({tot['n']} with a session line): verify steps {tot['vs']}, verified {tot['ver']/tot['vs']:.3f}, accepted {tot['acc']/tot['vs']:.3f} per step")
print(f"  GPU ms/step {gm/ng:.3f} over {ng} timed steps; by rows: " + ", ".join(f"{R}: {v[1]/v[0]:.2f} ({v[0]})" for R, v in sorted(byR.items())))
if tot["n"]: print(f"  tokens {tot['tok']}, wall {tot['wall']:.0f} ms -> {1000*tot['tok']/tot['wall']:.2f} tok/s decode; wall/step {tot['wall']/tot['vs']:.3f} ms; tok/step {tot['tok']/tot['vs']:.3f}")
