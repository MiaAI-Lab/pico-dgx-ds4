#!/usr/bin/env python3
"""catms.py NAME=RANK0LOG:GEN.jsonl ... : per-category decode tok/s (tokens / wall of the engine's own session lines) for runs of
the same prompts; texts compared with the first run."""
import re, sys, json, collections
runs = []
for a in sys.argv[1:]:
    name, rest = a.split("=", 1); logp, genp = rest.split(":")
    log = open(logp, errors="replace").read()
    ss = re.findall(r"req (\d+) session \S+ prompt (\d+), reused \d+ \((?:cold|warm)[^)]*\), prefill ([\d.]+) ms \([\d.]+ tok/s\), (\d+) tokens in ([\d.]+) ms", log)
    dr = {int(m[0]): (int(m[1]), float(m[2]), float(m[3])) for m in re.findall(r"req (\d+) drafts: (\d+) verify steps, ([\d.]+) drafts verified and ([\d.]+) accepted per step", log)}
    gen = [json.loads(l) for l in open(genp)]
    ss = ss[-len(gen):] if len(ss) >= len(gen) else ss
    runs.append((name, ss, gen, dr))
ref = runs[0]
for name, ss, gen, dr in runs:
    T = collections.defaultdict(lambda: [0, 0.0, 0, 0.0, 0.0])
    same = sum(1 for a, b in zip(gen, ref[2]) if a.get("text") == b.get("text"))
    for s, g in zip(ss, gen):
        for c in (g["cat"], "all"):
            t = T[c]; t[0] += int(s[3]); t[1] += float(s[4])
            if int(s[0]) in dr: v = dr[int(s[0])]; t[2] += v[0]; t[3] += v[0] * v[1]; t[4] += v[0] * v[2]
    print(f"{name}: texts identical to {ref[0]}: {same}/{len(gen)}; " + " | ".join(f"{c} {1000*T[c][0]/T[c][1]:.2f} tok/s ({100*((1000*T[c][0]/T[c][1])/BASE[c]-1) if 'BASE' in globals() else 0:+.2f} %), ver {T[c][3]/max(1,T[c][2]):.2f} acc {T[c][4]/max(1,T[c][2]):.2f}" for c in ("chat", "code", "math", "all")))
    if "BASE" not in globals(): BASE = {c: 1000 * T[c][0] / T[c][1] for c in T}
