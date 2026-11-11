#!/usr/bin/env python3
"""realized.py HELD_ROWS.json: at the served anchors that follow a rejection, with the gate's own verify count k of that step:
accepted drafts min(length, k) for the shipped drafter (torch mirror), the carry drafter, and what the engine actually accepted.
Then the first-order tok/s effect per category (fewer steps for the same tokens, plus the pass's extra cost)."""
import json, sys, collections
rows = json.load(open(sys.argv[1])); XMS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
reqs = []
for l in open("held_steps.jsonl"):
    try: r = json.loads(l)
    except Exception: continue
    if "req" in r: reqs.append([])
    elif reqs: reqs[-1].append(r)
gen = [json.loads(l) for l in open("held_gen.jsonl")]
cat = {i + (len(reqs) - len(gen)): g["cat"] for i, g in enumerate(gen)}
K = {}; S = collections.defaultdict(lambda: collections.Counter())
for qi, st in enumerate(reqs):
    c = cat.get(qi)
    for a, b in zip(st[:-1], st[1:]):
        K[(qi, a["p"] + a["n"] - 1)] = (b["R"] - 1, b["n"] - 1)
    if c:
        for s in st[1:]: S[c]["steps"] += 1; S[c]["tok"] += s["n"]; S[c]["rows"] += s["R"]
T = collections.defaultdict(lambda: collections.Counter())
for q, a, n, l0, l1 in rows:
    if (q, a) not in K or q not in cat: continue
    k, acc = K[(q, a)]; t = T[cat[q]]
    t["n"] += 1; t["k"] += k; t["eng"] += acc; t["m0"] += min(l0, k); t["m1"] += min(l1, k); t["l0"] += l0; t["l1"] += l1
    t["agree"] += min(l0, k) == acc
for c in ("chat", "code", "math"):
    t = T[c]; s = S[c]; n = t["n"]
    d = (t["m1"] - t["m0"])                       # more accepted drafts over the category's steps
    ms = {"chat": 25.3, "code": 30.1, "math": 30.5}[c]
    gain = (s["tok"] + d) / s["tok"] / (1 + XMS * n / (ms * s["steps"])) - 1
    print(f"{c}: {n} anchors after a rejection of {s['steps']} steps; drafts verified there {t['k']/n:.3f}; accepted: engine {t['eng']/n:.4f}, mirror shipped {t['m0']/n:.4f} (same as the engine at {100*t['agree']/n:.1f} %), carry {t['m1']/n:.4f} ({100*(t['m1']/t['m0']-1):+.2f} %); uncapped length {t['l0']/n:.3f} -> {t['l1']/n:.3f}; first-order tok/s {100*gain:+.2f} % (pass +{XMS} ms on those steps)")
