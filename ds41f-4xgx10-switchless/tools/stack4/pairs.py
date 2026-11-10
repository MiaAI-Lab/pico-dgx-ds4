#!/usr/bin/env python3
"""pairs.py A_STEPS A_GEN B_STEPS B_GEN: the same prompts served by two builds/flags: are the tokens identical; steps, drafts verified
and accepted per step by category (paired on identical token sequences)."""
import json, sys, collections
def load(steps, gen):
    g = [json.loads(l) for l in open(gen)]; reqs = []; cur = None
    for l in open(steps):
        r = json.loads(l)
        if "req" in r: cur = dict(steps=[]); reqs.append(cur)
        else: cur["steps"].append(r)
    reqs = reqs[-len(g):]
    for a, q in zip(g, reqs): q["id"] = a["id"]; q["cat"] = a["cat"]; q["text"] = a.get("text")
    return reqs
A = load(sys.argv[1], sys.argv[2]); B = load(sys.argv[3], sys.argv[4])
T = collections.defaultdict(lambda: collections.Counter())
same = 0
for a, b in zip(A, B):
    ta = [t for s in a["steps"] for t in s["out"][:s["n"]]]; tb = [t for s in b["steps"] for t in s["out"][:s["n"]]]
    same += ta == tb
    if ta != tb: continue
    for c in (a["cat"], "all"):
        k = T[c]; k["tok"] += len(ta)
        for nm, q in (("A", a), ("B", b)):
            k[nm + "steps"] += len(q["steps"]) - 1; k[nm + "ver"] += sum(s["R"] - 1 for s in q["steps"][1:]); k[nm + "acc"] += sum(s["n"] - 1 for s in q["steps"][1:])
print("requests", len(A), "token-identical", same)
for c in ("chat", "code", "math", "all"):
    k = T[c]
    print(f"{c}: tokens {k['tok']}; verify steps A {k['Asteps']} B {k['Bsteps']} ({100*(k['Bsteps']/k['Asteps']-1):+.2f} %); verified/step {k['Aver']/k['Asteps']:.3f} -> {k['Bver']/k['Bsteps']:.3f}; accepted/step {k['Aacc']/k['Asteps']:.3f} -> {k['Bacc']/k['Bsteps']:.3f} ({100*((k['Bacc']/k['Bsteps'])/(k['Aacc']/k['Asteps'])-1):+.2f} %); rows/step {1+k['Aver']/k['Asteps']:.3f} -> {1+k['Bver']/k['Bsteps']:.3f}")
