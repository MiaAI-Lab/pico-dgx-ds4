#!/usr/bin/env python3
"""conc_img.py URL OUT.json [with_image 0|1] : 8 distinct text users (512 tok, temp 0, streamed); with_image=1 fires one
image request 4 s in. Reports each text user's tok/s overall and in windows outside / inside the image's prefill."""
import base64, json, os, sys, threading, time, urllib.request
URL, OUT = sys.argv[1], sys.argv[2]; MODE = sys.argv[3] if len(sys.argv) > 3 else "0"  # 0 none, 1 image, 2 text control
WITH = MODE in ("1", "2")
KEY = open(os.environ["KEYFILE"]).read().strip()
P = ["Write a Python LRU cache class with get/put, type hints, docstrings and unit tests.",
 "Write a Rust function that parses a CSV line with quoted fields, plus tests.",
 "Implement Dijkstra's algorithm in TypeScript with a binary heap, with comments.",
 "Write a Go HTTP server with graceful shutdown, a /health route and middleware for request logging.",
 "Implement a thread-safe bounded blocking queue in Java with put/take and timeouts.",
 "Write a C function that reverses a singly linked list in place, plus a small test harness.",
 "Write a SQL schema for a library system (books, members, loans) with indexes and three example queries.",
 "Implement a trie in C++ with insert, search, prefix count and delete."]
def post(content, mt):
    b = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": content}], "max_tokens": mt, "temperature": 0.0,
         "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": False}}
    return urllib.request.Request(URL, json.dumps(b).encode(), {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
T = [[] for _ in P]; img = {}
def user(i):
    with urllib.request.urlopen(post("[%d] " % i + P[i], 512), timeout=600) as r:
        for line in r:
            if line.startswith(b"data:") and b'"content"' in line:
                d = json.loads(line[5:]); c = (d.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                if c: T[i].append(time.time())
def image():
    d = "data:image/png;base64," + base64.b64encode(open(os.path.join(os.environ.get("IMGDIR", "."), "05_bar_chart.png"), "rb").read()).decode()
    q = "Which day has the most orders? Answer as 'day, number'."
    if MODE == "2":  # a text request of the same length (~455 prompt tokens) at the same moment: the control
        content = " ".join("Day %d had %d orders." % (i, (i * 37) % 50) for i in range(64)) + " " + q
    else:
        content = [{"type": "image_url", "image_url": {"url": d}}, {"type": "text", "text": q}]
    t0 = time.time(); first = None; txt = ""
    with urllib.request.urlopen(post(content, 40), timeout=600) as r:
        for line in r:
            if line.startswith(b"data:") and b'"content"' in line:
                d = json.loads(line[5:]); c = (d.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                if c: first = first or time.time(); txt += c
    img.update(t0=t0, first=first, end=time.time(), text=txt)
th = [threading.Thread(target=user, args=(i,)) for i in range(len(P))]; t0 = time.time(); [t.start() for t in th]
if WITH:
    time.sleep(4); ti = threading.Thread(target=image); ti.start(); ti.join()
[t.join() for t in th]
def rate(ts, a, b):
    n = sum(1 for t in ts if a <= t < b); return n / max(1e-6, b - a)
res = {"mode": {"0": "none", "1": "image", "2": "text_control"}[MODE], "users": []}
for ts in T:
    u = {"tokens": len(ts), "tps": round((len(ts) - 1) / max(1e-6, ts[-1] - ts[0]), 1)}
    if WITH and img.get("first"):
        a, b = img["t0"], img["first"]
        u["tps_before"] = round(rate(ts, ts[0], a), 1); u["tps_during_image_prefill"] = round(rate(ts, a, b), 1)
        u["tps_after"] = round(rate(ts, b, min(ts[-1], b + 6)), 1)
    else:
        mid = (ts[0] + ts[-1]) / 2; u["tps_steady_mid6s"] = round(rate(ts, mid - 3, mid + 3), 1)
    res["users"].append(u)
res["mean_tps"] = round(sum(u["tps"] for u in res["users"]) / len(P), 2)
res["total_tps"] = round(sum(u["tokens"] for u in res["users"]) / (max(max(t) for t in T) - t0), 1)
if WITH:
    res["image"] = {"ttft": round(img["first"] - img["t0"], 3), "text": img["text"]}
    for k in ("tps_before", "tps_during_image_prefill", "tps_after"):
        res["mean_" + k] = round(sum(u[k] for u in res["users"]) / len(P), 2)
print(json.dumps(res)); json.dump(res, open(OUT, "w"), indent=1)
