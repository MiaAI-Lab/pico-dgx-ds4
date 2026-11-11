#!/usr/bin/env python3
"""The bench behind the README numbers. Streams an OpenAI-compatible /v1/chat/completions.

  bench.py URL c1 [RUNS]        one user: RUNS chat + RUNS code requests (default 9), medians
  bench.py URL distinct N       N users at once, a different coding prompt each (N <= 16)
  bench.py URL copies N         N users at once, 3 prompts repeated

512 output tokens per request, temperature 0, thinking off. Each prompt gets a random prefix so
the prefix cache cannot help. The API key is read from $DSGB10_KEY, or from the file named by
$DSGB10_KEY_FILE. Run it with no other traffic on the endpoint. Chat medians of one build
swing by about +-3 tok/s between runs, so compare builds with many runs.
"""
import json, os, random, statistics, sys, threading, time, urllib.request

URL, MODE = sys.argv[1], sys.argv[2]
ARG = int(sys.argv[3]) if len(sys.argv) > 3 else 9
MODEL = os.environ.get("DSGB10_MODEL", "deepseek-v4.1-flash")
KEY = os.environ.get("DSGB10_KEY") or (
    open(os.path.expanduser(os.environ["DSGB10_KEY_FILE"])).read().strip() if os.environ.get("DSGB10_KEY_FILE") else "")

CHAT = ["Explain how a refrigerator works, in plain language, with a few paragraphs.",
        "Write a short story about a lighthouse keeper who finds a message in a bottle.",
        "Compare the Roman and Han empires: government, economy, military, legacy."]
CODE = ["Write a Python LRU cache class with get/put, type hints, docstrings and unit tests.",
        "Write a Rust function that parses a CSV line with quoted fields, plus tests.",
        "Implement Dijkstra's algorithm in TypeScript with a binary heap, with comments.",
        "Write a Go HTTP server with graceful shutdown, a /health route and middleware for request logging.",
        "Implement a thread-safe bounded blocking queue in Java with put/take and timeouts.",
        "Write a C function that reverses a singly linked list in place, plus a small test harness.",
        "Write a SQL schema for a library system (books, members, loans) with indexes and three example queries.",
        "Implement a trie in C++ with insert, search, prefix count and delete.",
        "Write a bash script that rotates log files older than 7 days and compresses them with gzip.",
        "Write a React component in TSX for a paginated, sortable table with a search box.",
        "Implement the A* pathfinding algorithm in Python on a 2D grid with obstacles.",
        "Write a Kotlin data class and repository for a todo app using coroutines and Flow.",
        "Implement a token-bucket rate limiter in Rust that is safe across threads.",
        "Write a Python asyncio web scraper that fetches 50 URLs with a concurrency limit of 5 and retries.",
        "Implement a Swift struct for a matrix with addition, multiplication and transpose, plus tests.",
        "Write a Haskell function for run-length encoding and decoding of strings with QuickCheck properties."]


def req(prompt, max_tokens=512):
    body = {"model": MODEL, "messages": [{"role": "user", "content": f"[{random.random():.8f}] " + prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": False}}
    head = {"Content-Type": "application/json"}
    if KEY:
        head["Authorization"] = "Bearer " + KEY
    r = urllib.request.Request(URL, json.dumps(body).encode(), head)
    t0 = time.time(); first = None; n = 0; usage = None
    with urllib.request.urlopen(r, timeout=900) as resp:
        for line in resp:
            line = line.strip()
            if not line.startswith(b"data:") or line.endswith(b"[DONE]"):
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for c in d.get("choices", []):
                if c.get("delta", {}).get("content") or c.get("delta", {}).get("reasoning_content"):
                    first = first or time.time(); n += 1
    t1 = time.time()
    toks = (usage or {}).get("completion_tokens", n)
    return toks, (toks - 1) / max(1e-6, t1 - (first or t0))


req("hi", 8)  # warm
if MODE == "c1":
    for kind, P in (("chat", CHAT), ("code", CODE[:3])):
        v = [round(req(P[i % 3])[1], 1) for i in range(ARG)]
        print(f"{kind} 1 user: median {statistics.median(v):.1f} tok/s  {v}", flush=True)
else:
    P = CODE if MODE == "distinct" else CODE[:3]
    res = [0] * ARG
    def w(i): res[i] = req(P[i % len(P)])[0]
    th = [threading.Thread(target=w, args=(i,)) for i in range(ARG)]
    t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]; wall = time.time() - t0
    print(f"{MODE} {ARG} users: {sum(res) / wall:.1f} tok/s total  ({sum(res)} tokens in {wall:.1f} s)", flush=True)
