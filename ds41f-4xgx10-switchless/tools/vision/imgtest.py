#!/usr/bin/env python3
"""imgtest.py URL IMGDIR OUT.json : image correctness + TTFT through the OpenAI API (base64 data URLs and an
http URL). For every case also the same request without its image, so that the TTFT an image adds is measured."""
import base64, json, os, sys, time, urllib.request

URL, DIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]  # KEYFILE=<file with the API key>
KEY = open(os.environ["KEYFILE"]).read().strip()
MODEL = os.environ.get("MODEL", "deepseek-v4.1-flash")
HTTP_IMG = os.environ.get("HTTP_IMG", "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png")


def b64(name):
    ext = name.rsplit(".", 1)[1].replace("jpg", "jpeg")
    return "data:image/%s;base64,%s" % (ext, base64.b64encode(open(os.path.join(DIR, name), "rb").read()).decode())


def chat(content, max_tokens=80, stream=True):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0.0, "stream": stream, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": False}}
    r = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time(); first = None; text = ""; usage = None
    try:
        with urllib.request.urlopen(r, timeout=600) as resp:
            for line in resp:
                line = line.strip()
                if not line.startswith(b"data:") or line.endswith(b"[DONE]"):
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    usage = d["usage"]
                if "error" in d:
                    text += "[ERROR %s]" % d["error"]
                for c in d.get("choices", []):
                    s = c.get("delta", {}).get("content") or ""
                    if s and first is None:
                        first = time.time()
                    text += s
    except urllib.error.HTTPError as e:
        return {"error": "HTTP %d %s" % (e.code, e.read()[:300].decode(errors="replace")), "ttft": None, "text": ""}
    t1 = time.time()
    return {"ttft": round((first or t1) - t0, 3), "total": round(t1 - t0, 3), "text": text.strip(),
            "prompt_tokens": (usage or {}).get("prompt_tokens")}


def ok(text, want):
    t = text.lower().replace(",", "").replace("$", "")
    return all(any(w2.lower().replace(",", "") in t for w2 in (w if isinstance(w, list) else [w])) for w in want)


cases = json.load(open(os.path.join(DIR, "cases.json")))
res = []
chat([{"type": "text", "text": "hi"}], 4)  # warm
for name, q, want in cases:
    img = chat([{"type": "image_url", "image_url": {"url": b64(name)}}, {"type": "text", "text": q}])
    txt = chat([{"type": "text", "text": q}])
    r = {"case": name, "q": q, "want": want, "answer": img.get("text"), "ok": ok(img.get("text", ""), want),
         "ttft": img.get("ttft"), "ttft_text_only": txt.get("ttft"), "prompt_tokens": img.get("prompt_tokens"),
         "error": img.get("error")}
    r["ttft_added"] = round(r["ttft"] - r["ttft_text_only"], 3) if r["ttft"] is not None and r["ttft_text_only"] is not None else None
    print(json.dumps(r), flush=True)
    res.append(r)

# multi-image in one request
m = chat([{"type": "text", "text": "First image:"}, {"type": "image_url", "image_url": {"url": b64("01_ocr_sign.png")}},
          {"type": "text", "text": "Second image:"}, {"type": "image_url", "image_url": {"url": b64("02_ocr_receipt.png")}},
          {"type": "text", "text": "What gate is on the first image, and what is the TOTAL on the second? Answer as 'gate, total'."}])
r = {"case": "multi_image(01+02)", "answer": m.get("text"), "ok": ok(m.get("text", ""), ["47B", "10.50"]), "ttft": m.get("ttft"),
     "prompt_tokens": m.get("prompt_tokens"), "error": m.get("error")}
print(json.dumps(r), flush=True); res.append(r)

# image + long text context (~30K tokens of filler with a fact in the middle)
filler = " ".join("The committee reviewed item %d of the quarterly maintenance log and found nothing unusual." % i for i in range(1700))
mid = len(filler) // 2
doc = filler[:mid] + " IMPORTANT: the storage room access code is 6619. " + filler[mid:]
lt = chat([{"type": "text", "text": doc}, {"type": "image_url", "image_url": {"url": b64("03_count_red.png")}},
           {"type": "text", "text": "Two questions: (1) what is the storage room access code in the text above? (2) how many red circles are in the image? Answer as 'code, count'."}])
r = {"case": "long_text+image", "answer": lt.get("text"), "ok": ok(lt.get("text", ""), ["6619", "7"]), "ttft": lt.get("ttft"),
     "prompt_tokens": lt.get("prompt_tokens"), "error": lt.get("error")}
print(json.dumps(r), flush=True); res.append(r)

# http(s) URL image (fetched by the encoder)
h = chat([{"type": "image_url", "image_url": {"url": HTTP_IMG}}, {"type": "text", "text": "What objects are shown in this image? Answer in a few words."}])
r = {"case": "http_url", "url": HTTP_IMG, "answer": h.get("text"), "ok": ok(h.get("text", ""), [["dice", "die"]]), "ttft": h.get("ttft"),
     "prompt_tokens": h.get("prompt_tokens"), "error": h.get("error")}
print(json.dumps(r), flush=True); res.append(r)

# a bad image must be a 400, not a crash
bad = chat([{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}, {"type": "text", "text": "what?"}])
r = {"case": "bad_image_400", "answer": bad.get("error") or bad.get("text"), "ok": "400" in str(bad.get("error")) or "'code': 400" in str(bad.get("text")), "ttft": None}
print(json.dumps(r), flush=True); res.append(r)
# a text request still answers
t = chat([{"type": "text", "text": "What is 17*23? Just the number."}], 10)
r = {"case": "text_after_images", "answer": t.get("text"), "ok": "391" in t.get("text", ""), "ttft": t.get("ttft")}
print(json.dumps(r), flush=True); res.append(r)
json.dump(res, open(OUT, "w"), indent=1)
print("PASS %d/%d" % (sum(1 for x in res if x["ok"]), len(res)))
