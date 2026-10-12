#!/usr/bin/env python3
"""Ask a question about a local image (PNG, JPEG or WebP). Images go in as base64 data: URLs; the server does not fetch URLs.
Usage: python3 examples/image.py path/to/image.png "What does this screenshot show?"    (BASE_URL defaults to http://127.0.0.1:8012)"""
import base64, json, mimetypes, os, sys, urllib.request

base = os.environ.get("BASE_URL", "http://127.0.0.1:8012")
path = sys.argv[1]
question = sys.argv[2] if len(sys.argv) > 2 else "Describe this image."
mime = mimetypes.guess_type(path)[0] or "image/png"
data = base64.b64encode(open(path, "rb").read()).decode()
body = {"model": "GLM-5.3-Flash", "max_tokens": 600, "reasoning_effort": "low",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}]}]}
req = urllib.request.Request(f"{base}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as resp:
    out = json.load(resp)
print(out["choices"][0]["message"].get("content"))
print(f"[prompt tokens incl. image: {out.get('usage', {}).get('prompt_tokens')}]")
