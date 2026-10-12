#!/usr/bin/env python3
"""Stream a chat completion and print tokens as they arrive, then the decode speed. Standard library only.
Usage: python3 examples/stream.py "your question"    (BASE_URL defaults to http://127.0.0.1:8012)"""
import json, os, sys, time, urllib.request

base = os.environ.get("BASE_URL", "http://127.0.0.1:8012")
question = sys.argv[1] if len(sys.argv) > 1 else "Explain tensor parallelism in three sentences."
body = {"model": "GLM-5.3-Flash", "stream": True, "max_tokens": 600, "reasoning_effort": "low",
        "messages": [{"role": "user", "content": question}]}
req = urllib.request.Request(f"{base}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
first = None; chunks = 0
with urllib.request.urlopen(req, timeout=600) as resp:
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        delta = json.loads(line[5:])["choices"][0].get("delta", {})
        text = delta.get("content") or ""
        if text or delta.get("reasoning_content"):
            first = first or time.time(); chunks += 1
        print(text, end="", flush=True)
if first:
    print(f"\n\n[{chunks} chunks in {time.time() - first:.1f} s after the first token]")
