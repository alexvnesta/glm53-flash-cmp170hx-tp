#!/usr/bin/env python3
"""Ask a question that needs a tool, run the tool locally, send the result back, print the final answer.
Usage: python3 examples/tool_call.py    (BASE_URL defaults to http://127.0.0.1:8012)"""
import json, os, urllib.request

base = os.environ.get("BASE_URL", "http://127.0.0.1:8012")
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]

def chat(messages):
    body = {"model": "GLM-5.3-Flash", "messages": messages, "tools": tools, "max_tokens": 800, "reasoning_effort": "low"}
    req = urllib.request.Request(f"{base}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)["choices"][0]["message"]

messages = [{"role": "user", "content": "Is it warm enough for shorts in Lisbon right now?"}]
msg = chat(messages)
for call in msg.get("tool_calls") or []:
    args = json.loads(call["function"]["arguments"])
    print("model called", call["function"]["name"], args)
    result = {"city": args.get("city"), "temperature_c": 24, "conditions": "sunny"}   # stand-in for a real weather API
    messages += [msg, {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)}]
    msg = chat(messages)
print(msg.get("content"))
