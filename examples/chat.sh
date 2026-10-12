#!/usr/bin/env bash
# One chat request with curl. Usage: ./examples/chat.sh "your question"   (BASE_URL defaults to http://127.0.0.1:8012)
BASE_URL="${BASE_URL:-http://127.0.0.1:8012}"
Q="${1:-Write a haiku about HBM.}"
python3 -c 'import json,sys; print(json.dumps({"model": "GLM-5.3-Flash", "reasoning_effort": "low", "max_tokens": 400,
  "messages": [{"role": "user", "content": sys.argv[1]}]}))' "$Q" |
curl -s "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' -d @- |
python3 -c 'import json,sys; m=json.load(sys.stdin)["choices"][0]["message"]; print(m.get("content"))'
