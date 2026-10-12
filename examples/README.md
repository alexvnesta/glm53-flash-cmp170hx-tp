# Examples

Standard-library clients for the server started by `scripts/run_tp.sh`. Set `BASE_URL` if it isn't on `http://127.0.0.1:8012`.

| file | what it does |
|---|---|
| `chat.sh` | one request with curl |
| `stream.py` | streams a reply and prints the decode time |
| `tool_call.py` | a tool call round trip (the "tool" is a stand-in) |
| `image.py` | asks about a local PNG/JPEG/WebP image, sent as base64 |

Any OpenAI-compatible client also works: base URL `http://127.0.0.1:8012/v1`, model `GLM-5.3-Flash` (or `GLM-5.3-Flash-low`,
`-high`, `-max` to pick the reasoning effort), any API key.
