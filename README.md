# GLM-5.3-Flash on 2× CMP 170HX (tensor parallel + vision)

![cards](https://img.shields.io/badge/GPUs-2%C3%97%20CMP%20170HX%2064%20GB-555)
![engine](https://img.shields.io/badge/engine-ExLlamaV3%20%40%203a8e2fb-555)
![licence](https://img.shields.io/badge/licence-MIT-blue)
![status](https://img.shields.io/badge/status-early%2C%20YMMV-orange)

Run **GLM-5.3-Flash** (320B total, ~18B active MoE) entirely in HBM on **two CMP 170HX 64 GB** cards, with an
OpenAI-compatible API: tensor parallel across both cards, DFlash2 speculative decoding plus hybrid drafting, up to 393K
tokens of context, two agents at once, idle sessions parked in system RAM, and image input.

It builds on [Flun's GLM-5.3-Flash EXL3 setup](https://github.com/Flun/glm53-flash-cmp170hx-exl3) and turboderp's
[ExLlamaV3](https://github.com/turboderp-org/exllamav3); see [CREDITS.md](CREDITS.md).

> **Early release.** Everything here runs in production on one machine and passed soak tests, but it hasn't had much
> real-world use beyond that. Expect rough edges, and please open an issue with what you see.

## What you get

| | Flun's original (layer split) | this repo (tensor parallel) |
|---|---|---|
| Decode, structured text | 96 tok/s | **~120 tok/s** (134-147 on code edits and JSON) |
| Decode, hard reasoning | 34 tok/s | **~44 tok/s** |
| Prefill | ~1.26-1.32K tok/s | **~1.33K tok/s** (flat from 8K to 225K tokens) |
| Requests at once | 1 | **2** (more queue) |
| Context | 384K | **393K** per request; two ~320K sessions fit together |
| Idle sessions | re-read on return | **parked in system RAM**, resume in 1-3 s |
| Repeated system/tool prompt | previous request only (opt-in) | **cached across sessions** (26.8K prefix: 20.8 s → 0.7 s) |
| Images | no | **yes** (full-HD screenshot ~3.2 s to first token) |

Both columns measured on the same machine (2 cards on PCIe Gen2 x16). Details and methods: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## What you need

- **Two CMP 170HX that CUDA sees with the full 64 GiB** (65,536 MiB) each, on **x16 links**. This repo doesn't cover getting the
  cards to that state.
- Linux, an NVIDIA driver with CUDA 13 support, the CUDA 13.x toolkit with `nvcc`, Python **3.13**.
- **~128 GB system RAM** and **~150 GB disk** for the weights.
- Optional: `bwrap` (bubblewrap) for the strict image-decoding sandbox.

Tested on: Threadripper 3970X, 128 GB RAM, driver 610.57.04, CUDA 13.3, PyTorch 2.14.1+cu130, Triton 3.8.0.

## Quick start

```bash
git clone https://github.com/alexvnesta/glm53-flash-cmp170hx-tp.git && cd glm53-flash-cmp170hx-tp
./scripts/setup_venv.sh     # Python venv with pinned PyTorch and dependencies
./engine/apply_engine.sh    # fetches ExLlamaV3 at the pinned commit, applies the patches, checks every file
./engine/build_native.sh    # builds the CUDA extension for the CMP's sm_80 (~25 min)
./scripts/get_models.sh     # downloads the model (~125 GB) and the drafter, converts the drafter (~tens of minutes)
./scripts/run_tp.sh         # starts the API on 127.0.0.1:8012 (first start takes 2-3 min)
```

Then try it:

```bash
./examples/chat.sh "Write a haiku about HBM."
python3 examples/image.py screenshot.png "What's on this screen?"
```

To keep it running, see [systemd/glm53-flash.service.example](systemd/glm53-flash.service.example).

## Using the API

Any OpenAI-compatible client works with base URL `http://127.0.0.1:8012/v1` and model **`GLM-5.3-Flash`**.

- **Reasoning effort:** send `reasoning_effort` (`low`, `high`, `max`), or use the model names `GLM-5.3-Flash-low`,
  `-high`, `-max`. The default is `high`. `max` is much slower on everyday prompts but more careful on hard lookups.
- **Tools:** standard OpenAI `tools` / `tool_calls`, streaming or not.
- **Images:** content parts with `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` in user messages.
  PNG, JPEG or WebP; up to 16 images per request. Remote image URLs are refused, so clients must send base64.
- **Endpoints:** `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/tokenize`, `/health`.

Ready-made clients: [examples/](examples) (curl chat, streaming, tool call round trip, image question).

## Settings

`scripts/run_tp.sh` reads these environment variables:

| variable | default | what it does |
|---|---|---|
| `PORT`, `HOST` | `8012`, `127.0.0.1` | where the API listens. There is **no authentication**: keep it on localhost or put a proxy with auth in front |
| `VISION` | `1` | `0` turns image input off and frees the vision tower's memory on GPU 0 |
| `HYBRID_DRAFT_TOKENS` | `15` | `0` turns hybrid drafting off (15 tested faster than 8) |
| `RAM_TIER_GIB` | `8` | system RAM for parked sessions |
| `MODELS_DIR` | `./models` | where the weights live |
| `TP_BACKEND` | `native` | tensor-parallel backend; `nccl` is experimental (see Roadmap) |

## How two agents share the cards

Two requests generate at the same time; a third waits for a free slot. The two active requests share a 655,360-token cache, so
~320K + ~320K or 393K + ~255K fit, while 330K + 330K does not; if they don't fit, the second waits until the first finishes.
Running requests are never evicted. Two streams together are about as fast as one in total (9-30 % lower), so the win is that
a short subagent doesn't wait behind a long main thread. Sessions that go idle move to system RAM and come back in seconds.

## Troubleshooting

- **`apply_engine.sh` reports a hash mismatch:** delete `engine-src/` and run it again. The server checks engine files at
  startup and refuses to start vision or concurrency on a modified engine.
- **"missing native/exllamav3_ext…":** run `./engine/build_native.sh` with the venv's Python 3.13 and CUDA 13.x `nvcc` on `PATH`.
- **Out of GPU memory at startup or with big images:** close anything else on the GPUs. `VISION=0` frees the vision tower's ~0.4 GB on GPU 0.
- **First start is slow:** it compiles a small fused kernel once; every start also loads ~117 GiB of weights and captures CUDA graphs (2-3 minutes).
- **"bwrap not found":** images still work, decoded under memory and CPU limits only. Install `bubblewrap` for the full sandbox.
- **Keep the vision tower on GPU 0.** With it on GPU 1 we saw one unexplained tensor-parallel timeout.

## Roadmap

- **NCCL over GPU P2P.** The driver reports P2P between the cards, but today's tensor-parallel reductions go through system
  memory. An A/B of ExLlamaV3's NCCL backend is next; it may raise decode and prefill speed.
- **Quality evals** (HumanEval, GSM8K) on this build, so quality can be compared with other setups.
- **A one-command start** that runs all five setup steps.

## Other setups

- **Two cards, simpler:** [Flun/glm53-flash-cmp170hx-exl3](https://github.com/Flun/glm53-flash-cmp170hx-exl3), the layer-split
  original this repo extends.
- **Four cards:** [Morrowmake/glm53-flash-cmp170hx-recipe](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe) runs a vLLM
  fork with 4-bit weights on 4× CMP 170HX with P2P (482 tok/s single-user structured, by their measurement). Its weights don't
  fit on two cards.

## What runs, exactly

| component | detail |
|---|---|
| Model | [turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3), 3.05 bpw (rev `332ab457`), ~116.6 GiB |
| Drafter | [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (rev `bf582e4e`), converted locally to EXL3 6 bpw, 7 draft tokens |
| Engine | ExLlamaV3 at `3a8e2fb84c` + [9 patches](engine/NOTES.md): TP DFlash2, a compact KV layout (`k_idx` ring) that fits 655K tokens, graph/BLAS fixes, fair scheduling, hybrid drafting hook |
| Server | [service/](service): Flun's server plus TP serving, two-session scheduler, prefix cache, RAM tier, vision, hybrid drafting |
| KV cache | Q8, 655,360-token pool |

| path | contents |
|---|---|
| `engine/` | base commit, patches, apply/verify/build scripts, notes |
| `service/` | the server (`glm/`), drafter converter (`dflash2/`), fused kernels, CPU tests |
| `scripts/` | setup, model download, start, GPU wait |
| `examples/` | API clients |
| `docs/PERFORMANCE.md` | all measurements and the ideas that didn't help |

## Known limits

- Speed depends on the text: ~40 tok/s on hard reasoning, 115-147 tok/s on structured output.
- A large new prompt slows a session that's already generating until its prefill finishes.
- Two identical requests that start at the exact same instant can, rarely, differ in early tokens.
- For the layer-split profile, use Flun's repo; this one targets tensor parallel.
- Quantization quality figures (~93 % top-1 agreement, KLD ~0.05 vs BF16 for 3.0 bpw) are turboderp's and Flun's, not ours.

## Licences

MIT ([LICENSE](LICENSE)). The server is derived from Flun's MIT repo and the engine patches modify turboderp's MIT ExLlamaV3
([NOTICE.md](NOTICE.md)). **No model weights are included.** The DFlash2 drafter is CC-BY-NC-ND-4.0: `get_models.sh` converts
it on your machine; please don't share converted copies.
