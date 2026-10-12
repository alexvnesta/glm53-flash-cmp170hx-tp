# GLM-5.3-Flash on 2× CMP 170HX 64 GB (tensor parallel)

Serve **GLM-5.3-Flash** (320B total, ~18B active MoE) with every weight resident in HBM across two **NVIDIA CMP 170HX
64 GB** cards, using ExLlamaV3 with tensor parallelism, DFlash2 speculative decoding, hybrid drafting, a 393K-token context,
two concurrent sessions, prefix and RAM-tier caching, and image input. OpenAI-compatible API.

This builds on [Flun/glm53-flash-cmp170hx-exl3](https://github.com/Flun/glm53-flash-cmp170hx-exl3) (the original two-card
EXL3 server and recipe) and [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp. See [CREDITS.md](CREDITS.md).

> **Status (2026-10-12):** the documentation, measurements and scripts are here. The ExLlamaV3 patch set (`engine/`) and the
> server code (`service/`) are being checked byte-for-byte against the running setup and will be pushed shortly; until then
> the quick start below can't be completed. **Next:** an A/B of ExLlamaV3's NCCL tensor-parallel backend over GPU P2P
> (today the TP all-reduce goes through host memory), which may raise decode and prefill speed. Results will be added here.

## Performance (measured on one machine, 2026-10-12)

| measure | value |
|---|---|
| Decode, one request | ~115-125 tok/s structured text · ~40 tok/s hard reasoning · **134-147 tok/s** on code edits and JSON (hybrid drafting) |
| Prefill | **~1.32-1.36K tok/s**, flat from 8K to 225K tokens (a cold 225K prompt takes ~170 s) |
| Context | 392,960 tokens per request; two ~220K sessions fit in VRAM at once |
| Concurrency and caching | 2 active sessions · a cached 26.8K system/tool prefix: 0.7 s to first token (20.8 s cold) · parked 220K session resumes in 1-3 s |
| Vision | full-HD screenshot ~3.1-3.4 s to first token · 31/32 screenshot questions correct · decode speed unchanged with images |
| vs. Flun's layer-split layout, same machine | +28 % decode, prefill not slower |

Full tables, methods and the things we tried that did not help: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## What you need

- **Two CMP 170HX 64 GB cards** that CUDA already sees with their full 65,536 MiB each, on PCIe x16 links (ours run Gen2 x16).
  This repo does not cover getting the cards to that state.
- Linux, NVIDIA driver with CUDA 13.x support (we use driver 610.57.04, CUDA toolkit 13.3 with `nvcc`), Python **3.13**.
- **~128 GB host RAM** (the RAM tier pins 8 GiB, plus model loading) and **~150 GB disk** for weights.
- `git`, `ninja` (installed by the setup script), and `bwrap` (bubblewrap) if you want the strict image-decoding sandbox.

Our machine: Threadripper 3970X, 128 GB RAM, driver 610.57.04, CUDA 13.3, PyTorch 2.14.1+cu130, Triton 3.8.0.

## Quick start

```bash
git clone <this repo> glm53-flash-tp && cd glm53-flash-tp
./scripts/setup_venv.sh        # .venv with pinned PyTorch 2.14.1+cu130 and the service dependencies
./engine/apply_engine.sh       # clones ExLlamaV3 at the pinned commit, applies our patches, verifies every file hash
./engine/build_native.sh       # builds the sm_80 native extension into native/ (about 25 min)
./scripts/get_models.sh        # downloads the 3.05 bpw target and the DFlash2 drafter, converts the drafter to EXL3 6 bpw
./scripts/run_tp.sh            # serves the OpenAI-compatible API on 127.0.0.1:8012
```

The first start takes 2-3 minutes; the service also compiles a small fused kernel (`glm53_k101.so`) on its first run.
Check it with:

```bash
curl -s http://127.0.0.1:8012/health
curl -s http://127.0.0.1:8012/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"What is 2 plus 2?"}],"max_tokens":64}'
```

To run it as a service, see [systemd/glm53-flash.service.example](systemd/glm53-flash.service.example).

## What runs

| component | detail |
|---|---|
| Target model | [turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3), 3.05 bpw branch, revision `332ab457b709…`, ~116.6 GiB |
| Drafter | [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), revision `bf582e4eacc1…`, converted locally to EXL3 6 bpw (~0.96 GiB), 7 draft tokens |
| Engine | ExLlamaV3 `dev` at commit `3a8e2fb84c` plus the patches in [engine/patches](engine/patches) |
| Service | [service/](service): Flun's server, extended (below) |
| KV cache | Q8, a 655,360-token pool with a page-local `k_idx` ring |

**Features**
- **Tensor parallel (TP=2):** every layer split across both cards, with DFlash2 drafting running under TP.
- **Hybrid drafting** (`--hybrid-draft-tokens 15`): when the output copies earlier context (code edits, JSON, quotes), a
  suffix-match draft of up to 15 tokens replaces the 7-token DFlash2 draft. Verification is unchanged, so outputs are identical.
- **Two concurrent sessions**, a **shared prefix cache** across new sessions, and an **8 GiB RAM tier** for idle sessions.
- **Vision:** base64 PNG/JPEG/WebP images in user messages, up to 16 per request and 6,144 tokens per image. The vision
  tower is part of the checkpoint (6-bit) and runs on GPU 0.
- **API:** `/v1/chat/completions`, `/v1/completions`, tool calls (streaming and not), reasoning effort `low`/`high`/`max`
  (model aliases `GLM-5.3-Flash-low/-high/-max`), `/v1/models`, `/tokenize`, `/health`.

## Configuration

`scripts/run_tp.sh` takes environment variables:

| variable | default | meaning |
|---|---|---|
| `MODELS_DIR` | `./models` | where `get_models.sh` put the weights |
| `HOST`, `PORT` | `127.0.0.1`, `8012` | bind address. The API has no authentication; keep it on loopback or put an authenticating proxy in front |
| `VISION` | `1` | `0` disables image input (and frees the vision tower's VRAM on GPU 0) |
| `HYBRID_DRAFT_TOKENS` | `15` | `0` disables hybrid drafting. 8 was slower than 15 in our tests |
| `RAM_TIER_GIB` | `8` | pinned host RAM for parked sessions |
| `TP_BACKEND` | `native` | ExLlamaV3 TP backend. `native` reduces through host memory; `nccl` is untested here |

## Concurrency and memory

Two requests generate at once; a third waits in a queue for a free slot. Both active requests together must fit the
655,360-token KV pool (prompt plus output, in 256-token pages). For example, ~320K + ~320K or 392,960 + ~255K fit;
330K + 330K does not. If they don't fit, the second request waits until the first finishes; an active request is never
evicted. Total throughput with two streams is about the same as one stream (we measured 9-30 % lower); the gain is that
no one waits behind a long job. Idle sessions move to the RAM tier and come back in seconds when they're resumed.

## Vision: safety notes

- Only base64 `data:` URLs are accepted. The server never fetches `http(s)` or `file` URLs (no SSRF); clients that send
  remote URLs get a 400.
- Images are decoded in a separate process under memory and CPU limits, inside a bubblewrap sandbox (no network, read-only
  mounts) when `bwrap` is installed. Without `bwrap` the decoder still runs under the limits only.
- Images are accepted only in user messages; PNG, JPEG and WebP only; requests over 64 MiB of images are refused.
- Keep the tower on GPU 0 (`--vision-device 0`). With it on GPU 1 we saw one unexplained TP synchronization timeout.

## Known limits

- Tested on one machine with one user. Speculative decoding speed depends on the text: ~40 tok/s on hard reasoning,
  115-147 tok/s on structured output.
- Two long contexts fit together; two 393K requests do not.
- A large new prompt slows a session that is already decoding until its prefill finishes.
- Two identical requests that start at the same instant can rarely differ in early tokens (batched vs. single prefill).
- The 3.05 bpw quant's quality figures (~93 % top-1 agreement, KLD ~0.05 vs BF16, for the 3.0 bpw build) are turboderp's and
  Flun's measurements, not ours.

## Licences

Our code is MIT ([LICENSE](LICENSE)). The service is derived from Flun's MIT repo and the engine patches modify turboderp's
MIT ExLlamaV3; see [NOTICE.md](NOTICE.md). **No model weights are included.** The DFlash2 drafter is CC-BY-NC-ND-4.0
(non-commercial, no redistribution of derivatives): convert it yourself with `get_models.sh` and don't share the converted
weights.
