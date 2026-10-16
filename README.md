# GLM-5.3-Flash on 2× CMP 170HX (tensor parallel + vision)

![cards](https://img.shields.io/badge/GPUs-2%C3%97%20CMP%20170HX%2064%20GB-555)
![engine](https://img.shields.io/badge/engine-ExLlamaV3%20%40%200f685aa-555)
![licence](https://img.shields.io/badge/licence-MIT-blue)
![status](https://img.shields.io/badge/status-early%2C%20YMMV-orange)

Run **GLM-5.3-Flash** (320B total, ~18B active MoE) entirely in HBM on **two CMP 170HX 64 GB** cards, with an
OpenAI-compatible API: tensor parallel across both cards (NCCL over GPU P2P), DFlash2 speculative decoding plus hybrid drafting, up to 393K
tokens of context, two agents at once, idle sessions parked in system RAM, and image input.

It builds on [Flun's GLM-5.3-Flash EXL3 setup](https://github.com/Flun/glm53-flash-cmp170hx-exl3) and turboderp's
[ExLlamaV3](https://github.com/turboderp-org/exllamav3); see [CREDITS.md](CREDITS.md).

> **Early release.** Everything here runs in production on one machine and passed soak tests, but it hasn't had much
> real-world use beyond that. Expect rough edges, and please open an issue with what you see.

The [optional Weightless GLP-44 profile](profiles/weightless/README.md) adds a
fused post-layer projector and native prefill-stall hardening. It is enabled on
the local host at strength 2 across layers 1–44; see the
[October 14 validation](docs/WEIGHTLESS_QUALIFICATION_2026-10-14.md) and
[October 15 cache/FP16 qualification](docs/CACHE_QUALIFICATION_2026-10-15.md). The regular
launcher remains unsteered, and the performance tables below describe the
earlier unsteered P16 measurements.

The [latest sustained-traffic measurements](docs/SUSTAINED_CONCURRENCY_2026-10-16.md)
cover the local 32-row variant: high-effort aggregate delivery rose from
111.1 tok/s with one agent to 125.4 with two, delivering 65.8 and 59.6 tok/s
per agent. It improves concurrent throughput by 1.8–3.8% over the qualified
16-row variant. The report separates effort, elapsed time, cache hits and
actual per-agent rates; four submitting agents still share two active slots.

The [MoE and communication follow-up](docs/MOE_COMMUNICATION_2026-10-16.md)
retains that production profile. Global MoE overrides failed the screening,
and moving broadcasts above 2 KiB to NCCL regressed low-effort concurrent
delivery by 11.5–12.9%, with no consistent high-effort gain. The report includes
matched aggregate/per-agent token rates, elapsed time and all 384 scored tasks.

The [routed-MoE investigation](docs/ROUTED_MOE_2026-10-16.md) identifies the
3-bit gate/up kernel as the largest individual GPU cost. A staged decoder
produced no consistent serving gain; deeper prefetch crossed a register
occupancy threshold and slowed kernel timings. Production retains row32.
The report includes matched aggregate/per-agent tok/s and hardware counters.

The [instruction-decoder follow-up](docs/DECODE_INSTRUCTIONS_2026-10-16.md)
tests procedural byte sums and a fixed-codebook lookup while preserving the
64-register block-residency limit. It includes the unchanged serving benchmark,
actual per-agent rates, exhaustive codebook checks and cache/occupancy evidence.

The [warp-layout follow-up](docs/MOE_RETILE_2026-10-16.md) tests a different
SM80 gate/up layout. It reduces registers from 64 to 60 while preserving
occupancy, but regresses serving throughput. The report retains the same
benchmark, actual per-agent rates and CPU-contention flags; the qualified
row32 layout remains the default.

## What you get

| Feature or bounded probe | Native-SWA profile |
|---|---|
| Counting decode | **129.38 tok/s**; +2.30% versus the installed stage-1 rebase |
| LRU reasoning decode | **46.77 tok/s**; +0.60% versus stage 1 |
| Short structured/code/prose probes | **+3.7–5.1%** versus stage 1 |
| Cold prefill | **1.49–1.53K tok/s**, two requests each at 8K/32K/96K |
| Two-stream aggregate | **67.25 tok/s**, +7.01% versus stage 1 |
| Requests at once | **2 active**; more queue |
| Context | **392,960** tokens including output; shared pool 655,360 |
| Four idle 220K sessions | Cached restores matched cold outputs; first token in **3.8–4.0s** |
| Images | GPU0 vision tower, bubblewrap decode, base64 input |

These October 13 measurements used the same machine (PCIe Gen2 x16) and NCCL
P2P transport in both arms. They are workload-specific, not general task-quality
claims. Two near-limit 327K requests ran sequentially in the pressure probe.
Max-effort retrieval was 56/56; mixed-traffic soaks retain known model mistakes.
Methods, correctness limits and fresh-build checks:
[qualification](docs/QUALIFICATION_2026-10-13.md).
Earlier layer-split, hybrid and NCCL comparisons remain in
[historical performance](docs/PERFORMANCE.md), with credit to Flun's original.

## What you need

- **Two CMP 170HX that CUDA sees with the full 64 GiB** (65,536 MiB) each, on **x16 links**. This repo doesn't cover getting the
  cards to that state.
- Linux, an NVIDIA driver with CUDA 13 support, the CUDA 13.x toolkit with `nvcc`, Python **3.13**.
- **~128 GB system RAM** and **~150 GB disk** for the weights.
- `bwrap` (bubblewrap) for image input. Use `VISION=0` for text-only serving without it.

Tested on: Threadripper 3970X, 128 GB RAM, driver 610.57.04, CUDA 13.3, PyTorch 2.14.1+cu130, Triton 3.8.0.

## Quick start

```bash
git clone https://github.com/alexvnesta/glm53-flash-cmp170hx-tp.git && cd glm53-flash-cmp170hx-tp
./scripts/setup_venv.sh     # Python venv with pinned PyTorch and dependencies
./engine/apply_engine.sh    # fetches ExLlamaV3 at the pinned commit, applies the patches, checks every file
./engine/build_native.sh    # builds the CUDA extension for the CMP's sm_80 (~15–25 min)
./scripts/get_models.sh     # downloads the model (~125 GB) and the drafter, converts the drafter (~tens of minutes)
.venv/bin/python scripts/p2p_check.py   # optional: checks GPU P2P before using the default NCCL backend
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
- **JSON objects:** opt in at startup with `--experimental-json-mode` and request
  `response_format: {"type":"json_object"}`. See the
  [contract and qualification](docs/JSON_ROW16_QUALIFICATION_2026-10-16.md).
- **Images:** content parts with `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` in user messages.
  PNG, JPEG or WebP; up to 16 images per request. Remote image URLs are refused, so clients must send base64.
- **Endpoints:** `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/tokenize`, `/health`.

Ready-made clients: [examples/](examples) (curl chat, streaming, tool call round trip, image question).

## Deployed GLM versus Qwen

The dated [comparison](docs/GLM_QWEN_COMPARISON_2026-10-14.md) records throughput,
cold-prefill latency and the 92-scenario tool suite. GLM scored 90/100 and Qwen
85/100 at backend defaults and explicit maximum effort; all four runs failed
the benchmark safety gate. Qwen maximum is its supported `xhigh`, since its
template rejects literal `max`. Rate units,
limited cells and effort application are explicit in the report.

## Settings

`scripts/run_tp.sh` reads these environment variables:

| variable | default | what it does |
|---|---|---|
| `PORT`, `HOST` | `8012`, `127.0.0.1` | where the API listens. There is **no authentication**: keep it on localhost or put a proxy with auth in front |
| `VISION` | `1` | `0` turns image input off and frees the vision tower's memory on GPU 0 |
| `HYBRID_DRAFT_TOKENS` | `15` | `0` turns hybrid drafting off (15 tested faster than 8) |
| `RAM_TIER_GIB` | `8` | system RAM for parked sessions; persistence uses upstream's fixed pinned arenas within this ceiling |
| `RECURRENT_STATE_DTYPE` | `fp32` | target recurrent-state storage; the Weightless launcher defaults to `fp16` |
| `GLM_PERSISTENT_CACHE_DIR` | unset | enables upstream disk prefix/checkpoint caching behind the B2 RAM tier |
| `GLM_PERSISTENT_CACHE_GIB` | `50` | disk cache ceiling across profiles sharing that root; keeps 16 GiB free by default |
| `MODELS_DIR` | `./models` | where the weights live |
| `TRITON_CACHE_DIR` | Triton default | optional cache directory; measurements used pinned host FLA choices, described in [engine notes](engine/NOTES.md) |
| `TP_BACKEND` | `nccl` | tensor-parallel backend: `nccl` reduces GPU-to-GPU over P2P (+4-6 % decode, +10 % prefill here); `native` goes through system RAM. Run `scripts/p2p_check.py` first; if it fails, use `native` |

## How two agents share the cards

Two requests generate at the same time; a third waits for a free slot. The two active requests share a 655,360-token cache, so
~320K + ~320K or 393K + ~255K fit, while 330K + 330K does not; if they don't fit, the second waits until the first finishes.
Running requests are never evicted. Throughput depends on the workload: the qualified paired probe measured 67.25 tokens/s
in aggregate. The scheduler lets a short request make progress alongside a long request. Sessions that go idle move to
system RAM; the qualified long-session restores took 3.8–4.0 seconds.

## Troubleshooting

- **`apply_engine.sh` reports a hash mismatch:** delete `engine-src/` and run it again. The server checks engine files at
  startup and refuses to start vision or concurrency on a modified engine.
- **"missing native/exllamav3_ext…":** run `./engine/build_native.sh` with the venv's Python 3.13 and CUDA 13.x `nvcc` on `PATH`.
- **Out of GPU memory at startup or with big images:** close anything else on the GPUs. `VISION=0` frees the vision tower's ~0.4 GB on GPU 0.
- **First start is slow:** it loads ~117 GiB of weights, compiles any missing Triton kernels and captures CUDA graphs (2-3 minutes).
- **"bwrap not found":** install `bubblewrap` or set `VISION=0`. Image serving refuses to start without the sandbox.
- **Keep the vision tower on GPU 0.** With it on GPU 1 we saw one unexplained tensor-parallel timeout.
- **NCCL problems** (hangs at startup, `NCCL WARN` in the log, or `p2p_check.py` FAILED): start with `TP_BACKEND=native`. NCCL needs working
  GPU peer-to-peer between the cards; `nvidia-smi topo -p2p r` should show `OK`.

## Roadmap

- **Concurrency:** the [sustained-traffic follow-up](docs/SUSTAINED_CONCURRENCY_2026-10-16.md)
  profiles mixed request backfill, expands fused decode to 32 rows and reports
  aggregate/per-agent tok/s at one, two and four submitting agents. The
  [earlier JSON qualification](docs/JSON_ROW16_QUALIFICATION_2026-10-16.md)
  records the sixteen-row deployment, isolated-pair speeds and JSON contract.
  Runtime soak, restart recovery, failed semantic cells and max-effort token
  limits remain explicit. The original
  [investigation](docs/EXLLAMA_CONCURRENCY_2026-10-16.md) retains the prompt-only
  formatting failures. The general launcher remains unchanged.

- **Engine:** pins upstream dev `0f685aa9fd` with selected upstream recurrent-state/recovery backports and the original four P16 patches. See [engine/NOTES.md](engine/NOTES.md) for source reproduction and [cache/metrics notes](docs/CACHE_AND_OBSERVABILITY.md) for the optional persistent profile.

- **Done (2026-10-12): NCCL over GPU P2P.** A content check (peer copies and NCCL all-reduce against a CPU reference) passed on our
  cards, NCCL picked the P2P/CUMEM transport, and an A/B on this build measured +4.4 / +6.5 % decode and +10 % prefill with no memory cost.
  It is the default now.
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
| Engine | ExLlamaV3 at `0f685aa9fd` + [9 patches](engine/NOTES.md): the four P16 patches, selected upstream FP16/recovery/logit checks, and persistent-cache profile identity. TP DFlash2, hybrid drafting and recurrent paired states use upstream implementations |
| Server | [service/](service): Flun's server plus TP serving, two-session scheduler, prefix cache, RAM tier, vision, hybrid drafting |
| TP backend | NCCL 2.30.7 over GPU P2P (transport P2P/CUMEM); `native` (host-RAM all-reduce) as fallback |
| Target KV cache | Q8, 655,360-token shared pool |
| Draft state | Native FP16 sliding-window rings; 2,560 rows per active slot |
| Paired checkpoints | Upstream recurrent paired-state cache, 5 GiB budget |

| path | contents |
|---|---|
| `engine/` | base commit, patches, apply/verify/build scripts, notes |
| `service/` | the server (`glm/`), drafter converter (`dflash2/`), CPU tests |
| `scripts/` | setup, model download, start, GPU wait, P2P/NCCL check |
| `examples/` | API clients |
| `docs/PERFORMANCE.md` | all measurements and the ideas that didn't help |

## Known limits

- Speed depends on the workload and draft acceptance. The bounded counting and reasoning probes above do not predict every task.
- A large new prompt slows a session that's already generating until its prefill finishes.
- Two identical requests that start at the exact same instant can, rarely, differ in early tokens.
- For the layer-split profile, use Flun's repo; this one targets tensor parallel.
- Quantization quality figures (~93 % top-1 agreement, KLD ~0.05 vs BF16 for 3.0 bpw) are turboderp's and Flun's, not ours.

## Licences

MIT ([LICENSE](LICENSE)). The server is derived from Flun's MIT repo and the engine patches modify turboderp's MIT ExLlamaV3
([NOTICE.md](NOTICE.md)). **No model weights are included.** The DFlash2 drafter is CC-BY-NC-ND-4.0: `get_models.sh` converts
it on your machine; please don't share converted copies.
