> Historical measurements from earlier profiles. The native-SWA rebase measurements and qualification limits are in [the October 13 qualification](QUALIFICATION_2026-10-13.md); use the root README for the current recipe.

# Performance: GLM-5.3-Flash on 2× CMP 170HX 64 GB

All numbers were measured on one machine on 2026-10-12, mostly at temperature 0, unless marked *(reported)*. Speculative-decoding speed depends on how predictable the text is.

How things were measured: decode with fixed counting and free-text prompts (n=10 runs per arm for A/B gates); prefill with unique cold prompts sized by the server's tokenizer (n=3 per size, server-side prompt time); hybrid drafting with 7 workloads × 3 passes at temperature 0, comparing output hashes; soaks with a mixed two-worker client (shared-prefix sessions with ground-truth answers, chats, counting, cancels, tool calls).

## 1. The machine (observed 2026-10-12)
| item | value |
|---|---|
| GPUs | 2× NVIDIA CMP 170HX, 65,536 MiB HBM2e each (SK hynix), SM80, max SM clock 1,695 MHz, power limit 180 W |
| Links | PCIe **Gen2 x16** per card; measured 6.8 GB/s host→device and 6.7 GB/s device→host per card, both cards at once |
| GPU peer-to-peer | The driver reports P2P **read/write OK** between the two cards (`nvidia-smi topo -p2p r/w`; atomics not supported). **Our tensor parallelism does not use it:** ExLlamaV3's native TP backend does each all-reduce through pinned host memory with a CPU helper process. NCCL/P2P has not been tried on this box |
| Host | AMD Threadripper 3970X (32 cores), 128 GB RAM, Ubuntu, NVIDIA driver 610.57.04 |
| Software | CUDA 13.3, PyTorch 2.14.1+cu130, Triton 3.8.0 |

## 2. Measurements

### Decode, one request (tok/s)
| workload | tok/s |
|---|---|
| easy/structured text (counting) | 115-125 |
| hard free-form reasoning text | ~40 |
| at ~220K tokens of context (counting-style) | 111-116 |
| at ~220K tokens of context (essay) | ~37 |
| with hybrid drafting: code edit / refactor | 134 / 135 (+17 % each) |
| with hybrid drafting: JSON text / recitation | 147 / 89 (+20 % / +15 %) |
| with hybrid drafting: prose / counting controls | 38 / 125 (unchanged, +3 % within noise) |

Hybrid drafting changed no output: temperature-0 outputs were byte-identical with it on and off on every workload.

### Prefill (prompt processing)
| prompt | server-side tok/s | time to first token |
|---|---|---|
| 8K tokens | 1,316 | ~6 s |
| 32K | 1,343 | ~25 s |
| 96K | 1,356 | ~73 s |
| 225K | 1,325 | ~170 s |
| 330K | 1,267 | 261 s |

### Caching, resume, concurrency
| scenario | result |
|---|---|
| New session on a cached 26.8K-token system/tool prefix | 0.7 s to first token (20.8 s cold) |
| Resume a parked 220K-token session from RAM | 1-3 s (cold re-prefill ~168 s) |
| Resume a parked 330K session | 1.3 s |
| Two sessions decoding at once | each gets ~40-45 % of its solo speed; total ≈ 9 % *below* serial. It's for latency/fairness, not throughput |

### Vision (production config)
| measure | result |
|---|---|
| Time to first token, new image | 448 px 0.7 s · full-HD screenshot 3.1-3.4 s · retina screenshot 6.8-6.9 s |
| Same image re-sent | 0.2-0.4 s (cached) |
| Decode speed with images in context | 117-118 tok/s vs 113-115 for the same length of text (image embeddings are dropped after prefill) |
| Accuracy | 31/32 computer-use screenshot questions at the 6,144-token cap; 6/6 known-answer images (OCR, invoice fields, colours, shapes, a chart value, text in a 4K photo) |
| Soak | 45 min: 646 requests with 1,427 images, 0 failures |

### Stability and quality
| check | result |
|---|---|
| 30-min mixed soak with hybrid + vision live | 912 requests, 0 failures; retrieval accuracy 94.4 % (94.1 % / 93.8 % in the two soaks before hybrid) |
| VRAM (of 65,536 MiB per card) | ~62,900 / 62,300 MiB idle; peaks ~64,400 / 63,000 MiB with vision under load |
| Startup | ~2-3 min to healthy |
| GPU consistency check | 0 bad memory words on either card; repeated fp16/bf16 GEMMs bit-identical, no zero rows |

### Compared with the original layer-split setup (Flun's, same machine)
| | Flun's layer split | our TP build |
|---|---|---|
| decode, easy / hard text | 96.2 / 34.2 tok/s | 123.6 / 44.3 tok/s (**+28 %**) |
| prefill | 0.95-0.98× of ours | 1.32-1.36K tok/s |
| concurrency | one request at a time | two |

Flun reports *(reported)* ~90-95 tok/s decode and 1.5-1.65K tok/s prefill on his own box (PCIe Gen2 x8); on ours his
layout prefills at ~1.26-1.32K.

## 3. Tried and dropped
| idea | result |
|---|---|
| `k_hcfuse` fused hyper-connection kernels under TP | correct (0 mismatches over 12k checked sites per GPU) but decode **1.5-4.2 % slower** → off |
| Hybrid drafting with H=8 | slower than H=15 on edits (0.83-0.86×) → use 15 |
| Layer split for faster prefill | not faster here (2-5 % slower than TP) |
| ExLlamaV3 sparse-attention fix (#494 / `74d48b7a10`) | not needed: the bug only triggers without CUDA graphs |
| vLLM routes | stock vLLM's GLM-5.3 sparse attention needs SM90+; the SM80 vLLM fork that runs Flash needs 4 cards (4-bit weights ~178 GiB) |
| More concurrency for throughput | aggregate throughput does not rise on these cards |

## 4. Limits
- Single machine, single user; numbers are medians or ranges from our test logs, mostly temperature 0. Speculative decoding
  speed depends on how predictable the text is: ~40 tok/s on hard reasoning, 115-147 tok/s on structured output.
- Two long contexts can be resident together, but two 393K requests cannot.
- A large new prompt slows a session already decoding (prefill shares the GPUs) to about a third of its speed until it finishes.
- When two identical requests start at the exact same moment, rare early-token flips occur (batched vs. single prefill),
  with or without images. Outputs stay coherent.
- The 3.05 bpw quant's quality figures are turboderp's/Flun's *(reported)*: ~93 % top-1 agreement and KLD ~0.05 vs BF16
  for the 3.0 bpw build. We have not run our own fidelity evaluation.

## 5. NCCL vs native tensor-parallel backend (2026-10-12)

Same build (vision and hybrid drafting on), same machine, consecutive arms on a test port. Native = ExLlamaV3's default TP backend
(all-reduce through pinned host memory with a CPU helper). NCCL = `-tpb nccl`, NCCL 2.30.7, bootstrap on loopback.

**Content check first** (our `scripts/p2p_check.py`; idea from Morrowmake's recipe): 74 random-byte peer copies (both directions,
128 KiB ±1 to 32 MiB) with 0 mismatches; 84 NCCL all_reduce cases (fp32 and bf16) bit-exact against a CPU sum; NCCL chose the
**P2P/CUMEM** transport.

| metric | native | NCCL | change |
|---|---|---|---|
| decode, counting (serial n=10) | 119.1 tok/s | 124.3 | +4.35 % (Welch p=3e-9) |
| decode, hard free text (serial n=10) | 42.75 | 45.52 | +6.48 % (p=2e-9) |
| prefill 8K / 32K (cold, n=2 each) | 1,360 / 1,385 tok/s | 1,477 / 1,545 | +8.6 % / +11.5 % (pooled +10.05 %, p=1.5e-4) |
| structured / code / prose prompts (temp 0, 400 tokens, thinking low; n=5) | 89.0 / 101.5 / 43.1 | 94.3 / 105.5 / 46.5 | +6.0 / +3.9 / +7.9 % |
| GPU0 headroom at an image-heavy stress peak | 1,037 MiB | 1,099 MiB | no extra memory |

Sanity answers and the identical-pair concurrency test passed on both arms; temperature-0 output hashes differ between backends
(different reduction order), as expected. These prompt numbers use the lowest reasoning effort; the server cannot switch thinking off,
so they are not directly comparable with "thinking off" benchmarks.

After switching production to NCCL, a 30-minute mixed soak (two workers: shared-prefix sessions with ground-truth answers, chats, counting, cancels, tool calls) passed: 945 requests, 0 failures besides 3 known wrong retrieval answers (97.4 % retrieval accuracy), a clean log (no NCCL warnings, no synchronization timeouts) and a healthy server throughout. The same soak on the native backend served 912 requests.
