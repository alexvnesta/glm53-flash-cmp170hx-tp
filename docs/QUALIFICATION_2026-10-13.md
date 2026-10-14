# Native SWA qualification, October 13, 2026

The pinned ExLlamaV3 source and native-SWA service passed a full matched
baseline/candidate GPU window on the documented two-CMP machine. The native
profile was installed with all live checks passing. The production post-install soak and fresh public-build GPU check also passed
on October 13–14 UTC.

The engine reproduces all 968 files at upstream commit
`0f685aa9fda3a98a864149b367b7ef570257572e` plus four patches. The service runtime
matches the GPU-qualified candidate, apart from trailing EOF whitespace in one
file. The public launcher replaces machine-specific paths with environment
variables and computes the actual native-extension hash at startup.

| Profile item | Qualified value |
|---|---|
| Target cache | Q8, 655,360 shared tokens, 2,560 pages |
| Request context | 392,960 tokens including output |
| Concurrent requests | Two active; additional requests queue |
| Draft state | Upstream FP16 SWA, 2,560 rows per active slot |
| Paired/recurrent budget | 5 GiB |
| Target CPU tier | 8 GiB |
| Drafting | DFlash2 K7 and hybrid15 |
| Tensor parallel | NCCL over GPU P2P/CUMEM |
| Vision | GPU0, bubblewrap decode, up to 6,144 tokens/image and 16 images |

## GPU window

The installed stage-1 rebase served as baseline. The candidate and two
hybrid-off controls used the same pinned source and host FLA winners. Every
arm passed its gates. The final recommendation was GO, followed by verified
baseline restoration before installation. Seven solo hybrid-on/off workloads
had identical outputs. This is bounded evidence, not universal continuation
or floating-point equivalence.

| Workload | Stage 1 baseline | Native stage 2 | Observed difference |
|---|---:|---:|---:|
| Counting decode, tok/s | 126.47 | 129.38 | +2.30% |
| LRU reasoning decode, tok/s | 46.49 | 46.77 | +0.60% |
| Short structured text, tok/s | 92.05 | 96.71 | +5.06% |
| Short code, tok/s | 103.78 | 107.67 | +3.74% |
| Short prose, tok/s | 45.59 | 47.83 | +4.92% |
| 220K-depth decode, chunks/s | 120.11 | 121.04 | +0.78% |
| Two-stream aggregate, tok/s | 62.84 | 67.25 | +7.01% |

The paired aggregate uses finalized completion-token usage divided by the
span from the first POST to the last response completion; it includes request
latency. The 220K depth rate counts stream chunks. Neither is interchangeable
with every partial-stream rate in the separate deployed-system comparison.

Ten measured runs per arm for counting/LRU; five for short output probes;
four cached depth runs; six paired aggregate samples. No metric was at least
3% slower with Welch p < 0.01. Cold prefill, two runs at each size, measured
1,512.5/1,532.7/1,489.0 tok/s at 8K/32K/96K. Pooled change +0.82%, p=0.453;
this is not a statistically established prefill gain.

- Max-effort retrieval: 40/40 main and 16/16 borderline, no missing/truncated answers.
- Four 220K sessions: every cached restore matched its cold output; 219,904
  cached tokens returned in 3.787–4.003 seconds; no CPU-tier pinning/ceiling faults.
- Near-limit page budget: two 327K requests completed sequentially, cached
  reruns matched. No claim of simultaneous decoding at that limit.
- Text, tools, screenshots, shared-prefix reuse, pair cancellation/recovery,
  scheduler fairness and source/native/FLA pins passed. Peer p95 gap was 2 ms.
- Candidate soak: 656 requests, raw 647 successful; the existing verdict
  accepted 9 known retrieval misses and 0 other failures. Prefix ground-truth
  accuracy was 248/265, distinct from reference consistency. This is not
  perfect model correctness. Image-sidecar gates passed with 40 rounds.

The two-long-context raw client failed its five-second overlap threshold
because one answer finished quickly. The pre-existing judge accepted 115/113
cross-stream deltas (threshold 50) during a 2.48-second overlap; both outputs
matched solo. The raw failure and adjudication remain in the private evidence
archive. No gate was changed during this GPU window.

Open-ended image pairs matched their solo reference in 4/10 cases in both
baseline and candidate. Paired output variability remains visible; the hard
numeric P13 pair test passed.

## Build and CPU checks

The public engine apply/verify reproduced 968/968 files; native-concurrency
pins 29/29 and vision pins 9/9 matched. CPU checks ran 19 source-contract cases,
27 TP cases, 63 concurrency cases (one skip),20 HTTP cases and 4 actual response-
lifetime AST cases. CUDA stayed uninitialized in tensor/source tests; the
HTTP/AST checks did not import Torch. The two repaired CPU harnesses had stale
MTP/generator assumptions; runtime source was not altered to make them pass.

A fresh public native build completed in 891 seconds with Python 3.13.7,
PyTorch 2.14.1+cu130, CUDA toolkit 13.3.73, SM80, four low-priority compile jobs
and GPUs hidden. CPU import passed without CUDA initialization; all 192 public
Python bindings match the qualified binary. The rebuilt hash is
`51471f13a5b4bfe8eaec4d9d4c3ff3b9baae2f1248295c94f9fa089c11f40e03`.
The full GPU window used the existing binary from identical native source,
`b3708b38f2f9f008e97bf29d5fb92803219ec35ff1442097dcc9d5f5dab53dc9`.
Build paths affect binary hashes. No binary, host cache or weights are distributed.

The GPU measurements used a pinned FLA autotune cache on this host. A fresh
installation may retune; the recipe honors `TRITON_CACHE_DIR` and does not
claim that the measured host's cache is portable. Raw machine paths, device
UUIDs, operator keys and private logs are omitted from this public summary.

Post-install production soak: 959 requests, one known tolerated retrieval miss,
zero other failures, clean journal and 175 successful health samples. Prefix
ground-truth 388/400. The CPU-only native compile overlapped this operational
soak after all matched performance measurements were finished. A separate
bounded launch with the fresh public binary passed the 13-case API client and
image checks, with the loaded binary hash verified and a clean pre-stop journal.
The qualified production binary was then restored and checked by a completion.
This fresh-build smoke is separate from the full cache-pressure qualification.
