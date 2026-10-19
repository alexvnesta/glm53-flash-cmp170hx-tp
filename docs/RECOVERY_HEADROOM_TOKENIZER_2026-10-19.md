# Recovery, headroom and tokenizer qualification, October 19

**Status: CPU, small KDA GPU and four-slot lifecycle checks passed; remaining
full-model qualification and deployment pending.** The production four-slot
profile, native library and cache configuration are preserved. Bounded private
GPU windows now stop and restore its service, while the existing router holds
incoming requests. No candidate has been deployed or merged.

## Observed failure and change

On October 17 at 20:09:52 UTC, a 96,526-token request with one image failed
with CUDA graph-instantiation out of memory during TP prefill. Three following
requests each waited approximately 300 seconds for a worker. NCCL reported a
600-second all-reduce timeout; the service restarted at 20:28:29 UTC. This
sequence is observed in the logs. It does not prove the initial allocation's
underlying cause or establish that every vision request risks the same fault.

The prefill path could contain that accelerator error as one request's
failure. It now sends typed accelerator/distributed/transport failures to the
existing shared terminal latch before native per-job cleanup can perform
further rank I/O. The real async failure sink wakes all consumers and the
actual health handler returns 503. The existing watchdog remains responsible
for restart decisions. Ordinary request errors retain per-job handling.

An isolated KDA native patch makes new graph captures optional below a chosen
memory floor, retaining the existing eager kernels for those slots. The
candidate floor is 1 GiB. It does not reserve that memory or guarantee freedom
from OOM, and may reduce speed for newly encountered shapes. The baseline
engine, native profile and default graph policy are preserved.

## CPU/tokenizer results

The initial, unchanged upstream tokenizer backport failed two exact-ID cases
because its one-time probe changed special-token mode. A local restoration
fix is included. The corrected tokenizer passes all **54** cases, including
actual multimodal alias replacement with small CPU tensors. No CUDA context
was initialized.

The following timings use the model's own chat template with a synthetic
code/tool prompt on the serving host, four CPU cores, four Rayon threads,
low process priority, and 12 alternating-order repeats per arm. Each timed
call checks the complete token tensor against the pinned tokenizer. Times
include tokenization and integer tensor construction; rendering is separate.

| Actual tokens | Pinned median | Corrected backport median | Time saved |
|---:|---:|---:|---:|
| 8,356 | 14.61 ms | 14.18 ms | 0.44 ms |
| 32,932 | 79.24 ms | 16.49 ms | 62.75 ms |
| 98,468 | 220.38 ms | 50.17 ms | 170.21 ms |
| 202,182 | 416.43 ms | 94.64 ms | 321.79 ms |
| 288,179 | 652.55 ms | 141.46 ms | 511.09 ms |

These are synthetic CPU preparation measurements, not end-to-end request or
decode gains. The 8K input is below the parallel path's character threshold.
The request-log median prompt was approximately 202K tokens, making the
measured large-input path relevant without establishing an exact production
latency saving.

| Check | Result |
|---|---|
| Actual Generator/Job concurrency, recovery and ASGI health contracts | 72 run, 71 passed, 1 existing optional native-SAM skip |
| HTTP contracts | 52 passed |
| Actual response-handler AST / real Starlette cleanup | 4 passed |
| Model-router contracts | 25 passed |
| Existing concurrency-experiment CPU contracts | 55 passed |
| Actual KDA guard compiled against CPU CUDA boundary | 5 passed |
| Corrected tokenizer exact-ID qualification | 54 passed |
| Native compilation | One KDA C++ object rebuilt; 190 attested objects reused; import passed with CUDA hidden |
| Small synthetic KDA GPU graph/eager comparison | 14 cases across both SM80 cards, four runs each; output, convolution and recurrent state bit exact |
| Private full-model four-slot JSON/code and lifecycle | Five four-way waves; 24 completed, two canceled; survivors and replacement requests pass |
| Actual HTTP code/tool tokenization | Six repeats per frozen body; identical token counts and body hashes, 194K: 392.62 to 90.50 ms; 277K: 544.84 to 127.68 ms |
| Frozen four-request warm-cache cohort | Three repeats, identical input hashes, output token counts and cached-prefix lengths; median 97.77 to 102.47 tokens/s (+4.81%), wall time 10.30 to 9.83 s |
| Full-model mixed vision/headroom and recovery | Pending baseline-first positive cold-tail and rank/cache tests |
| Production deployment | Not performed |

The exact pinned source and community profile reproduce before candidate
patching. Candidate manifests and startup source pins are checked separately;
the changed native extension must never be paired with unmodified sources.
New source/native identity creates a separate persistent-cache fingerprint.

The KDA GPU oracle forces the guard with a 1 TiB floor in a separate process;
14 expected refusal logs are verified. It checks the new fallback control path,
not actual low-memory full-model behavior. The candidate native SHA is
`e4afb2ed95824640a16fb0a183b37a9322d3ab145786a4c0f4b94a5a2fbc2a7e`.
The private model's profile digest is
`0820957d56c3b80384616a5f4995135aa55b1ded622c54e231a03831c541e0b3`.

The first private model launch failed before readiness because the transient
test unit omitted production's `LimitNOFILE=65536`. The attempt is retained;
the corrected launch passed the four-slot checks. The original production
profile was restored after both attempts. A private control benchmark also
hit its own 180-second read deadline on an uncached 196K prompt; that failed
run is retained and is not a candidate regression or a passed benchmark.

The actual HTTP `/tokenize` measurement includes JSON transport, the serving
renderer and real tokenizer/tensor construction. Its 194K/277K medians improve
4.34x/4.27x; this is prompt preparation, not a decode multiplier. The GPU cohort
has four independent 8K/32K/98K/196K prompts. Compare its three fully warm waves;
the first control retry wave was partly warm following the retained timeout.
A separate private original-engine replay reproduced 97.69 tokens/s, closely
matching the first 97.77 control. These are bounded synthetic measurements,
not a general production throughput claim.

An initial mixed-vision marker/JSON oracle failed on a 34,158-token blue-image
request: it returned arrays for a/b/c instead of the required marker strings.
The unchanged original engine reproduced the **identical complete answer**,
608 completion tokens, 617 proposed / 545 accepted draft tokens, and 62
reasoning characters. Both absolute oracle failures are retained. All four
recorded cohort token-ID tensors also match exactly between the two real
tokenizer classes, using the actual service renderer, vision rewrite and
MMEmbedding splice with CPU dummy embedding values. This is evidence of an
existing task limitation, not a repaired answer or a newly passing oracle.

The headroom policy was observed refusing new full-model graph slots at
405 MiB free under the 1024 MiB floor, without a native failure in that cohort.
A separate baseline-first positive test asks the image requests only for their
dominant colors and retains the long text-only marker/JSON oracles. It targets
exact total lengths 98318/32784/131074/196625, leaving native prefill tails of
13/15/1/16 tokens after page boundaries. Those checks and actual rank/cache
recovery remain pending; no deployment follows from the partial results.

## Remaining gates

Full-model GPU qualification requires exclusive access and cannot run alongside
the loaded production model. Alex authorized bounded windows after the current
job finished. Stable router/backend idle is rechecked, an independent restoration
timer is armed before stopping the model, and the window yields when public
traffic arrives. Measured production reload is about two to three minutes;
the router's readiness deadline is 1500 seconds. Preserve the original profile,
unit, caches and restoration path. Full-model mixed vision/headroom, comparable
performance, committed-cache recovery and actual rank-failure recovery must
pass before deployment. The draft PR remains unmerged.

Candidate preparation and policy limits:
[`experiments/recovery_headroom`](../experiments/recovery_headroom/README.md).
Tokenizer upstream attribution, regression and benchmark command:
[`experiments/tokenizer`](../experiments/tokenizer/README.md).
