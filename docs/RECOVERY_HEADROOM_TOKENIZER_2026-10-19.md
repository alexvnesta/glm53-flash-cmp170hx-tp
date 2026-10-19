# Recovery, headroom and tokenizer qualification, October 19

**Status: final GPU qualification passed; qualified cutover staged, not deployed
or merged.** A real209,680-token public request arrived before activation at18:11Z,
so the original production profile is still serving it. The follow-up is scheduled
for15:11Eastern and must reverify currentstate/idle/artifactidentity before acting.
Bounded private GPU windows stopped/restored the original service, with existing
router admission and independent restoration. No active public request was killed.

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
| Actual Generator/Job concurrency, recovery and ASGI health contracts | 75 run, 74 passed, 1 existing optional native-SAM skip |
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
| Full-model mixed vision/headroom and recovery | Final v4 positive cold-tail,16postvision, realrankkill/restart and committedcache restore passed |
| Production deployment | Not performed |

The exact pinned source and community profile reproduce before candidate
patching. Candidate manifests and startup source pins are checked separately;
the changed native extension must never be paired with unmodified sources.
New source/native identity creates a separate persistent-cache fingerprint.

The KDA GPU oracle forces the guard with a 1 TiB floor in a separate process;
14 expected refusal logs are verified. It checks the new fallback control path,
not actual low-memory full-model behavior. The candidate native SHA is
`e4afb2ed95824640a16fb0a183b37a9322d3ab145786a4c0f4b94a5a2fbc2a7e`.
The earlier v2 profile digest is
`0820957d56c3b80384616a5f4995135aa55b1ded622c54e231a03831c541e0b3`.
The final v4 profile digest is
`ea113986b1e75876c7f5efd202015941b6b25bf81e64a4a037f3086b1a3a59cd`.

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
13/15/1/16 tokens after page boundaries. The final runtime passed both cold and warm waves, plus16postvisionrequests.
Earlier negative outcomes remain retained rather than being converted to passes.

## Actual rank-loss negative control and additional recovery fix

The private v2 candidate passed the baseline-first positive mixed-vision/cold-tail
cohort and16 post-vision requests. A fresh process also restored all four committed
text contexts from its isolated persistent cache (1293 page reads, four state reads,
zero I/O errors). These results did **not** establish rank recovery.

A verified private GPU1 worker was killed during simultaneous decoding and long
prefill. Both HTTP consumers stalled, while `/health` falsely returned200 for330s.
The failed receipt and original v2 source/profile remain intact. The TP dispatch
exception path cannot catch a rank death while its executor is inside a native
collective; no deployment followed this negative result.

The service now reads the owned multiprocessing children's CPU exit state in the
existing health, readiness and completion gates. A dead child uses the existing
shared failure coordinator to abort uncertain persistence and wake registered and
pending consumers. Failed cancellation returns before acquiring the blocked native
step lock. In-flight GPU work remains quarantined until the existing watchdog
restarts the process; late results and further idle work are suppressed. This adds
no supervisor, background liveness monitor or restart policy.

The75 CPU checks include a real child-process exit and a deliberately blocked
executor step: health/readiness/completion return503, all consumers receive the
same first error, and HTTP cleanup completes without unlocking the running native
step.52 HTTP, four actual response-lifetime AST and25 router checks also pass.
The final service is immutable candidate-v4, with a fresh profile/cache identity.
Its972 engine files and native artifact are byte-identical to v2. Candidate-v3 was
withdrawn before readiness after an additional CPU negative control exposed a
cancellation already waiting on the step lock; the corrected waiter test passes.

The actual v4 GPU1 rank-kill now produces health and readiness503 in1.787seconds
under0.25s health probes. Both existing SSE consumers receive errors and close in
1.79seconds. The real disk-cache abort warning is recorded. The installed watchdog
restarted the private process after three bad checks, and the same profile became
ready in128.33seconds. This test invokes watchdog checks1sapart and uses a private
5sstop timeout; production retains60schecks,480sstartupgrace and60sstop timeout.
These private timings are not a production recovery SLA.

The fresh recovered process then restores all four8K/32K/98K/196Kcommitted contexts
in28.86seconds:1293page reads, four state reads, zeroI/Oerrors. A final GPU repeat
also passes five four-way/cancellation waves, two mixed vision/text cold/warm waves
with exact tails13/15/1/16, and16low/high postvisionrequests. Native graph refusal
is observed again at405MiBfree under the1024MiBfloor.

The first final-runtime benchmark overlapped external CPU-heavy CI and measured
93.37tok/s. Once pressure cleared, the same runtime repeated102.14tok/s (9.86s),
versus a fresh original-engine control101.07tok/s (9.94s). Treat warm throughput as
similar; this1.06percent bounded difference and earlier4.81percent result do not
establish a general decode gain. All answer/caching oracles pass; output formatting
and reasoning vary by two tokens in some waves. The4x HTTP tokenizer preparation
improvement is separate. Full verdict: `final_qualification_2026-10-19.json`.

## Cutover pending

Full-model GPU qualification requires exclusive access and cannot run alongside
the loaded production model. Alex authorized bounded windows after the current
job finished. Stable router/backend idle is rechecked, an independent restoration
timer is armed before stopping the model, and the window yields when public
traffic arrives. Measured production reload is about two to three minutes;
the router's readiness deadline is 1500 seconds. Preserve the original profile,
unit, caches and restoration path. Those final gates now pass. The exact qualified profile has been prepared in the
canonical production root and its relocation digest reverified, while the live
single drop-in remains original. Activation is deferred for the real public job.
An idle cutover must arm independent rollback, close the admission race through
the existing router, load the exact qualified source/native/flags, and pass plain
and `clear_thinking=false` chat canaries through the router. The original profile,
cache namespace and drop-in backup remain available. The draft PR stays unmerged.

Candidate preparation and policy limits:
[`experiments/recovery_headroom`](../experiments/recovery_headroom/README.md).
Tokenizer upstream attribution, regression and benchmark command:
[`experiments/tokenizer`](../experiments/tokenizer/README.md).
