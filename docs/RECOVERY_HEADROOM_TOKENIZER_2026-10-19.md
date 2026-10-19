## Latest attempt: yielded to real traffic at 19:58 UTC

A stable idle boundary allowed a bounded private v5 pressure window at 19:53.
The exact candidate/native identities and all engine/service manifests were
reverified. The candidate was ready at 19:55:37 with an effective namespace budget
of zero under the private 50 GiB root. A 19:57:38 monitor sample showed responsive
health, growing long-prompt prefill, 752 refused writes, no disk writes and zero
write errors. The first cold four-context wave had not completed when public
traffic arrived at 19:58:23. This is partial evidence, **not a passed GPU gate**.
The 512 MiB phase and full combined suite have not run.

The harness yielded immediately; original production started at 19:58:31 and was
ready at 20:00:29. At 20:00:42 it was healthy and streaming a real 176,151-token
request, with 180 generated tokens and 38 ms progress age. The router held this
request during restoration. Private units/timers are inactive, the watchdog is
active, and the canonical drop-in is unchanged. The synthetic fixture was archived
after the private model stopped. No production namespace was moved or deleted.

The interrupted first cold wave had not saved its benchmark summary. Its fixture
and journals are preserved. The harness now saves initial/phase checkpoints and
flushed health progress, and handles SIGTERM to retain partial receipts. A CPU-only
interruption smoke passed; candidate serving sources are unchanged.

A read-only canonical-root inventory at 19:54:55 found 53,641,060,553 logical bytes
under the 53,687,091,200-byte root budget, leaving just **46,030,647 bytes (44 MiB)**
for a new namespace at that moment. This makes effective cache capacity a separate
deployment gate. Historical namespace `db87d6cf58243789` holds 24,134,296,538 bytes;
it is a potential reversible archive candidate, subject to fresh owner/handle
verification. No archive was performed. Preserve exact contents/permissions and
paired configuration/cache rollback before any migration after owners stop.

# Post-deployment failure: original production restored, work unfinished

The first real request after the initial canaries stalled at18:19Z. Two
non-pausing Python stack samples show the GPU executor spinning in the upstream
DiskPageCache eviction loop while the HTTP loop waits for the cache lock.
Physical disk space is ample. Unlike the isolated qualification root, the
production cache root contains several retained fingerprints sharing its50GiB
budget. A zero or insufficient namespace share and rooted checkpoints can leave
no eviction candidate; the old make_room loop neither progresses nor refuses.
The exact historical effective share was not recorded, so zero share remains a
reproduction condition rather than a measured live value.

The original profile-db01c25/nativef0/drop-in was restored at18:25:20Z. Its model
became ready18:28:35Z and a real214,978-token request streamed with fresh progress.
The stalled request lost its connection during rollback. The earlier idle cutover
record below is historical evidence, not approval for continued use. Failed
profile/namespace and both stack samples are preserved. The one-off follow-up is
active again for17:01Eastern; the weekly read-only review is unchanged.

A new staged patch bounds eviction progress and refuses page/checkpoint writes
before file creation, TP dispatch or reservation when no budget can be made.
It preserves the RAM tier and existing disk data, and reports the effective
namespace budget separately from the shared root budget. Twelve CPU checks run
actual pinned cache methods: the original negative control spins in a bounded
child process, protected siblings/rooted states and pending reservations refuse
safely, normal leaf-first eviction works, no rejected write issues file/rank I/O,
reservation/error accounting is preserved, and actual CPUPageCache.store retains
its RAM entry. The serving-host17combinedpolicy tests,75concurrency checks(74passed/oneexisting
skip) and53HTTP checks pass; CI succeeds on059749c. Immutablecandidate-v5 is
prepared and imported with GPUs hidden. New GPU/shared-root qualification remains
required; the18:47attempt sawbusy_no_changes before touching serving. No serving code is being edited in place.
Current upstream cache commit1ac6d1271e7a5a75839804b9cfb358cc475f46a4 addresses Windows
statvfs compatibility; it does not fix this eviction condition.

## Superseded initial qualification and cutover record

# Recovery, headroom and tokenizer qualification, October 19

**Status: qualified and deployed2026-10-19 18:17:14 UTC; PR22 remains draft and
unmerged.** A real209,680-token request arrived before activation and was left
running on the original profile. It completed before the stable idle cutover.
Both plain and `clear_thinking=false` real router canaries returned exactly `ok`.
Independent verification matched972engine files,74service Python files, actual
process cwd/cmdline/native maps and selected environment, the single drop-in,
backend/router digest and exclusive canonical GPU ownership. The original
profile/drop-in/cache remain retained for rollback. The one-off follow-up is
removed; the weekly read-only review remains active.

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
| Production deployment | Passed18:17:14Z: exact qualified profile, both router canaries; independent live verification18:17:55Z |

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

## Production activation

All required qualification gates passed before deployment. The exact prepared
profile was relocated into the canonical production root and its digest reverified.
The real209Kpublic request arriving18:11Z was allowed to finish; stable idle was
observed18:14:15Z. The guarded cutover then rechecked idle, armed independent
rollback and closed admission through the existing router. Both real chat canaries
passed18:17:14Z,125.43seconds after admission closed. The original profile and its
exact single-drop-in backup, compilation cache and persistent namespace remain
available. New prefixes warm in a distinct persistent-cache namespace.

Actual production: `profile-recovery-f8bd8a3`, PID1310405 at18:17:55Z,
routerPID895282 unchanged, digest
`ea113986b1e75876c7f5efd202015941b6b25bf81e64a4a037f3086b1a3a59cd`,
native `e4afb2ed95824640a16fb0a183b37a9322d3ab145786a4c0f4b94a5a2fbc2a7e`.
Independent verification matched all972engine files,74service Python files, actual
process/native maps/environment and selected configuration. Only the canonical
service owns both GPUs; Qwen and private units are inactive, the existing watchdog
is active, and deployment rollback is disarmed by the committed canaries. The
canonical deployment README records the exact runtime recipe; its `repo/` main
checkout remainsdb01c25 and is not the live profile. PR22 stays draft and unmerged.
The one-off follow-up is removed; the weekly read-only review remains active.

Candidate preparation and policy limits:
[`experiments/recovery_headroom`](../experiments/recovery_headroom/README.md).
Tokenizer upstream attribution, regression and benchmark command:
[`experiments/tokenizer`](../experiments/tokenizer/README.md).
