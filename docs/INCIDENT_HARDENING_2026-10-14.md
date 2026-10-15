# Tool argument parser stall hardening

A deployed GLM HTTP service stopped answering health checks and was recovered by its existing watchdog. No thread dump or final generated tool block was retained before recovery, so the exact blocked operation in that incident is unproven. This change fixes a separately reproduced stall mechanism; it does not establish the historical cause or rule out native/GPU stalls.

The tool parser ran a backtracking regex over model-generated argument tags on the HTTP event loop. Repeated `<arg_key>` tags without a closing key made the scan grow quadratically. On an offline CPU reproduction, 1,000 opening tags took0.023s,4,000 took0.349s, and16,000 exceeded a3s subprocess deadline. Such a scan blocks other HTTP work while it runs.

The parser now consumes the expected delimiters in one forward pass. Missing delimiters fail promptly; the existing non-strict streaming behavior still returns malformed calls as text without executing them. String values, Unicode whitespace, typed JSON/Python-literal values and ordinary streamed calls retain their behavior. The new parser rejected16,000 tags in0.000037s and150,000 in0.000327s in the same offline environment. These are parser timings, not model inference benchmarks.

Validation: all26 HTTP/parser tests passed under the pinned Python3.13/Starlette0.41.3/AnyIO4.15.1 test environment. Six tests cover this parser change, including a bounded subprocess regression for150,000 opening tags. Another504 bounded differential cases matched the previous parser's returned arguments or errors. No Torch, CUDA, model weights or GPU workload was used.

Operationally, preserve consumers and existing watchdog thresholds. Capture process stacks before a future recovery, without pausing the target or collecting local variables. Parser code already imported by a live Python process will require a safe restart to activate; replacing a source file alone is not evidence of activation. No GPU replay or service restart was performed for this fix's verification.
