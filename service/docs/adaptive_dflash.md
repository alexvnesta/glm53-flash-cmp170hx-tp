# Optional adaptive DFlash rounds

Defaults retain fixed K7. `GLM53_DYNAMIC_DRAFT=1` enables the existing engine’s confidence-based truncation within its fixed block 8 geometry. `GLM53_DRAFT_CONFIDENCE` selects the finite target acceptance probability in (0,1), default 0.4. `GLM53_RECORD_DRAFT_STATS=1` independently records existing verification observations and adds their round/proposed/accepted totals to final API timings. Stats are read from the actual job wrapped by AsyncJob.

Only the native Generator options `dynamic_draft_tokens`, `draft_confidence` and `record_draft_stats` are supplied. The adapter does not create a new verifier, change cache allocations or change block size. MTP/undrafted configurations refuse enabled DFlash options. Invalid switches/confidence fail before model allocation.

The DFlash diffusion drafter still executes its full fixed block 8. Adaptive mode truncates the subsequent target verification window after confidence export/calibration, which may reduce wasted target work but adds CPU transfers and more rounds. Changing query shape can change floating-point kernel behavior, so this is experimental default-off; no output parity or speed benefit is asserted without a matched model campaign. Acceptance is a performance observation, not a losslessness proof. Evaluate cache restoration separately with adaptive mode disabled first.

The source-level CPU tests require an explicit ExLlama source checkout through `EXLLAMAV3_ENGINE_ROOT`. They exercise the existing calibrator and DFlash truncation method with CPU fakes and initialize no CUDA context.
