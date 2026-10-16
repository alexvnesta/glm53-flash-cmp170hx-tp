"""Bounded request timing and progress; never retain prompts, text or tool arguments."""
from collections import Counter, deque
import json
import threading
import time


def journal(record):
    try:
        print("GLM_REQUEST " + json.dumps(record, separators=(",", ":"), allow_nan=False), flush=True)
    except (OSError, ValueError):
        # An unavailable log sink must not change a model response or strand an admission slot.
        pass


class RequestMetrics:
    def __init__(self, *, clock=time.perf_counter, sink=journal, history=32):
        self.clock, self.sink = clock, sink
        self.lock = threading.Lock()
        self.active = {}
        self.recent = deque(maxlen=history)
        self.counts = Counter()
        self.identity = {}

    def start(self, ident, *, received, prompt_tokens, effort, streaming, images=0):
        record = RequestRecord(self, ident, received, prompt_tokens, effort, streaming, images)
        with self.lock:
            self.active[ident] = record
        return record

    def snapshot(self):
        now = self.clock()
        with self.lock:
            return {"active": [r.progress(now) for r in self.active.values()],
                    "completed": dict(self.counts), "recent": list(self.recent)}


class RequestRecord:
    def __init__(self, owner, ident, received, prompt_tokens, effort, streaming, images):
        self.owner, self.ident, self.received = owner, ident, received
        self.prepared = owner.clock()
        self.last_progress = self.prepared
        self.phase = "admission"
        self.prefill_done = self.prefill_total = self.generated = 0
        self.engine_first = self.visible_first = None
        self.outcome = "cancelled"
        self.error_type = None
        self.finished = False
        self.fields = {"id": ident, "profile": dict(owner.identity), "prompt_tokens": prompt_tokens, "reasoning_effort": effort,
                       "streaming": streaming, "images": images}

    def observe(self, result):
        now = self.owner.clock()
        stage = result.get("stage", "unknown")
        self.phase = stage
        progress = result.get("prefill_progress")
        if isinstance(progress, (tuple, list)) and len(progress) == 2:
            self.prefill_done, self.prefill_total = map(int, progress)
        elif isinstance(progress, (int, float)):
            self.prefill_done = int(progress)
        # Native prefill results report curr/max_progress; these are CPU metadata only.
        self.prefill_done = int(result.get("curr_progress", self.prefill_done))
        self.prefill_total = int(result.get("max_progress", self.prefill_total))
        native = result.get("job")
        # Streaming results expose new_tokens at EOS only; native counters are plain
        # Python integers, so this does not copy tensors or synchronize a GPU.
        generated = result.get("new_tokens", getattr(native, "rq_new_tokens", 0)+getattr(native, "new_tokens", 0))
        self.generated = max(self.generated, int(generated))
        if stage == "streaming" and self.engine_first is None:
            self.engine_first = now
        self.last_progress = now

    def visible(self):
        if self.visible_first is None:
            self.visible_first = self.owner.clock()

    def fail(self, error, outcome="error"):
        self.outcome, self.error_type = outcome, type(error).__name__

    def progress(self, now):
        return {**self.fields, "phase": self.phase, "elapsed_ms": (now-self.received)*1000,
                "progress_age_ms": (now-self.last_progress)*1000,
                "prefilled_tokens": self.prefill_done, "prefill_total_tokens": self.prefill_total,
                "generated_tokens": self.generated}

    def finish(self, timings, usage, reason):
        if self.finished:
            return
        self.finished = True
        now = self.owner.clock()
        row = {"event": "glm_request", "version": 1, **self.fields,
               "outcome": self.outcome, "finish_reason": reason, "error_type": self.error_type,
               "preparation_ms": (self.prepared-self.received)*1000,
               "wall_ms": (now-self.received)*1000,
               "engine_ttft_ms": None if self.engine_first is None else (self.engine_first-self.received)*1000,
               "visible_ttft_ms": None if self.visible_first is None else (self.visible_first-self.received)*1000,
               "completion_tokens": usage.get("completion_tokens") or self.generated,
               **{k: v for k, v in timings.items() if k in
                  ("http_queue_ms", "prompt_ms", "predicted_ms", "queue_ms", "prompt_n", "predicted_n",
                   "source", "cached_tokens", "cache_n", "draft_stats", "draft_n", "draft_n_accepted")}}
        if row.get("predicted_ms", 0) > 0:
            row["decode_tokens_per_second"] = row["completion_tokens"]*1000/row["predicted_ms"]
        if row.get("draft_n", 0) > 0:
            row["draft_acceptance"] = row.get("draft_n_accepted", 0)/row["draft_n"]
        with self.owner.lock:
            self.owner.active.pop(self.ident, None)
            self.owner.counts[self.outcome] += 1
            self.owner.recent.append(row)
        self.owner.sink(row)
