"""Default-off continuous batching options, validated before model allocation."""
import asyncio
import os
from dataclasses import dataclass


class CapacityGate(asyncio.BoundedSemaphore):
    """Admission-compatible owned capacity, held through response cleanup."""
    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        super().__init__(capacity)
        self.capacity = capacity

    @property
    def active(self):
        return self.capacity - self._value


@dataclass(frozen=True)
class ConcurrentOptions:
    enabled: bool = False
    max_sequences: int = 1
    fair_gen_rounds: int = 1
    fair_chunk_size: int | None = None

    def generator_kwargs(self):
        return {} if not self.enabled else dict(fair_gen_rounds=self.fair_gen_rounds,
                                                fair_chunk_size=self.fair_chunk_size)


def concurrent_options(args, serving_chunk):
    enabled = getattr(args, "experimental_concurrent_generation", False)
    if type(enabled) is not bool:
        raise ValueError("experimental_concurrent_generation must be a bool")
    overrides = [getattr(args, name, None) for name in
                 ("concurrent_max_sequences", "concurrent_fair_gen_rounds", "concurrent_fair_chunk_size")]
    if not enabled:
        if any(value is not None for value in overrides):
            raise ValueError("Concurrency settings require --experimental-concurrent-generation")
        return ConcurrentOptions()
    if os.environ.get('GLM53_TP_ACTIVE_HOST', '0') not in ('', '0'):
        raise ValueError("Concurrent initial profile excludes TP active-host migration")
    sequences, rounds, chunk = [default if value is None else value for value, default in
                              zip(overrides, (2, 4, 512))]
    if type(sequences) is not int or not 2 <= sequences <= 8:
        raise ValueError("concurrent max sequences must be an integer 2..8")
    if type(rounds) is not int or not 1 <= rounds <= 8:
        raise ValueError("concurrent fair rounds must be an integer 1..8")
    if type(chunk) is not int or chunk <= 0 or chunk % 256 or chunk > serving_chunk:
        raise ValueError("concurrent fair chunk must be positive, page aligned and no greater than serving chunk")
    if getattr(args, "tensor_parallel", False) is not True or getattr(args, "experimental_dflash2_tp", False) is not True:
        raise ValueError("Concurrent profile requires the explicit TP DFlash2 opt-in")
    if getattr(args, 'num_draft_tokens', 7) != 7:
        raise ValueError("Concurrent initial profile requires fixed K7")
    if getattr(args, "swa_full", False):
        # P16 stage 2: each job's draft state is upstream's own sliding-window ring (7de419b), checkpointed with the
        # target as one PairedState; -swa_full would build a pool-sized paged draft cache and no ring state.
        raise ValueError("Native concurrency uses upstream's sliding-window DFlash2 draft state; drop -swa_full")
    if not getattr(args, "draft_model_dir", None) or getattr(args, "mtp", False):
        raise ValueError("Concurrent profile requires DFlash2")
    if getattr(args, "autosplit_max_batch_size", None) != sequences:
        raise ValueError("--autosplit_max_batch_size must equal concurrent max sequences for allocation parity")
    if any(getattr(args, name, False) for name in ("dflash_prefix_cache", "dflash_session_cache")):
        raise ValueError("Single-active paired retention cannot be combined with concurrent generation")
    if getattr(args, "target_cpu_cache_gib", 0) or getattr(args, "cpu_cache_size", 0):
        raise ValueError("Concurrent initial profile excludes CPU tiers")
    if serving_chunk > 2048:
        raise ValueError("Concurrent profile requires a serving chunk of at most 2048 (qualified)")
    return ConcurrentOptions(True, sequences, rounds, chunk)


def require_fair_scheduling(generator_class, job_class):
    """Before load: the engine Generator takes fair_gen_rounds/fair_chunk_size and Job.prefill a chunk size.

    Upstream's Generator(**kwargs) accepts and ignores unknown keyword arguments, so an engine without the fair
    scheduling patch would start and silently lose it (P16 inventory). Fail loudly instead."""
    import inspect
    parameters = inspect.signature(generator_class.__init__).parameters
    prefill = inspect.signature(getattr(job_class.prefill, "_glm_native", job_class.prefill)).parameters
    if (any(name not in parameters or parameters[name].kind is inspect.Parameter.VAR_KEYWORD
            for name in ("fair_gen_rounds", "fair_chunk_size"))
            or "chunk_size" not in prefill):
        raise ValueError("The engine Generator does not support fair scheduling (fair_gen_rounds, fair_chunk_size "
                         "and Job.prefill(chunk_size)); this service needs the P16 fair scheduling engine patch")


def check_fair_scheduling(generator, options):
    """After construction: the generator holds exactly the requested fair scheduling settings."""
    if not options.enabled:
        return
    if (getattr(generator, "fair_gen_rounds", None) != options.fair_gen_rounds
            or getattr(generator, "fair_chunk_size", None) != options.fair_chunk_size):
        raise ValueError("The engine Generator did not apply the requested fair scheduling settings")
