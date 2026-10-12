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
    ring_tokens_per_sequence: int = 8192

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
    if not getattr(args, "draft_model_dir", None) or getattr(args, "mtp", False):
        raise ValueError("Concurrent profile requires DFlash2")
    if getattr(args, "autosplit_max_batch_size", None) != sequences:
        raise ValueError("--autosplit_max_batch_size must equal concurrent max sequences for allocation parity")
    if any(getattr(args, name, False) for name in ("dflash_prefix_cache", "dflash_session_cache")):
        raise ValueError("Single-active paired retention cannot be combined with concurrent generation")
    if getattr(args, "target_cpu_cache_gib", 0) or getattr(args, "cpu_cache_size", 0):
        raise ValueError("Concurrent initial profile excludes CPU tiers")
    if serving_chunk > 2048 or getattr(args, "cache_size", 0) < 8192 * sequences:
        raise ValueError("Concurrent ring requires serving chunk <=2048 and cache >=8192 tokens per sequence")
    return ConcurrentOptions(True, sequences, rounds, chunk)
