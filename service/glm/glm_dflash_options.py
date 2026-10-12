"""Optional DFlash generator settings; no model, verifier or cache mutation.

Use generator_kwargs() before constructing the existing Generator. Defaults
retain fixed K7. Stats describe observed verification rounds, not losslessness.
"""
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _flag(env: Mapping[str, str], name: str) -> bool:
    value = env.get(name, "0").strip().lower()
    if value not in ("0", "1", "false", "true"):
        raise ValueError(f"{name} must be 0, 1, false or true")
    return value in ("1", "true")


@dataclass(frozen=True)
class DFlashOptions:
    adaptive: bool = False
    confidence: float = 0.4
    record_stats: bool = False

    def __post_init__(self):
        if type(self.adaptive) is not bool or type(self.record_stats) is not bool:
            raise ValueError("DFlash switches must be bool")
        if not math.isfinite(self.confidence) or not 0.0 < self.confidence < 1.0:
            raise ValueError("draft confidence must be finite and in (0, 1)")

    @classmethod
    def from_environment(cls, env: Mapping[str, str] | None = None):
        env = os.environ if env is None else env
        return cls(_flag(env, "GLM53_DYNAMIC_DRAFT"),
                   float(env.get("GLM53_DRAFT_CONFIDENCE", "0.4")),
                   _flag(env, "GLM53_RECORD_DRAFT_STATS"))

    def generator_kwargs(self, *, num_draft_tokens: int = 7,
                         draft_block_size: int = 8, max_q_size: int = 8,
                         max_batch_size: int = 1) -> dict[str, Any]:
        # This service adapter's qualified geometry. Truncation changes the used
        # draft window, while the fixed draft block and cache allocation stay 8.
        if self.adaptive and (num_draft_tokens != 7 or draft_block_size != 8 or
                              max_q_size != 8 or max_batch_size != 1):
            raise ValueError("adaptive service mode requires K7, block8, q8 and batch1")
        return {"dynamic_draft_tokens": self.adaptive,
                "draft_confidence": self.confidence,
                "record_draft_stats": self.record_stats}


def generator_kwargs(env: Mapping[str, str] | None = None, **geometry):
    """Return only existing Generator options; leave geometry to the caller."""
    return DFlashOptions.from_environment(env).generator_kwargs(**geometry)


def summarize_draft_stats(rounds) -> dict[str, int | float | None]:
    """Summarize existing job.draft_stats (position, window, accepted) records."""
    count = proposed = accepted = 0
    for row in rounds:
        if len(row) != 3 or any(type(value) is not int for value in row):
            raise ValueError("draft round must contain three integers")
        position, window, verified = row
        if position < 0 or window < 0 or not 0 <= verified <= window:
            raise ValueError("invalid observed draft round")
        count += 1
        proposed += window
        accepted += verified
    return {"rounds": count, "proposed": proposed, "accepted": accepted,
            "acceptance": accepted / proposed if proposed else None}
