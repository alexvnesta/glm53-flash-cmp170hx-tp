"""Explicit JSON-object output using upstream per-job grammar state.

Imports remain CPU-only until a filter is requested. Free-form text and tool
calls retain their existing behavior; no output is repaired after generation.
"""
from functools import lru_cache
import json
import weakref
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text", "json_object"]


def requested(body, *, enabled, chat):
    fmt = getattr(body, "response_format", None)
    if fmt is None or fmt.type == "text":
        return False
    if not enabled:
        raise ValueError("JSON mode is disabled; enable --experimental-json-mode after qualification")
    if not chat:
        raise ValueError("JSON mode requires /v1/chat/completions")
    if body.tools or body.tool_choice not in (None, "none", "auto"):
        raise ValueError("JSON mode cannot be combined with function tools")
    if body.stop or body.ignore_eos or body.min_tokens:
        raise ValueError("JSON mode cannot use custom stop, ignore_eos or min_tokens")
    return True


@lru_cache(1)
def filter_type():
    from exllamav3.generator.filter.llguidance import LLGuidanceFilter

    class JsonObjectFilter(LLGuidanceFilter):
        def __init__(self, tokenizer):
            trigger = tokenizer.single_id("</think>")
            if type(trigger) is not int or trigger < 0:
                raise ValueError("JSON mode requires a single-token </think> marker")
            super().__init__(tokenizer, trigger_token=trigger,
                             eos_after_completed=True, json_schema={"type": "object"})
            # Retain the issued grammar identity; arbitrary filters and modified
            # grammar objects must not cross the qualified concurrency guard.
            self._glm_json_grammar = json.dumps(self._grammar, sort_keys=True)
            self._glm_json_owner = None

    return JsonObjectFilter


def build_filter(tokenizer):
    return filter_type()(tokenizer)


def permitted_filters(filters, tokenizer, job):
    if len(filters) != 1:
        return False
    f = filters[0]
    valid = (type(f) is filter_type() and f.tokenizer is tokenizer
            and f.trigger_token == tokenizer.single_id("</think>")
            and f.eos_after_completed is True and f.prefix_str is None
            and f._consume_prefix is False
            and json.dumps(f._grammar, sort_keys=True) == f._glm_json_grammar)
    if not valid:
        return False
    if f._glm_json_owner is None:
        f._glm_json_owner = weakref.ref(job)
    return f._glm_json_owner() is job


def validate_complete(text):
    def invalid_constant(value):
        raise ValueError("Non-JSON number: " + value)

    value = json.loads(text, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("JSON mode produced a non-object response")


def check_dependencies():
    from exllamav3.generator.filter.llguidance import llguidance_available
    if not llguidance_available:
        raise ValueError("JSON mode requires the optional llguidance dependency")


def health(enabled):
    return {"enabled": enabled, "formats": ["text", "json_object"] if enabled else ["text"],
            "backend": "upstream_llguidance" if enabled else None,
            "reasoning_trigger": "</think>" if enabled else None,
            "schema_output": False}
