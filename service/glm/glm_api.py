#!/usr/bin/env python3
"""OpenAI-compatible text API for the existing GLM EXL3/MTP installation.

One request runs at a time by default. Explicit experimental concurrency uses
separate draft rings and the native continuously batched target verifier.
The model's own Jinja chat template is used without requiring transformers.
"""
import argparse
import asyncio
import anyio
from contextlib import asynccontextmanager, nullcontext
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Literal
import uuid

# Make sibling modules importable when launched as a script from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from glm_response_cleanup import OwnedJobCleanup, OwnedStreamingResponse
from glm_admission import RequestAdmission, AdmissionPolicy, AdmissionError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from glm_profile import CONTEXT_RESERVE, MODEL_CONTEXT_LIMIT, MODEL_ID, PORT, GENERATION_DEFAULTS
from glm_tools import prepare_tools, ToolSplitter, names_strict
from glm_paths import KERNELS_DIR, HCFUSE_ENV, HCFUSE_CHECK_ENV
from glm_session_options import session_options
from glm_tp_options import tp_dflash_options, tp_dflash_health
from glm_tp_lifecycle import tp_lifecycle
from glm_dflash_options import DFlashOptions, summarize_draft_stats
from glm_concurrent_options import concurrent_options, CapacityGate
from glm_b2_options import b2_retention_options
from glm_body_limit import RequestBodyLimit, body_limit_state

# A call to a function the request did not advertise: "pass" hands it to the client as a normal tool call, "error" fails the request.
UNADVERTISED_TOOL_CALLS = os.environ.get("GLM_UNADVERTISED_TOOL_CALLS", "pass").strip().lower()
if UNADVERTISED_TOOL_CALLS not in ("pass", "error"):
    raise ValueError("GLM_UNADVERTISED_TOOL_CALLS must be pass or error")

# Largest accepted request body (GLM_MAX_REQUEST_BYTES, default 32 MiB; an invalid value stops the service at start):
# a larger one gets 413 before it is parsed. The limiter's state lives in runtime, where /health reads it.
runtime = {"request_body": body_limit_state(os.environ)}
REQUEST_ADMISSION_POLICY = AdmissionPolicy.from_environment()


# The deployed ExLlamaV3 generator page size is fixed at 256 tokens.
PREFILL_PAGE_SIZE = 256
# A defaulted output limit is lowered to the room left only when at least this much remains; less room is reported as a
# context overflow, so agent clients compress instead of getting a reasoning-only reply cut off at a few tokens.
DEFAULT_OUTPUT_FLOOR = 4096


def prefill_chunk_size(args):
    """Validate before model allocation; optionally cap serving only, not autosplit."""
    load_chunk = args.chunk_size
    cap = getattr(args, "prefill_chunk_size", None)
    for name, value in (("--chunk_size", load_chunk), ("--prefill-chunk-size", cap)):
        if value is None and name == "--prefill-chunk-size":
            continue
        if type(value) is not int or value <= 0 or value % PREFILL_PAGE_SIZE:
            raise ValueError(f"{name} must be a positive multiple of {PREFILL_PAGE_SIZE}")
    if cap is not None and cap > load_chunk:
        raise ValueError("--prefill-chunk-size must not exceed --chunk_size")
    return load_chunk if cap is None else cap



class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ToolDefinition(BaseModel):
    type: Literal["function"]
    function: FunctionDefinition


class FunctionCall(BaseModel):
    name: str = Field(min_length=1)
    # History: null/empty/non-object arguments render as {} (prepare_tools) instead of failing every later turn.
    arguments: str | dict | None = "{}"


class ToolCall(BaseModel):
    # Optional in history: without ids the template pairs tool results by message order.
    id: str | None = None
    type: Literal["function"] = "function"
    function: FunctionCall


class NamedToolChoice(BaseModel):
    type: Literal["function"]
    function: dict[str, str]


class Message(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict] | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


# The chat template only knows low/high/max (any other value renders as Max), so
# OpenAI-standard effort names are normalized to those before reaching it.
# Thinking cannot be disabled, so none/off/empty are refused rather than mapped.
EFFORT_LEVELS = ("low", "high", "max")
EFFORT_NAMES = {"minimal": "low", "low": "low", "medium": "high",
                "high": "high", "xhigh": "max", "max": "max"}
EFFORT_MAPPING = ", ".join(f"{name}->{level}" for name, level in EFFORT_NAMES.items())
EFFORT_HELP = (f"Thinking effort (default {GENERATION_DEFAULTS['reasoning_effort']}). Case-insensitive; "
               f"accepted names map as {EFFORT_MAPPING}. Thinking cannot be disabled. "
               f"An explicit value overrides the effort of a {MODEL_ID}-<level> model alias.")
# Some clients (e.g. pi-ai with thinkingFormat deepseek) send thinking={"type": "enabled"|"disabled"}; vLLM-style servers accept adaptive/auto too.
THINKING_TYPES = ("enabled", "disabled", "adaptive", "auto")
# For clients that can only pick a model name: "<MODEL_ID>-<level>" is the same model at that effort.
MODEL_ALIASES = {f"{MODEL_ID}-{level}": level for level in EFFORT_LEVELS}
# OpenAI documents these as nullable and openai-python sends an explicit None as null: null means "not given" (the default), as in vLLM.
NULL_MEANS_DEFAULT = ("model", "max_tokens", "max_completion_tokens", "temperature", "top_p", "frequency_penalty",
                      "presence_penalty", "stream", "ignore_eos", "min_tokens", "reasoning_effort", "n", "parallel_tool_calls")


def normalize_effort(value):
    if isinstance(value, str) and value.lower() in EFFORT_NAMES:
        return EFFORT_NAMES[value.lower()]
    raise ValueError(f"unsupported reasoning_effort {repr(value)[:40]}; accepted (case-insensitive): "
                     f"{EFFORT_MAPPING}. Thinking cannot be disabled on this model, "
                     "so none, off and empty values are rejected")


def resolve_request_model(model, effort, effort_given):
    """Return (served model id, effort); an alias supplies its effort unless the body gave one."""
    alias_effort = MODEL_ALIASES.get(model)
    if alias_effort is None:
        return model, effort
    return MODEL_ID, effort if effort_given else alias_effort


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = MODEL_ID
    messages: list[Message] | None = None
    prompt: str | None = None
    max_tokens: int = Field(default=GENERATION_DEFAULTS["max_tokens"], ge=1, le=MODEL_CONTEXT_LIMIT)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=MODEL_CONTEXT_LIMIT)
    temperature: float = Field(default=GENERATION_DEFAULTS["temperature"], ge=0, le=2, allow_inf_nan=False)
    top_p: float = Field(default=GENERATION_DEFAULTS["top_p"], gt=0, le=1, allow_inf_nan=False)
    frequency_penalty: float = Field(default=0, ge=-2, le=2, allow_inf_nan=False)
    presence_penalty: float = Field(default=0, ge=-2, le=2, allow_inf_nan=False)
    stop: str | list[str] | None = None
    seed: int | None = None
    stream: bool = False
    stream_options: dict | None = None
    ignore_eos: bool = Field(default=False, strict=True)
    min_tokens: int = Field(default=0, ge=0, le=MODEL_CONTEXT_LIMIT)
    # pi-ai-style clients send store=false for OpenAI chat requests. This API does not
    # persist completions; reject an explicit storage request before generation.
    store: bool | None = Field(default=False, strict=True)
    reasoning_effort: Literal["low", "high", "max"] = Field(
        default=GENERATION_DEFAULTS["reasoning_effort"], description=EFFORT_HELP,
        json_schema_extra={"enum": list(EFFORT_NAMES)})
    n: Literal[1] = 1
    user: str | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: Literal["auto", "none", "required"] | NamedToolChoice | None = None
    parallel_tool_calls: bool = True
    # Accepted for client compatibility. This model always thinks, so thinking "disabled" selects the lowest effort unless
    # reasoning_effort is given; the other fields are standard OpenAI request fields with no effect here.
    thinking: dict | None = None
    metadata: dict | None = None
    service_tier: str | None = None
    prompt_cache_key: str | None = None
    safety_identifier: str | None = None

    @model_validator(mode="before")
    @classmethod
    def drop_null_defaults(cls, data):
        if isinstance(data, dict):
            data = {key: value for key, value in data.items() if value is not None or key not in NULL_MEANS_DEFAULT}
        return data

    @field_validator("thinking", mode="before")
    @classmethod
    def check_thinking(cls, value):
        if value is None:
            return None
        kind = value.get("type") if isinstance(value, dict) else None
        if not isinstance(kind, str) or kind.lower() not in THINKING_TYPES:
            raise ValueError('thinking must be an object like {"type": "enabled"} or {"type": "disabled"} (also adaptive or auto); '
                             "this model always thinks, so disabled selects the lowest effort")
        return {**value, "type": kind.lower()}

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def normalize_reasoning_effort(cls, value):
        return normalize_effort(value)

    @model_validator(mode="after")
    def resolve_model_alias(self):
        # Aliases are request-side selectors for the one served model: later code sees MODEL_ID.
        given = "reasoning_effort" in self.model_fields_set
        if not given and self.thinking and self.thinking["type"] == "disabled":
            self.reasoning_effort, given = "low", True
        self.model, self.reasoning_effort = resolve_request_model(self.model, self.reasoning_effort, given)
        return self


def render_prompt(template, messages, effort, tools=None):
    for message in messages:
        if isinstance(message["content"], list):
            if any(part.get("type") != "text" or not isinstance(part.get("text"), str)
                   for part in message["content"]):
                raise ValueError("Only text message content is supported")
    return template.render(messages=messages, tools=tools, add_generation_prompt=True,
                           reasoning_effort=effort)


class ReasoningSplitter:
    """Separate template-prefilled <think> output, even across token boundaries."""
    def __init__(self):
        self.thinking = True
        self.pending = ""

    def feed(self, text, final=False):
        if not self.thinking:
            return {"content": text} if text else {}
        self.pending += text
        marker = "</think>"
        if marker in self.pending:
            thought, content = self.pending.split(marker, 1)
            self.pending = ""
            self.thinking = False
            return {key: value for key, value in
                    (("reasoning_content", thought), ("content", content)) if value}
        held = 0
        if not final:
            for count in range(1, len(marker)):
                if self.pending.endswith(marker[:count]):
                    held = count
        emit = self.pending[:-held] if held else self.pending
        self.pending = self.pending[-held:] if held else ""
        return {"reasoning_content": emit} if emit else {}


@asynccontextmanager
async def lifespan(app):
    from exllamav3 import AsyncGenerator, GreedySampler, model_init
    args = runtime["args"]
    serving_chunk_size = prefill_chunk_size(args)
    concurrency = concurrent_options(args, serving_chunk_size)
    tp_opt_in = tp_dflash_options(args)
    retention = session_options(args)
    b2_retention = b2_retention_options(args, concurrency, retention)
    from glm_b2_options import context_cap_option, cache_geometry
    context_cap = context_cap_option(args)
    draft_options = DFlashOptions.from_environment()
    if concurrency.enabled and draft_options.adaptive:
        raise ValueError("Concurrent initial profile uses fixed K7; adaptive batching needs separate qualification")
    if getattr(args, "q8_staging", False):
        from glm_q8_staging import install
        install(max_batch_size=concurrency.max_sequences)
    if getattr(args, "q8_head_gpu0", False):
        from glm_q8_layout import install
        install(reserve_head=bool(args.draft_model_dir) and not args.mtp)
    if getattr(args, "fp16_draft_cache", False):
        from glm_q8_layout import keep_mtp_cache_fp16
        keep_mtp_cache_fp16()
    from glm_vision_options import vision_options
    vision = vision_options(args)
    dflash = bool(args.draft_model_dir) and not args.mtp
    if not dflash and (draft_options.adaptive or draft_options.record_stats):
        raise ValueError("DFlash diagnostics/options require a DFlash2 drafter")
    if dflash:
        from glm_dflash import bounded_draft_cache, pin_drafter
        if concurrency.enabled:
            from glm_concurrent_ring import install_concurrent_rings
            install_concurrent_rings(concurrency.max_sequences)
        else:
            bounded_draft_cache()
        pin_drafter(0 if getattr(args, "q8_head_gpu0", False) else 1)
    method = "DFlash2 EXL3 6bpw K7" if dflash else "MTP d2"
    print(f"Loading GLM: GPU 0+1, full VRAM, {method} + k_hcfuse", flush=True)
    # Hybrid drafting: a hybrid draft verifies up to H + 1 rows, so the recurrent (KDA) history and the k_idx ring check
    # must see H; min_draft_len is the existing model_init hook for that. Unset, the call is unchanged
    hybrid = getattr(args, "hybrid_draft_tokens", 0) or 0
    model, config, cache, tokenizer, draft_model, _, draft_cache = model_init.init(
        args, quiet=True, progress=False, **({"min_draft_len": hybrid} if hybrid else {}))
    async with tp_lifecycle(tp_opt_in, model, draft_model) as owned_tp:
        if tp_opt_in:
            if (not model.loaded_tp or draft_model is None
                    or getattr(draft_model.config, "experimental_tensor_parallel", False) is not True):
                raise ValueError("The engine did not enable the requested typed TP DFlash2 configuration")
        sys.path.insert(0, str(KERNELS_DIR))
        import k_hcfuse
        k_hcfuse.install(model)
        if dflash:
            if not draft_model.caps.get("dflash_draft") or draft_model.config.block_size != 8:
                raise ValueError("This API requires a DFlash2 block-size-8 drafter")
            from glm_dflash import preserve_fused_exports
            preserve_fused_exports(model)
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                           extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = lambda value, **kwargs: json.dumps(value, **kwargs)
        template = env.from_string((Path(args.model_dir) / "chat_template.jinja").read_text())
        runtime.update(tokenizer=tokenizer, config=config, template=template,
                       lock=CapacityGate(concurrency.max_sequences) if concurrency.enabled else asyncio.Lock(),
                       concurrency=concurrency, b2_retention=b2_retention, admission=RequestAdmission(REQUEST_ADMISSION_POLICY), loaded_at=int(time.time()),
                       context_cap=context_cap, cache_geometry=cache_geometry(cache),
                       # --context-cap decouples the per-request limit from the page pool
                       context_length=min(config.max_position_embeddings,
                                          args.cache_size - CONTEXT_RESERVE if context_cap is None else context_cap))
        from glm_async import responsive_generator_class
        generator_class = responsive_generator_class(AsyncGenerator)
        if concurrency.enabled:
            from glm_concurrent_async import concurrent_generator_class
            generator_class = concurrent_generator_class(generator_class)
            if b2_retention.enabled:
                from glm_b2_retention_coordinator import paired_async_generator_class
                generator_class = paired_async_generator_class(generator_class)
        runtime["generator"] = generator_class(
            model=model, cache=cache, tokenizer=tokenizer, sampler=GreedySampler(),
            draft_model=draft_model, draft_cache=draft_cache,
            num_draft_tokens=args.num_draft_tokens, max_batch_size=concurrency.max_sequences,
            max_chunk_size=serving_chunk_size, recurrent_checkpoint_interval=2048,
            recurrent_cache_size=retention.recurrent_bytes,
            **concurrency.generator_kwargs(),
            **draft_options.generator_kwargs(num_draft_tokens=args.num_draft_tokens))
        if hybrid:
            # Wraps this generator's DFlash drafting method only; iterate_gen and the engine files are untouched
            from exllamav3.generator.hybrid_draft import install_hybrid_drafting
            install_hybrid_drafting(runtime["generator"].generator, hybrid)
        if tp_opt_in:
            owned_tp["generator"] = runtime["generator"]
        if concurrency.enabled:
            from glm_concurrent_ring import bind_concurrent_rings
            bind_concurrent_rings(runtime["generator"].generator, concurrency.max_sequences,
                                  **({"allow_embeddings": True, "drop_embeddings": not vision.keep_embeddings}
                                     if vision.enabled else {}))
            if b2_retention.enabled:
                from glm_b2_retention_coordinator import bind_b2_retention
                paired = bind_b2_retention(runtime["generator"].generator,
                    max_bytes=b2_retention.paired_bytes,
                    max_checkpoints=b2_retention.max_checkpoints)
                if b2_retention.target_cpu_bytes:
                    from glm_b2_cpu_cache import attach_b2_target_tier
                    attach_b2_target_tier(paired, b2_retention.target_cpu_bytes)
                # Shared mode records the exact tier attached above as its epoch.
                if b2_retention.shared_prefix_tokens:
                    paired.configure_shared_chat(b2_retention.shared_prefix_tokens, template, tokenizer, render_prompt)
        runtime["draft_options"] = draft_options
        if retention.enabled:
            if tp_opt_in:
                from glm_tp_sessions import enable_session_cache
            else:
                from glm_dflash_sessions import enable_session_cache
            enable_session_cache(runtime["generator"].generator,
                                 max_bytes=retention.paired_bytes,
                                 max_checkpoints=retention.max_checkpoints,
                                 target_cpu_bytes=retention.target_cpu_bytes,
                                 namespace=str(Path(args.model_dir).resolve()))
            print("GLM DFlash2: bounded completed-session retention enabled; "
                  f"target CPU tier ceiling {retention.target_cpu_bytes} bytes (pins only on page eviction)", flush=True)
        if dflash and getattr(args, "dflash_prefix_cache", False):
            if tp_opt_in:
                from glm_tp_prefix import enable_prefix_cache
            else:
                from glm_dflash_prefix import enable_prefix_cache
            enable_prefix_cache(runtime["generator"].generator)
            print("GLM DFlash2: previous-request prefix cache enabled (paired host snapshots)", flush=True)
        if vision.enabled:
            from glm_vision import Vision, load_tower
            runtime["vision"] = Vision(vision, None, tokenizer)        # probes the decode sandbox before the tower loads
            runtime["vision"].tower = load_tower(config, vision)
            runtime["request_body"]["max_request_bytes"] += vision.body_allowance()
            app.description = app.description.replace(
                "Multimodal input and JSON schema output are unsupported.",
                "Images: base64 data: URLs in user messages (PNG, JPEG, WebP); JSON schema output is unsupported.")
            print(f"GLM vision: tower on cuda:{vision.device}; at most {vision.max_images} images and "
                  f"{vision.max_image_tokens} tokens per image; data: URLs only", flush=True)
        print(f"GLM ready: http://{getattr(args, 'host', '0.0.0.0')}:{args.port}/v1 · context {runtime['context_length']} "
              f"· cache {args.cache_size}", flush=True)
        try:
            yield
        finally:
            if not tp_opt_in:
                try:
                    await runtime["generator"].close()
                finally:
                    manager = getattr(runtime["generator"].generator, "_glm_dflash_prefix_cache", None)
                    if retention.enabled and manager is not None:
                        manager.close()


app = FastAPI(title="GLM-5.3-Flash EXL3 API", version="1.0", lifespan=lifespan,
              description="Text chat/completions, SSE streaming, separate reasoning_content. "
                          "Context length is reported by /health and /v1/models; "
                          "one active request; defaults: high reasoning, temperature 1, "
                          "top_p 0.95, max_tokens 32768 including reasoning. "
                          "Function calls are returned to the client for execution. "
                          "Multimodal input and JSON schema output are unsupported.")
app.add_middleware(RequestBodyLimit, state=runtime["request_body"])


@app.exception_handler(HTTPException)
async def http_error(request, error):
    return JSONResponse(status_code=error.status_code, content={"error": {
        "message": str(error.detail), "type": "invalid_request_error", "code": error.status_code}})


@app.exception_handler(RequestValidationError)
async def validation_error(request, error):
    message = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
    return JSONResponse(status_code=400, content={"error": {
        "message": message, "type": "invalid_request_error", "code": 400}})


@app.get("/", include_in_schema=False)
async def home():
    return RedirectResponse("/docs")


@app.get("/health")
async def health():
    if runtime["generator"].error:
        raise HTTPException(503, "Generation engine failed; restart the GLM service")
    dflash = (bool(getattr(runtime["args"], "draft_model_dir", None))
              and not getattr(runtime["args"], "mtp", False))
    generator = getattr(runtime["generator"], "generator", runtime["generator"])
    draft_cache = getattr(generator, "draft_cache", None)
    prefix_cache = getattr(generator, "_glm_dflash_prefix_cache", None)
    b2_cache = getattr(generator, "_glm_b2_retention", None)
    return {"status": "ok", "engine": "ExLlamaV3", "model": MODEL_ID, "context_length": runtime["context_length"],
            "cache_tokens": runtime["args"].cache_size,
            "context_cap": runtime.get("context_cap"),
            **runtime.get("cache_geometry", {}),
            "rank_page_bytes": getattr(getattr(b2_cache, "tier", None), "rank_slot_bytes", None),
            "max_chunk_size": generator.max_chunk_size,
            "load_chunk_size": runtime["args"].chunk_size,
            "prefill_chunk_cap": getattr(runtime["args"], "prefill_chunk_size", None),
            "recurrent_checkpoint_interval": generator.recurrent_checkpoint_interval,
            "cache_format": cache_format(),
            "q8_staging": getattr(runtime["args"], "q8_staging", False),
            "draft_cache_format": ("FP16" if getattr(runtime["args"], "fp16_draft_cache", False)
                                   else cache_format()),
            "output_gpu": 0 if getattr(runtime["args"], "q8_head_gpu0", False) else 1,
            "workspace_margin_mb": int(os.environ.get("EXL3_AUTOSPLIT_MARGIN_MB", "256")),
            "mtp_depth": 0 if dflash else 2,
            "speculative_method": "dflash2" if dflash else "mtp",
            "draft_num_tokens": getattr(runtime["args"], "num_draft_tokens", None) or 2,
            "hybrid_drafting": (generator._hybrid_drafting.stats()
                                if getattr(generator, "_hybrid_drafting", None) is not None else None),
            "experimental_tp_dflash2": tp_dflash_health(generator),
            "target_tp": bool(generator.model.loaded_tp),
            "draft_ring_tokens": getattr(draft_cache, "dflash_ring_tokens", None),
            "prompt_cache_reuse": not dflash or prefix_cache is not None or b2_cache is not None,
            "prompt_cache": b2_cache.stats() if b2_cache is not None else (prefix_cache.stats() if prefix_cache is not None else None),
            "target_cpu_cache_budget_bytes": runtime["b2_retention"].target_cpu_bytes if runtime["b2_retention"].enabled else getattr(runtime["args"], "target_cpu_cache_gib", 0) * 1024**3,
            "experimental_b2_session_cache": runtime["b2_retention"].enabled,
            "experimental_b2_shared_prefix": bool(runtime["b2_retention"].shared_prefix_tokens),
            "recurrent_cache_budget_bytes": generator.recurrent_cache_size,
            "dynamic_draft_tokens": runtime["draft_options"].adaptive,
            "record_draft_stats": runtime["draft_options"].record_stats,
            "draft_confidence": runtime["draft_options"].confidence,
            "experimental_concurrent_generation": runtime["concurrency"].enabled,
            "max_active_sequences": generator.max_batch_size,
            "active_sequences": len(generator.active_jobs),
            "pending_sequences": len(generator.pending_jobs),
            "admitted_requests": runtime["lock"].active if runtime["concurrency"].enabled else int(runtime["lock"].locked()),
            "concurrent_ring": generator._glm_concurrent_ring.stats() if runtime["concurrency"].enabled else None,
            "fair_gen_rounds": getattr(generator, "fair_gen_rounds", 1),
            "fair_chunk_size": getattr(generator, "fair_chunk_size", None),
            "busy": runtime["lock"].active > 0 if runtime["concurrency"].enabled else runtime["lock"].locked(),
            "request_queue": runtime["admission"].statistics(),
            **runtime.get("request_body", {}),
            **({"vision": runtime["vision"].health()} if runtime.get("vision") is not None else {}),
            "generation_defaults": GENERATION_DEFAULTS}


class TokenizeRequest(BaseModel):
    model: str = MODEL_ID
    prompt: str | None = None
    content: str | None = None

    @field_validator("model")
    @classmethod
    def resolve_model_alias(cls, value):
        return resolve_request_model(value, None, False)[0]


@app.post("/tokenize")
async def tokenize(body: TokenizeRequest, request: Request):
    if body.model not in (MODEL_ID, "glm", Path(runtime["args"].model_dir).name):
        raise HTTPException(404, "Unknown model")
    text = body.prompt if body.prompt is not None else body.content
    if text is None and runtime.get("vision") is not None:
        fields = await request.json()
        if isinstance(fields, dict) and fields.get("messages") is not None:
            return chat_token_length(fields)
    if text is None:
        raise HTTPException(400, "prompt or content is required")
    return {"token_length": runtime["tokenizer"].encode(text, encode_special_tokens=True).shape[-1]}


def chat_token_length(fields):
    """--experimental-vision: the exact prompt length of a chat body, images included, without running the tower."""
    allowed = ("model", "messages", "tools", "tool_choice", "parallel_tool_calls", "reasoning_effort", "thinking")
    try:
        body = CompletionRequest(**{key: value for key, value in fields.items() if key in allowed})
        tool_choice = (body.tool_choice.model_dump() if isinstance(body.tool_choice, NamedToolChoice)
                       else body.tool_choice)
        messages, tools = prepare_tools([m.model_dump(exclude_none=True) for m in body.messages or []],
                                        [t.model_dump(exclude_none=True) for t in body.tools or []],
                                        tool_choice, body.parallel_tool_calls)
        messages, images = runtime["vision"].rewrite(messages)
        prompt = render_prompt(runtime["template"], messages, body.reasoning_effort, tools)
        tokens = (runtime["vision"].count_tokens(prompt, images, runtime["tokenizer"]) if images
                  else runtime["tokenizer"].encode(prompt, encode_special_tokens=True).shape[-1])
    except (ValueError, KeyError) as error:     # pydantic's ValidationError is a ValueError
        raise HTTPException(400, str(error)) from error
    return {"token_length": tokens, "image_tokens": sum(part.tokens for part in images)}


@app.get("/v1/models")
async def models():
    def card(model, effort, alias_of=None):
        reasoning = {"supported_efforts": list(EFFORT_LEVELS), "accepted_names": dict(EFFORT_NAMES),
                     "default": effort, "thinking_can_be_disabled": False}
        if alias_of:
            reasoning["alias_of"] = alias_of
        return {"id": model, "object": "model", "created": runtime["loaded_at"], "owned_by": "local",
                "context_length": runtime["context_length"], "cache_format": cache_format(),
                "max_model_len": runtime["context_length"], "reasoning": reasoning,
                **({"input_modalities": ["text", "image"], "vision": runtime["vision"].options.limits()}
                   if runtime.get("vision") is not None else {})}
    return {"object": "list", "data": [
        card(MODEL_ID, GENERATION_DEFAULTS["reasoning_effort"]),
        *(card(name, effort, MODEL_ID) for name, effort in MODEL_ALIASES.items())]}


def cache_format():
    quant = getattr(runtime["args"], "cache_quant", None)
    return f"Q{quant}" if quant is not None else "FP16"


def stream_text(result, tokenizer):
    text = result.get("text", "")
    # EXL3's piece join can emit an interior replacement character when the next
    # token completes a Korean UTF-8 character and also contains printable text.
    # Decode the held token group together, keeping stop-string trimming intact.
    if ("\ufffd" in text and result.get("token_ids") is not None
            and result.get("eos_reason") != "stop_string"):
        decoded = tokenizer.decode(result["token_ids"], decode_special_tokens=True)
        return decoded[0] if isinstance(decoded, list) else decoded
    return text


async def embed_images(prompt, images, prompt_tokens, limit, context_length, request):
    """After admission: decode and encode the images, then the exact ids. A tower length that differs from the
    pre-admission estimate is counted and the context check repeated with the real length."""
    from glm_vision import ImageRejected, VisionFailure
    vision = runtime["vision"]
    try:
        embeddings = await vision.embed(images, runtime["generator"], request.is_disconnected)
    except ImageRejected as error:      # the client's bytes did not decode, or the client left
        raise HTTPException(400, str(error)) from error
    except VisionFailure as error:
        raise HTTPException(500, str(error)) from error
    ids = runtime["tokenizer"].encode(prompt, encode_special_tokens=True, embeddings=embeddings)
    if ids.shape[-1] != prompt_tokens:
        vision.mismatch(prompt_tokens, ids.shape[-1])
        prompt_tokens = ids.shape[-1]
        if prompt_tokens + limit > context_length:
            raise HTTPException(400, f"Prompt ({prompt_tokens}) + output ({limit}) exceeds context {context_length}; "
                                     f"this model's maximum context length is {context_length} tokens")
    return ids, embeddings, prompt_tokens


async def complete(body, request, chat):
    from exllamav3 import AsyncJob, ComboSampler, GreedySampler
    if body.store:
        raise HTTPException(400, "Stored completions are unsupported; use store=false")
    if body.model not in (MODEL_ID, "glm", Path(runtime["args"].model_dir).name):
        raise HTTPException(404, f"Unknown model: {body.model}")
    tools, images = [], []
    tool_choice = (body.tool_choice.model_dump() if isinstance(body.tool_choice, NamedToolChoice)
                   else body.tool_choice)
    if chat:
        if not body.messages or body.prompt is not None:
            raise HTTPException(400, "A nonempty messages list is required")
        try:
            messages, tools = prepare_tools(
                [m.model_dump(exclude_none=True) for m in body.messages],
                [t.model_dump(exclude_none=True) for t in body.tools or []],
                tool_choice, body.parallel_tool_calls)
            if runtime.get("vision") is not None:
                # Image parts become alias text first, so render_prompt still refuses any other media.
                messages, images = runtime["vision"].rewrite(messages)
            prompt = render_prompt(runtime["template"],
                                   messages, body.reasoning_effort, tools)
        except (ValueError, KeyError) as error:
            raise HTTPException(400, str(error)) from error
    else:
        if body.tools or tool_choice not in (None, "none", "auto"):
            raise HTTPException(400, "Function tools require /v1/chat/completions")
        if not body.prompt or body.messages is not None:
            raise HTTPException(400, "A nonempty string prompt is required")
        prompt = body.prompt
    if images:
        # Exact length before the tower runs, so an oversized request never takes a slot; ids are built after admission.
        try:
            ids, prompt_tokens = None, runtime["vision"].count_tokens(prompt, images, runtime["tokenizer"])
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
    else:
        ids = runtime["tokenizer"].encode(prompt, encode_special_tokens=True)
        prompt_tokens = ids.shape[-1]
    limit = body.max_completion_tokens or body.max_tokens
    context_length = runtime["context_length"]
    if (prompt_tokens + limit > context_length and context_length - prompt_tokens >= DEFAULT_OUTPUT_FLOOR
            and not {"max_tokens", "max_completion_tokens"} & body.model_fields_set):
        # The server's default output limit must not by itself fail a prompt that fits: use the room left (vLLM does).
        limit = context_length - prompt_tokens
        print(f"Output limit lowered to {limit} tokens to fit the context", flush=True)
    if body.min_tokens > limit:
        raise HTTPException(400, "min_tokens exceeds the output limit")
    if prompt_tokens + limit > context_length:
        # Keep the original text (qualification gates match it) and add the wording clients' overflow detectors look for.
        raise HTTPException(400, f"Prompt ({prompt_tokens}) + output ({limit}) exceeds context {context_length}; "
                                 f"this model's maximum context length is {context_length} tokens")
    stops = list(runtime["config"].eos_token_id_list)
    if not body.ignore_eos and not body.min_tokens:
        stops += ["<|user|>", "<|observation|>"]
    stops += [body.stop] if isinstance(body.stop, str) else (body.stop or [])
    if any(not stop for stop in stops):
        raise HTTPException(400, "Stop strings must not be empty")
    sampler = (GreedySampler() if body.temperature == 0 and body.frequency_penalty == 0
               and body.presence_penalty == 0 else ComboSampler(
                   temperature=body.temperature, top_p=body.top_p,
                   freq_p=body.frequency_penalty, pres_p=body.presence_penalty))
    lock = runtime["lock"]
    try:
        admission_wait = await runtime["admission"].acquire(lock, request)
    except AdmissionError as error:
        raise HTTPException(error.status, str(error))
    shared = None
    try:
        if runtime["generator"].error:
            raise HTTPException(503, "Generation engine failed; restart the service")
        if images:
            ids, embeddings, prompt_tokens = await embed_images(prompt, images, prompt_tokens, limit, context_length,
                                                                request)
        paired = getattr(runtime["generator"].generator, "_glm_b2_retention", None)
        shared = None if paired is None else paired.shared
        prepared = (shared.prepared_chat(ids, messages, tools, body.reasoning_effort, prompt,
                        template=runtime["template"], tokenizer=runtime["tokenizer"], renderer=render_prompt)
                    if chat and shared is not None and not images else nullcontext())
        if images and shared is not None:
            # Image jobs take the ordinary paired-retention path until the shared prefix is qualified with them.
            runtime["vision"].counters["shared_prefix_skipped"] += 1
        with prepared:
            job = AsyncJob(runtime["generator"], input_ids=ids, max_new_tokens=limit,
                           min_new_tokens=limit if body.ignore_eos else body.min_tokens,
                           stop_conditions=stops, sampler=sampler, seed=body.seed,
                           decode_special_tokens=True, **({"embeddings": embeddings} if images else {}))
    except BaseException as error:
        lock.release()
        if shared is not None:
            from glm_b2_retention_coordinator import StopSharedGeneration
            if isinstance(error, StopSharedGeneration):
                raise runtime["generator"].latch_shared(error) from error
        raise
    ident = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
    created = int(time.time())
    splitter = ReasoningSplitter() if chat else None
    strict = names_strict(UNADVERTISED_TOOL_CALLS, tool_choice)
    # Non-strict: a model deviation in a call (unparseable, unterminated, extra) becomes text/is dropped instead of failing the stream.
    tool_splitter = (ToolSplitter(tools, body.parallel_tool_calls, strict_names=strict, strict_calls=strict)
                     if tools else None)
    collected = {"content": "", "reasoning_content": ""}
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": 0, "total_tokens": prompt_tokens}
    reason = "stop"
    timings = {"http_queue_ms": admission_wait * 1000}

    def envelope(choices, streaming=False):
        return {"id": ident, "object": ("chat.completion.chunk" if streaming else "chat.completion")
                if chat else "text_completion", "created": created, "model": MODEL_ID, "choices": choices}

    def event(payload):
        return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    cleanup = OwnedJobCleanup(job.cancel, lock.release)

    async def results():
        nonlocal reason
        try:
            if body.stream and chat:
                yield envelope([{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}], True)
            async for result in job:
                if await request.is_disconnected():
                    return
                if result["stage"] != "streaming":
                    continue
                text = stream_text(result, runtime["tokenizer"])
                delta = splitter.feed(text, final=result["eos"]) if chat else {"content": text}
                deltas = []
                if "reasoning_content" in delta:
                    deltas.append({"reasoning_content": delta["reasoning_content"]})
                if tool_splitter:
                    deltas.extend(tool_splitter.feed(delta.get("content", ""), final=result["eos"],
                                  truncated=result.get("eos_reason") == "max_new_tokens"))
                elif "content" in delta:
                    deltas.append({"content": delta["content"]})
                for delta in deltas:
                    for key in ("content", "reasoning_content"):
                        collected[key] += delta.get(key, "")
                    if body.stream:
                        choice = {"index": 0, "finish_reason": None}
                        choice.update({"delta": delta} if chat else {"text": delta.get("content", "")})
                        yield envelope([choice], True)
                if result["eos"]:
                    reason = "length" if result.get("eos_reason") == "max_new_tokens" else "stop"
                    usage["completion_tokens"] = int(result["new_tokens"])
                    usage["total_tokens"] = prompt_tokens + usage["completion_tokens"]
                    for native, public in [("time_prefill", "prompt_ms"),
                                           ("time_generate", "predicted_ms"),
                                           ("time_enqueued", "queue_ms")]:
                        if result.get(native) is not None:
                            timings[public] = result[native] * 1000
                    cached_tokens = min(prompt_tokens, max(0, int(result.get("cached_tokens", 0))))
                    usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
                    timings.update(prompt_n=prompt_tokens, predicted_n=usage["completion_tokens"],
                                   source="exllamav3", cached_tokens=cached_tokens, cache_n=cached_tokens)
                    if runtime["draft_options"].record_stats:
                        timings["draft_stats"] = summarize_draft_stats(job.job.draft_stats)
                    if result.get("accepted_draft_tokens") is not None:
                        accepted = result["accepted_draft_tokens"]
                        drafted = accepted + result.get("rejected_draft_tokens", 0)
                        usage.update(draft_tokens=drafted, draft_tokens_accepted=accepted)
                        timings.update(draft_n=drafted, draft_n_accepted=accepted)
            if tool_splitter and tool_splitter.calls and reason != "length":
                reason = "tool_calls"
            if reason != "length" and (tool_choice == "required" or isinstance(tool_choice, dict)):
                if not tool_splitter or not tool_splitter.calls:
                    raise ValueError("Model did not produce the requested function call")
            if body.stream:
                choice = {"index": 0, "finish_reason": reason}
                choice.update({"delta": {}} if chat else {"text": ""})
                yield {**envelope([choice], True), "timings": timings}
                if (body.stream_options or {}).get("include_usage"):
                    yield {**envelope([], True), "usage": usage}
        finally:
            await cleanup.close()

    if body.stream:
        async def stream():
            try:
                async for payload in results():
                    yield event(payload)
                yield "data: [DONE]\n\n"
            except Exception as error:
                print(f"Generation failed: {error!r}", flush=True)
                traceback.print_exception(error)
                oom = "out of memory" in str(error).lower()
                yield event({"error": {"message": "GPU memory exhausted; see service log" if oom
                                       else "Generation failed; see service log",
                                       "type": "server_error", "code": "out_of_memory" if oom else "generation_failed"}})
        return OwnedStreamingResponse(stream(), cleanup=cleanup, media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    try:
        async for _ in results():
            pass
    except Exception as error:
        print(f"Generation failed: {error!r}", flush=True)
        traceback.print_exception(error)
        raise HTTPException(500, "Generation failed; see service log") from error
    choice = {"index": 0, "finish_reason": reason}
    choice.update({"message": {"role": "assistant", **collected}} if chat
                  else {"text": collected["content"], "logprobs": None})
    if chat and tool_splitter and tool_splitter.calls:
        choice["message"]["tool_calls"] = tool_splitter.calls
    return {**envelope([choice]), "usage": usage, "timings": timings}


async def complete_reporting_effort(body, request, response, chat):
    result = await complete(body, request, chat)
    # A streaming reply is a Response itself; FastAPI merges `response` only into plain results.
    (result if isinstance(result, Response) else response).headers["X-Reasoning-Effort"] = body.reasoning_effort
    return result


@app.post("/v1/chat/completions")
async def chat_completions(body: CompletionRequest, request: Request, response: Response):
    return await complete_reporting_effort(body, request, response, chat=True)


@app.post("/v1/completions")
async def completions(body: CompletionRequest, request: Request, response: Response):
    return await complete_reporting_effort(body, request, response, chat=False)


if __name__ == "__main__":
    from exllamav3 import model_init
    import uvicorn
    parser = argparse.ArgumentParser()
    from glm_vision_options import add_vision_arguments, vision_options
    model_init.add_args(parser, cache=True, add_draft_model_args=True, default_chunk_size=2048)
    add_vision_arguments(parser)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--host", default="0.0.0.0", help="Bind address; use 127.0.0.1 for private qualification")
    parser.add_argument("--prefill-chunk-size", type=int,
                        default=os.environ.get("GLM53_PREFILL_CHUNK_SIZE"),
                        help="Cap serving prefill chunks independently of loader/autosplit --chunk_size "
                             "(default: GLM53_PREFILL_CHUNK_SIZE when set)")
    parser.add_argument("--q8-staging", action="store_true",
                        help="Bound packed-pool staging copies for the Q8 profile")
    parser.add_argument("--q8-head-gpu0", action="store_true",
                        help="Keep the output projection resident on GPU 0")
    parser.add_argument("--fp16-draft-cache", action="store_true",
                        help="Keep the MTP draft layer cache FP16")
    parser.add_argument("--dflash-prefix-cache", action="store_true",
                        help="Opt in to previous-request DFlash prefix reuse with paired host snapshots")
    parser.add_argument("--dflash-session-cache", action="store_true",
                        help="Experimental bounded paired prefix retention (LS or explicitly opted-in TP; one active request)")
    parser.add_argument("--target-cpu-cache-gib", type=int, default=0,
                        help="Target-only idle page spill ceiling in GiB; 0 disables; pins page slabs only on eviction")
    parser.add_argument("--recurrent-cache-gib", type=int, default=4,
                        help="Native recurrent checkpoint ceiling in GiB (default unchanged: 4)")
    parser.add_argument("--session-cache-gib", type=int, default=1,
                        help="Combined paired native checkpoint/draft snapshot retention ceiling in GiB")
    parser.add_argument("--session-cache-max-checkpoints", type=int, default=8,
                        help="Maximum retained paired checkpoint boundaries (at least 2)")
    parser.add_argument("--experimental-concurrent-generation", action="store_true",
                        help="Opt in to TP DFlash2 continuous batching with isolated cold draft rings")
    parser.add_argument("--concurrent-max-sequences", type=int)
    parser.add_argument("--concurrent-fair-gen-rounds", type=int)
    parser.add_argument("--concurrent-fair-chunk-size", type=int)
    parser.add_argument("--experimental-b2-session-cache", action="store_true",
                        help="Experimental two-job paired retention; exact full384Ki TP/K7 profile only")
    parser.add_argument("--experimental-b2-shared-prefix", action="store_true",
                        help="Experimental exact system/tool prefix reuse inside the bounded B2 paired owner")
    parser.add_argument("--b2-shared-prefix-tokens", type=int, default=None,
                        help="Shared page cap, default65536; requires explicit B2 shared flag")
    parser.add_argument("--b2-target-cpu-cache-gib", type=int,
                        help="B2-owned lazy inactive target page overflow GiB ceiling;0 disables, maximum8")
    parser.add_argument("--hybrid-draft-tokens", dest="hybrid_draft_tokens", type=int, default=0,
                        help="Hybrid drafting: when a sequence's suffix matches at least this many tokens of its "
                             "earlier context, verify that many tokens of the continuation (on top of DFlash2 K7); "
                             "0 disables, else 8..15")
    parser.add_argument("--context-cap", type=int, default=None,
                        help="Per-request context ceiling, decoupled from the page pool (-cs); at most 392960. "
                             "A B2 pool above 393216 requires it and the engine k_idx ring (EXL3_MLA_KIDX_RING)")
    args = parser.parse_args()
    try:
        concurrency = concurrent_options(args, prefill_chunk_size(args))
        tp_dflash_options(args)
        retention = session_options(args)
        b2_retention_options(args, concurrency, retention)
        from glm_b2_options import context_cap_option
        context_cap_option(args)
        DFlashOptions.from_environment()
    except ValueError as error:
        parser.error(str(error))
    if not ((args.mtp and not args.draft_model_dir and args.num_draft_tokens == 2)
            or (not args.mtp and args.draft_model_dir and args.num_draft_tokens == 7)):
        parser.error("Use either --mtp -ndt 2 or -dm <DFlash2 EXL3 6bpw> -ndt 7")
    # Hybrid drafting qualified envelope: DFlash2 K7, fixed window, 8..15 tokens (a verify pass of at most 16 rows: the CUDA-graph
    # decode bound and the k_idx ring's KIDX_RING_MAX_QLEN; 7 or fewer cannot improve on K7)
    if args.hybrid_draft_tokens:
        import importlib.util
        if (args.mtp or not args.draft_model_dir or not 8 <= args.hybrid_draft_tokens <= 15
                or DFlashOptions.from_environment().adaptive):
            parser.error("--hybrid-draft-tokens requires -dm DFlash2 -ndt 7, GLM53_DYNAMIC_DRAFT=0 and a value in 8..15")
        if importlib.util.find_spec("exllamav3.generator.hybrid_draft") is None:
            parser.error("--hybrid-draft-tokens requires the engine module exllamav3/generator/hybrid_draft.py")
    if (args.q8_staging or args.q8_head_gpu0 or args.fp16_draft_cache) and args.cache_quant != "8":
        parser.error("Q8 memory options require -cq 8")
    if args.dflash_prefix_cache and args.mtp:
        parser.error("--dflash-prefix-cache requires DFlash2; MTP already uses native prefix caching")
    if args.q8_staging and args.autosplit_max_batch_size != 1 and not args.experimental_concurrent_generation:
        parser.error("--q8-staging requires --autosplit_max_batch_size 1")
    try:
        vision_options(args)
    except ValueError as error:
        parser.error(str(error))
    runtime["args"] = args
    # k_hcfuse gates itself on this flag; glm_api always calls install() explicitly.
    os.environ.setdefault(HCFUSE_ENV, "1")
    uvicorn.run(app, host=args.host, port=args.port, timeout_graceful_shutdown=5)
