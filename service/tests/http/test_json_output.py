"""JSON contract, filter ownership and real ASGI responses without a model."""
import asyncio
import contextlib
import json
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import httpx
from pydantic import ValidationError

import glm_json_output as jo


class FakeLLGuidance:
    def __init__(self, tokenizer, trigger_token=None, eos_after_completed=False, json_schema=None):
        self.tokenizer, self.trigger_token = tokenizer, trigger_token
        self.eos_after_completed, self.prefix_str = eos_after_completed, None
        self._consume_prefix = False
        self._grammar = {"json_schema": json_schema}


class Tokenizer:
    def single_id(self, text):
        assert text == "</think>"
        return 42

    def encode(self, text, **kwargs):
        return NS(shape=(1, 12))


class Job:
    pass


class PolicyTests(unittest.TestCase):
    def body(self, **kwargs):
        return NS(response_format=jo.ResponseFormat(type="json_object"), tools=None,
                  tool_choice=None, stop=None, ignore_eos=False, min_tokens=0, **kwargs)

    def test_only_explicit_supported_contract_is_enabled(self):
        body = self.body()
        self.assertTrue(jo.requested(body, enabled=True, chat=True))
        for change in ({"tools": [object()]}, {"stop": ["}"]}, {"ignore_eos": True},
                       {"min_tokens": 1}, {"tool_choice": "required"}):
            other = self.body(); other.__dict__.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                jo.requested(other, enabled=True, chat=True)
        for enabled, chat in [(False, True), (True, False)]:
            with self.assertRaises(ValueError):
                jo.requested(body, enabled=enabled, chat=chat)
        for fmt in [None, jo.ResponseFormat(type="text")]:
            body.response_format = fmt
            self.assertFalse(jo.requested(body, enabled=False, chat=True))
        for value in [{"type": "json_schema"}, {"type": "json_object", "extra": True}]:
            with self.assertRaises(ValidationError):
                jo.ResponseFormat(**value)

    def test_completed_contract_rejects_fences_trailing_text_arrays_and_nan(self):
        jo.validate_complete(' {"code":"a\\nb", "x":[null,true,1]} ')
        for text in ['```json\n{}\n```', '{} extra', '[]', '{"x":NaN}', '{']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                jo.validate_complete(text)

    def test_only_issued_filter_configuration_and_unique_job_owner_are_allowed(self):
        upstream = NS(LLGuidanceFilter=FakeLLGuidance)
        jo.filter_type.cache_clear()
        self.addCleanup(jo.filter_type.cache_clear)
        with patch.dict(sys.modules, {"exllamav3.generator.filter.llguidance": upstream}):
            tokenizer, job, other = Tokenizer(), Job(), Job()
            first, second = jo.build_filter(tokenizer), jo.build_filter(tokenizer)
            self.assertIsNot(first, second)
            self.assertTrue(jo.permitted_filters([first], tokenizer, job))
            self.assertTrue(jo.permitted_filters([first], tokenizer, job))
            self.assertFalse(jo.permitted_filters([first], tokenizer, other))
            self.assertTrue(jo.permitted_filters([second], tokenizer, other))
            self.assertFalse(jo.permitted_filters([FakeLLGuidance(tokenizer)], tokenizer, job))
            self.assertFalse(jo.permitted_filters([first, second], tokenizer, job))
            second._grammar["json_schema"]["type"] = "array"
            self.assertFalse(jo.permitted_filters([second], tokenizer, other))
            self.assertFalse(jo.permitted_filters([first], Tokenizer(), job))


class HTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import glm_api as api
        from glm_admission import RequestAdmission, AdmissionPolicy
        self.api = api
        self.original = api.runtime.copy()
        self.addCleanup(lambda: (api.runtime.clear(), api.runtime.update(self.original)))
        self.jobs = []
        self.text = 'thinking</think>{"code":"def add(a,b): return a+b"}'
        self.finish = "stop"
        self.events = None
        owner = self

        class AsyncJob:
            def __init__(self, generator, **kwargs):
                self.kwargs, self.cancelled = kwargs, False
                owner.jobs.append(self)

            async def cancel(self):
                self.cancelled = True

            async def __aiter__(self):
                await asyncio.sleep(0)
                if owner.events is not None:
                    for event in owner.events:
                        yield {"stage": "streaming", "eos": False, **event}
                    return
                yield {"stage": "streaming", "text": owner.text, "eos": True,
                       "eos_reason": "max_new_tokens" if owner.finish == "length" else "stop",
                       "new_tokens": 37, "time_generate": .1, "time_prefill": .01}

        fake = NS(AsyncJob=AsyncJob, GreedySampler=lambda: None, ComboSampler=lambda **kwargs: None)
        self.module_patch = patch.dict(sys.modules, {"exllamav3": fake})
        self.module_patch.start(); self.addCleanup(self.module_patch.stop)
        self.factory = patch.object(jo, "build_filter", side_effect=lambda tokenizer: object())
        self.factory.start(); self.addCleanup(self.factory.stop)
        self.lock = asyncio.Lock()
        api.runtime.update(args=NS(model_dir="/model"), json_mode=True, tokenizer=Tokenizer(),
                           template=NS(render=lambda **kwargs: "<think>"),
                           config=NS(eos_token_id_list=[2]), context_length=8192,
                           lock=self.lock, generator=NS(error=None, generator=NS()),
                           admission=RequestAdmission(AdmissionPolicy()), draft_options=NS(record_stats=False))
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test")
        self.addAsyncCleanup(self.client.aclose)

    async def post(self, **kwargs):
        body = {"messages": [{"role": "user", "content": "Return JSON."}], "max_tokens": 128,
                "temperature": 0, "response_format": {"type": "json_object"}, **kwargs}
        return await self.client.post("/v1/chat/completions", json=body)

    async def test_actual_nonstream_response_keeps_usage_and_passes_filter_to_job(self):
        response = await self.post()
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(json.loads(data["choices"][0]["message"]["content"]),
                         {"code": "def add(a,b): return a+b"})
        self.assertEqual(data["usage"]["completion_tokens"], 37)
        self.assertEqual(len(self.jobs[0].kwargs["filters"]), 1)
        self.assertTrue(self.jobs[0].cancelled)
        self.assertFalse(self.lock.locked())

    async def test_actual_streamed_json_and_unconstrained_text_are_distinct(self):
        response = await self.post(stream=True, stream_options={"include_usage": True})
        events = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith("data: {")]
        content = ''.join(c.get("delta", {}).get("content", "") for e in events for c in e.get("choices", []))
        self.assertEqual(json.loads(content), {"code": "def add(a,b): return a+b"})
        self.assertEqual(events[-1]["usage"]["completion_tokens"], 37)
        self.text = 'thinking</think>```json\n{}\n```'
        response = await self.post(response_format={"type": "text"})
        self.assertEqual(response.json()["choices"][0]["message"]["content"], '```json\n{}\n```')
        self.assertNotIn("filters", self.jobs[-1].kwargs)

    async def test_continuous_usage_counts_native_tokens_once_across_split_channels(self):
        self.events = [
            {"text": "thinking", "token_ids": NS(numel=lambda: 3)},
            {"text": "</think>{\"x\":1}", "token_ids": NS(numel=lambda: 5)},
            {"text": "", "eos": True, "new_tokens": 9, "eos_reason": "stop"},
        ]
        response = await self.post(stream=True, stream_options={
            "include_usage": True, "continuous_usage_stats": True})
        events = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith("data: {")]
        self.assertFalse(any("error" in e for e in events), response.text)
        counts = [e["usage"]["completion_tokens"] for e in events]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(set(counts), {0, 3, 8, 9})
        self.assertEqual(counts[-1], 9)
        self.assertTrue(all(e["usage"]["total_tokens"] == e["usage"]["prompt_tokens"] + n
                            for e, n in zip(events, counts)))
        self.assertLess(sum(bool(e.get("choices")) for e in events), counts[-1])
        self.assertTrue(self.jobs[-1].cancelled)
        self.assertFalse(self.lock.locked())

    async def test_continuous_usage_counts_once_when_one_event_feeds_both_channels(self):
        # One engine event whose text crosses </think> yields a reasoning delta AND a content
        # delta; its native tokens must still be counted once, not once per delta.
        self.events = [
            {"text": "ab</think>{\"x\"", "token_ids": NS(numel=lambda: 4)},
            {"text": ":1}", "token_ids": NS(numel=lambda: 2)},
            {"text": "", "eos": True, "new_tokens": 7, "eos_reason": "stop"},
        ]
        response = await self.post(stream=True, stream_options={
            "include_usage": True, "continuous_usage_stats": True})
        events = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith("data: {")]
        self.assertFalse(any("error" in e for e in events), response.text)
        deltas = [c["delta"] for e in events for c in e.get("choices", []) if c.get("delta")]
        self.assertIn({"reasoning_content": "ab"}, deltas)
        self.assertIn({"content": "{\"x\""}, deltas)
        counts = [e["usage"]["completion_tokens"] for e in events]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(set(counts), {0, 4, 6, 7})

    async def test_continuous_usage_requires_include_usage_and_keeps_legacy_final_only(self):
        for options in ({"continuous_usage_stats": True}, {"include_usage": True}, {}):
            response = await self.post(stream=True, stream_options=options)
            events = [json.loads(line[6:]) for line in response.text.splitlines()
                      if line.startswith("data: {")]
            usage_events = [e for e in events if "usage" in e]
            self.assertEqual(len(usage_events), int(bool(options.get("include_usage"))))
            if usage_events:
                self.assertEqual(usage_events[0]["usage"]["completion_tokens"], 37)

    async def test_invalid_model_output_is_an_error_with_cleanup_not_silently_repaired(self):
        self.text = 'thinking</think>```json\n{}\n```'
        response = await self.post(stream=True)
        self.assertIn('"generation_failed"', response.text)
        self.assertNotIn('"finish_reason": "stop"', response.text)
        self.assertTrue(self.jobs[-1].cancelled)
        self.assertFalse(self.lock.locked())

    async def test_clear_thinking_reaches_the_template_only_when_a_client_asks(self):
        seen = []
        self.api.runtime["template"] = NS(render=lambda **kwargs: seen.append(kwargs) or "<think>")
        response = await self.post(response_format={"type": "text"},
                                   chat_template_kwargs={"clear_thinking": False})
        self.assertEqual(response.status_code, 200, response.text)
        response = await self.post(response_format={"type": "text"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIs(seen[0]["clear_thinking"], False)
        self.assertNotIn("clear_thinking", seen[1])

    async def test_tokenize_counts_a_chat_with_the_switch_it_will_be_rendered_with(self):
        seen = []
        self.api.runtime["template"] = NS(render=lambda **kwargs: seen.append(kwargs) or "<think>")
        self.api.runtime["vision"] = NS(rewrite=lambda messages: (messages, []))
        body = {"messages": [{"role": "user", "content": "Return JSON."}],
                "chat_template_kwargs": {"clear_thinking": False}}
        response = await self.client.post("/tokenize", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIs(seen[-1]["clear_thinking"], False)
        response = await self.client.post("/tokenize", json={**body, "chat_template_kwargs": {"x": 1}})
        self.assertEqual(response.status_code, 400, response.text)

    async def test_the_shared_prefix_gets_the_exact_bound_renderer_with_or_without_the_switch(self):
        """The shared prefix binds (template, tokenizer, renderer) at startup and refuses any other
        object, so a wrapped renderer failed every chat request in production (2026-10-17). Its
        boundary is checked against the request's actual tokens, so the switch need not reach it."""
        api, calls = self.api, []

        class Shared:
            @contextlib.contextmanager
            def prepared_chat(self, ids, messages, tools, effort, prompt, *, template, tokenizer, renderer):
                if template is not api.runtime["template"] or renderer is not api.render_prompt:
                    raise ValueError("One exact bound API render scope required")
                calls.append(effort)
                yield

        api.runtime["generator"] = NS(error=None, generator=NS(_glm_b2_retention=NS(shared=Shared())))
        for switches in (None, {"clear_thinking": False}):
            response = await self.post(response_format={"type": "text"}, chat_template_kwargs=switches)
            self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(calls), 2)

    async def test_an_unknown_or_loose_template_switch_is_refused_before_a_job(self):
        for switches in ({"enable_thinking": False}, {"clear_thinking": "false"}, {"clear_thinking": 0}):
            response = await self.post(response_format={"type": "text"}, chat_template_kwargs=switches)
            self.assertEqual(response.status_code, 400, (switches, response.text))
        self.assertEqual(self.jobs, [])

    async def test_disabled_or_conflicting_json_requests_do_not_allocate_a_job(self):
        self.api.runtime["json_mode"] = False
        response = await self.post()
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.jobs, [])
        self.api.runtime["json_mode"] = True
        for change in [{"stop": "}"}, {"min_tokens": 1}, {"ignore_eos": True},
                       {"response_format": {"type": "json_schema"}}]:
            response = await self.post(**change)
            self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.jobs, [])

    async def test_length_finish_remains_explicit_and_does_not_claim_complete_json(self):
        self.text, self.finish = 'thinking</think>{"code":', "length"
        response = await self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["finish_reason"], "length")
        self.assertFalse(self.lock.locked())


if __name__ == "__main__":
    unittest.main()
