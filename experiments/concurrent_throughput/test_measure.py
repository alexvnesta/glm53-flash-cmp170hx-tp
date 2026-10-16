import asyncio
import copy
import json
import unittest
import time
import subprocess
from unittest.mock import patch

import httpx

import measure


def snapshot():
    return {
        "status": "ok", "busy": False, "max_active_sequences": 2,
        "native_extension": {"sha256": "native"}, "profile_digest": "profile",
        "request_metrics": {"active": [], "recent": []},
        "request_queue": {"waiting": 0}, "switcher": {"inflight": 0},
    }


class GuardTests(unittest.TestCase):
    def test_four_slots_require_explicit_capacity_and_keep_idle_guards(self):
        h = snapshot(); h['max_active_sequences'] = 4
        with self.assertRaises(RuntimeError): measure.require_idle(h)
        measure.require_idle(h, 4)
        h['busy'] = True
        with self.assertRaises(RuntimeError): measure.require_idle(h, 4)
        for n in (True, 3, 8):
            with self.assertRaises(ValueError): measure.require_idle(snapshot(), n)

    def test_explicit_json_cohort_is_separate_and_legacy_fixture_is_unchanged(self):
        legacy = measure.cases(16, "mixed")
        fixed = measure.cases(16, "mixed_json")
        self.assertEqual(legacy[0]["prompt"], fixed[0]["prompt"])
        self.assertNotIn("response_format", legacy[0])
        self.assertEqual(fixed[0]["response_format"], {"type": "json_object"})
        self.assertEqual(fixed[1]["expected"], {"values": list(range(400, 416))})

    def test_nested_parallel_failures_preserve_failed_answer_for_diagnosis(self):
        response = {"answer": '{"code":"def compress_runs(values): return []"}',
                    "case": "compress-runs-code", "correct": False}
        error = ExceptionGroup("parallel wave", [
            ExceptionGroup("worker", [measure.ProbeResponseFailure("Known-answer failure", response)])])
        saved = json.loads(json.dumps(measure.failure_details(error)))
        leaf = saved["exceptions"][0]["exceptions"][0]
        self.assertEqual(leaf["response"], response)
        self.assertEqual(leaf["type"], "ProbeResponseFailure")

    def test_busy_admission_and_missing_identity_refuse_load(self):
        for key, value in (("busy", True), ("max_active_sequences", 4),
                           ("profile_digest", None)):
            h = snapshot()
            h[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                measure.require_idle(h)
        h = snapshot()
        h["request_metrics"]["active"] = [{"id": "someone-else"}]
        with self.assertRaises(RuntimeError):
            measure.require_idle(h)

    def test_batch_proof_requires_both_draft_and_verifier_counter_deltas(self):
        serial = {
            "batch_finish_s": 8, "requests": [{"correct": True, "usage": {"completion_tokens": 100}}],
            "server_metrics": [{"cached_tokens": 0}],
        }
        parallel = copy.deepcopy(serial)
        parallel.update(batch_finish_s=5, ring_before={"batched_verifier_rounds": 2, "batched_draft_rounds": 2},
                        ring_after={"batched_verifier_rounds": 3, "batched_draft_rounds": 2})
        result = measure.compare(serial, parallel)
        self.assertEqual(result["batch_speedup"], 1.6)
        self.assertEqual(result["serial_aggregate_completion_tokens_per_s"], 12.5)
        self.assertEqual(result["concurrent_aggregate_completion_tokens_per_s"], 20)
        self.assertFalse(result["real_draft_and_target_batch_observed"])
        parallel["ring_after"]["batched_draft_rounds"] = 3
        self.assertTrue(measure.compare(serial, parallel)["real_draft_and_target_batch_observed"])


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_brief_own_registration_wait_does_not_misidentify_other_traffic(self):
        owned, samples = measure.OwnedRequests(), []
        owned.pending["our-case"] = time.monotonic()

        async def registering(_client):
            h = snapshot()
            h["request_metrics"]["active"] = [{"id": "new-id"}]
            h["request_queue"]["waiting"] = 1
            return h

        with patch.object(measure, "health", registering):
            task = asyncio.create_task(measure.monitor(None, snapshot(), owned, samples))
            await asyncio.sleep(.02)
            self.assertFalse(task.done())
            self.assertEqual(len(samples), 1)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def client_response(self, answer, usage=True):
        events = [
            {"id": "our-request", "choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"reasoning_content": "check"}}]},
            {"choices": [{"delta": {"content": answer}, "finish_reason": "stop"}]},
        ]
        if usage:
            events.append({"choices": [], "usage": {"completion_tokens": 97}})
        wire = "".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n"
        transport = httpx.MockTransport(lambda _: httpx.Response(200, text=wire))
        owned = set()
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            result = await measure.stream(client, measure.cases(16)[0], "low", owned)
        return result, owned

    async def test_usage_counts_tokens_and_role_event_registers_ownership(self):
        result, owned = await self.client_response(json.dumps(list(range(100, 116))))
        self.assertEqual(owned, {"our-request"})
        self.assertEqual(result["usage"]["completion_tokens"], 97)
        self.assertTrue(result["correct"])
        self.assertAlmostEqual(result["end_to_end_completion_tokens_per_s"], 97 / result["finish_s"])
        self.assertIsNone(result["engine_decode_tokens_per_s"])
        self.assertLessEqual(result["ttft_s"], result["first_visible_s"])

    async def test_incorrect_or_incomplete_outputs_cannot_enter_comparison(self):
        with self.assertRaises(measure.ProbeResponseFailure) as caught:
            await self.client_response("[1,2]")
        self.assertEqual(caught.exception.response["answer"], "[1,2]")
        self.assertFalse(caught.exception.response["correct"])
        with self.assertRaises(ValueError):
            await self.client_response(json.dumps(list(range(100, 116))), usage=False)

    async def test_oracle_timeout_keeps_answer_and_distinguishes_validation_error(self):
        answer = '{"code":"def compress_runs(values): return []"}'
        case = {"id": "code", "validator": "code", "prompt": "fixture"}
        with patch.object(measure, "cases", return_value=[case]), patch(
                "code_oracle.validate", side_effect=subprocess.TimeoutExpired("oracle", 3)):
            with self.assertRaises(measure.ProbeResponseFailure) as caught:
                await self.client_response(answer)
        saved = caught.exception.response
        self.assertEqual(saved["answer"], answer)
        self.assertEqual(saved["validation_error"]["type"], "TimeoutExpired")

    async def test_waiting_client_cancels_every_probe_stream(self):
        started, cancelled = [], []

        async def blocked(_client, case, _effort, _owned):
            started.append(case["id"])
            try:
                await asyncio.Future()
            finally:
                cancelled.append(case["id"])

        async def busy(_client):
            await asyncio.sleep(.01)
            h = snapshot()
            h["request_queue"]["waiting"] = 1
            return h

        with patch.object(measure, "stream", blocked), patch.object(measure, "health", busy):
            with self.assertRaisesRegex(RuntimeError, "Another client"):
                await measure.wave(None, snapshot(), measure.cases(16), "low", True)
        self.assertEqual(set(started), set(cancelled))
        self.assertEqual(len(cancelled), 2)

    async def test_foreign_completed_request_is_detected_even_if_already_idle(self):
        async def other(_client):
            h = snapshot()
            h["request_metrics"]["recent"] = [{"id": "other"}]
            return h

        with patch.object(measure, "health", other):
            with self.assertRaisesRegex(RuntimeError, "completed"):
                await measure.monitor(None, snapshot(), set(), [])


if __name__ == "__main__":
    unittest.main()
