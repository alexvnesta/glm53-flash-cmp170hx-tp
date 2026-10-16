import json
import unittest
from types import SimpleNamespace as NS

from glm_request_metrics import RequestMetrics


class TimingContracts(unittest.TestCase):
    def setUp(self):
        self.now = 1.0
        self.rows = []
        self.metrics = RequestMetrics(clock=lambda: self.now, sink=self.rows.append, history=2)

    def start(self, ident="request"):
        return self.metrics.start(ident, received=0, prompt_tokens=1000,
                                  effort="high", streaming=True)

    def test_prefill_progress_and_exact_once_completion(self):
        r = self.start()
        self.now = 2
        r.observe({"stage": "prefill", "curr_progress": 256, "max_progress": 999})
        self.now = 3
        state = self.metrics.snapshot()["active"][0]
        self.assertEqual((state["prefilled_tokens"], state["prefill_total_tokens"]), (256, 999))
        self.assertEqual(state["progress_age_ms"], 1000)
        r.observe({"stage": "streaming", "new_tokens": 1, "text": "PRIVATE OUTPUT"})
        r.visible()
        self.now = 4
        r.outcome = "completed"
        r.finish({"predicted_ms": 1000, "draft_n": 10, "draft_n_accepted": 7,
                  "untrusted_prompt": "PRIVATE PROMPT"}, {"completion_tokens": 40}, "stop")
        r.finish({}, {}, "stop")
        self.assertEqual(len(self.rows), 1)
        self.assertEqual(self.rows[0]["engine_ttft_ms"], 3000)
        self.assertEqual(self.rows[0]["decode_tokens_per_second"], 40)
        self.assertEqual(self.rows[0]["draft_acceptance"], .7)
        self.assertNotIn("PRIVATE", json.dumps(self.rows))
        self.assertEqual(self.metrics.snapshot()["active"], [])

    def test_error_cancellation_and_bounded_history(self):
        for n in range(3):
            r = self.start(str(n))
            r.observe({"stage": "prefill", "curr_progress": 100, "max_progress": 1000})
            r.fail(RuntimeError("PRIVATE ERROR WITH PROMPT"), "disconnected")
            r.finish({}, {}, "stop")
        snapshot = self.metrics.snapshot()
        self.assertEqual(len(snapshot["recent"]), 2)
        self.assertEqual(snapshot["completed"], {"disconnected": 3})
        self.assertNotIn("PRIVATE", json.dumps(snapshot))
        self.assertEqual(snapshot["recent"][0]["error_type"], "RuntimeError")

    def test_generation_progress_before_eos_and_across_requeue(self):
        r = self.start()
        r.observe({"stage":"streaming", "eos":False, "job":NS(new_tokens=17,rq_new_tokens=100)})
        self.assertEqual(self.metrics.snapshot()["active"][0]["generated_tokens"],117)
        r.observe({"stage":"streaming", "eos":True, "new_tokens":130})
        self.assertEqual(self.metrics.snapshot()["active"][0]["generated_tokens"],130)


if __name__ == "__main__":
    unittest.main()
