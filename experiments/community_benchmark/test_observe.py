import unittest
from unittest.mock import patch
from observe import Counter, summarize


class Tests(unittest.TestCase):
    def test_observer_forwards_shared_updates_without_dividing_aggregate(self):
        shared, events = [10], []
        a, b = Counter(shared, 0, events), Counter(shared, 1, events)
        with patch("observe.time.monotonic", side_effect=[1.0, 2.0, 3.0]):
            a[0] += 3
            b[0] += 7
            a[0] += 5
        self.assertEqual(shared, [25])
        result = summarize(events, 1.5, 2, 2, 12)
        self.assertEqual(result["completion_tokens"], [5, 7])
        self.assertEqual(result["tokens_per_second"], [2.5, 3.5])
        with self.assertRaises(ValueError):
            summarize(events, 1.5, 2, 2, 15)


if __name__ == "__main__":
    unittest.main()
