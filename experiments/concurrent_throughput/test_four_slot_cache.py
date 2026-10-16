import unittest
from qualify_four_slot_cache import validate_budget


class FourContextBudgetTests(unittest.TestCase):
    def test_context_pool_and_reserved_generation_are_separate_limits(self):
        self.assertLess(validate_budget([131072]*4,2048,655360,392960),655360)
        self.assertLess(validate_budget([389120,65536,65536,65536],2048,655360,392960),655360)
        with self.assertRaisesRegex(ValueError,'shared GPU'):
            validate_budget([262144]*4,2048,655360,392960)
        with self.assertRaisesRegex(ValueError,'per-request'):
            validate_budget([392960,1024,1024,1024],2048,655360,392960)


if __name__=='__main__':unittest.main()
