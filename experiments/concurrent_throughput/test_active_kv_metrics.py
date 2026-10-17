import unittest
from probe_active_kv import decode_overlap, server_decode_rates
from prepare_active_kv import prepare
from qualify_four_slot_cache import validate_budget


def sample(seconds,counts,phase='streaming',active=4):
    # Reorder observations to check that rates follow request identity.
    return {'seconds':seconds,'active':active,'phase_progress':[
        {'id':str(i),'phase':phase,'generated_tokens':counts[i]} for i in reversed(range(active))]}


class CommonDecodeInterval(unittest.TestCase):
    def result(self,samples):
        return {'requests':[{'request_id':str(i)} for i in range(4)],'samples':samples}

    def test_excludes_prefill_and_serial_tail_uses_actual_agent_counts(self):
        r=self.result([sample(0,[0,0,0,0],'prefill'),
            sample(2,[10,20,30,40]),sample(4,[26,40,54,68]),
            sample(10,[900,900,900,900],active=3)])
        got=decode_overlap(r)
        self.assertEqual(got['seconds'],2)
        self.assertEqual(got['per_agent_tokens_per_second'],[8,10,12,14])
        self.assertEqual(got['aggregate_tokens_per_second'],44)

    def test_no_overlapping_decode_is_unmeasured(self):
        r=self.result([sample(0,[0,0,0,0],'prefill'),sample(2,[10,20,30,40]),
                       sample(3,[20,40,60,80],active=3)])
        self.assertIsNone(decode_overlap(r))

    def test_counter_regression_refuses_rate(self):
        r=self.result([sample(2,[10,20,30,40]),sample(4,[9,40,54,68])])
        with self.assertRaisesRegex(RuntimeError,'regressed'):decode_overlap(r)

    def test_server_metrics_follow_agent_identity_not_completion_order(self):
        r=self.result([])
        r['server_metrics']=[{'id':str(i),'decode_tokens_per_second':10+i} for i in (3,1,0,2)]
        self.assertEqual(server_decode_rates(r),[10,11,12,13])


class CapacityBounds(unittest.TestCase):
    def test_four_256k_inputs_reserve_normal_32k_outputs(self):
        reserved=validate_budget([262144]*4,32768,1187840,392960)
        self.assertEqual(reserved,1179904)
        with self.assertRaises(ValueError):validate_budget([262144]*4,32768,655360,392960)

    def test_unreviewed_pool_and_non_boolean_mode_fail_before_io(self):
        for mapped,pool in [(True,1188096),(True,1187841),(True,True),(1,1187840)]:
            with self.subTest(mapped=mapped,pool=pool):
                with self.assertRaises(ValueError):prepare(None,None,2,mapped,pool)

    def test_invalid_pid_fails_before_io(self):
        with self.assertRaisesRegex(ValueError,'production PID'):prepare(None,None,0,True,1187840)


if __name__=='__main__':unittest.main()
