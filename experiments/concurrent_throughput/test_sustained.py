import unittest
import sustained
import measure

class SustainedMeasurementTests(unittest.TestCase):
    def snapshot(self):
        return {'status':'ok','native_extension':{'sha256':'same'},'profile_digest':'same',
                'request_queue':{'admitted':12,'waiting':2},
                'request_metrics':{'active':[{'id':n} for n in ('a','b','c','d')],'recent':[]}}

    def test_own_queued_work_is_allowed_but_extra_traffic_yields(self):
        h=self.snapshot();identity=measure.identity(h)
        sustained.check_traffic(h,identity,set(),{'a','b'},4,10)
        h['request_metrics']['active'].append({'id':'foreign'})
        with self.assertRaisesRegex(RuntimeError,'Unowned'):
            sustained.check_traffic(h,identity,set(),{'a','b'},4,10)
        h=self.snapshot();h['request_queue']['admitted']=15
        with self.assertRaisesRegex(RuntimeError,'Other traffic'):
            sustained.check_traffic(h,identity,set(),{'a','b'},4,10)
        h=self.snapshot();h['profile_digest']='changed'
        with self.assertRaisesRegex(RuntimeError,'identity'):
            sustained.check_traffic(h,identity,set(),set(),4,10)

    def test_agent_rates_sum_to_aggregate_and_steady_uses_tokens(self):
        rows=[{'agent':a,'usage':{'completion_tokens':n},'correct':True,'finish_s':2}
              for a,n in ((0,100),(1,200))]
        samples=[{'seconds':s,'agent_tokens':n,'active_sequences':2,'waiting':2}
                 for s,n in [(0,[0,0]),(5,[10,20]),(10,[90,180]),(15,[100,200])]]
        d=sustained.summarize(rows,samples,2,15)
        self.assertAlmostEqual(sum(a['delivered_tok_s'] for a in d['agents']),d['aggregate_tok_s'])
        self.assertEqual(d['steady']['aggregate_tok_s'],48)
        self.assertEqual(d['steady']['per_agent_tok_s'],[16,32])

    def test_no_counter_repair_and_failed_tasks_stay_failed(self):
        rows=[{'agent':0,'usage':{'completion_tokens':10},'correct':False,'finish_s':1}]
        samples=[{'seconds':s,'agent_tokens':[n],'active_sequences':1,'waiting':0} for s,n in [(5,8),(10,7)]]
        with self.assertRaisesRegex(RuntimeError,'backwards'):
            sustained.summarize(rows,samples,1,15)
        result=sustained.summarize(rows,[],1,2)
        self.assertFalse(result['all_correct']);self.assertEqual(result['failed_tasks'],1)

if __name__=='__main__':unittest.main()
