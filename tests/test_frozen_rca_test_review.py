import json
import unittest
import pandas as pd
from scripts.p6.run_frozen_rca_test_review import masks, metrics_for
from src.e2e.rca_model import GAIA_SERVICES


class RcaTestReviewTests(unittest.TestCase):
    def test_feature_masks_drop_columns_in_frozen_order(self):
        self.assertEqual(masks()['no_metric'].tolist(), list(range(17, 68)))
        keep=masks()['no_all_magnitude'].tolist()
        self.assertEqual(len(keep),56)
        self.assertEqual(keep,[c*17+j for c in range(4) for j in range(3,17)])

    def test_invalid_context_and_false_alarm_keep_full_e2e_denominators(self):
        ranking=json.dumps(list(GAIA_SERVICES))
        frame=pd.DataFrame({'prediction_id':['hit','invalid','false'],'t_hat':[30,60,90],
            'legal':[True,False,True],'ranking':[ranking,'',ranking]})
        matching=pd.DataFrame({'prediction_id':['hit','invalid','false',None],
            'match_status':['matched','matched','false_alarm','miss'],
            'case_id':['a','b',None,'c'],'gt_service':[GAIA_SERVICES[0],GAIA_SERVICES[0],None,GAIA_SERVICES[0]],
            'fault_type':['login_failure','login_failure',None,'login_failure']})
        result,ledger=metrics_for(frame,matching,3)
        self.assertEqual(result['legal_matched'],1)
        self.assertEqual(result['e2e']['@1']['tp'],1)
        self.assertEqual(result['e2e']['@1']['fp'],2)
        self.assertEqual(result['e2e']['@1']['fn'],2)
        self.assertAlmostEqual(result['e2e']['@1']['f1'],1/3)
        self.assertEqual(result['failure'],{'SUCCESS_TOP1':1,'RCA_CONTEXT_INVALID':1,
            'EVENT_FALSE_ALARM':1,'EVENT_MISSED':1})

    def test_missing_legal_ranking_remains_zero_in_matched_rca(self):
        frame=pd.DataFrame({'prediction_id':['p'],'legal':[True],'ranking':['']})
        matching=pd.DataFrame({'prediction_id':['p'],'match_status':['matched'],'case_id':['a'],
            'gt_service':[GAIA_SERVICES[0]],'fault_type':['login_failure']})
        result,ledger=metrics_for(frame,matching,1)
        self.assertEqual(result['matched_rca']['n'],1)
        self.assertEqual(result['matched_rca']['AC@1'],0)
        self.assertEqual(result['matched_rca']['MRR'],0)
        self.assertEqual(result['e2e']['@1']['fn'],1)
        self.assertEqual(result['failure'],{'RCA_RANKING_MISSING':1})


if __name__=='__main__':unittest.main()
