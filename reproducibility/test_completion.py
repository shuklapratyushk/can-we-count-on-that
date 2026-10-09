"""Independent brute-force contracts for the completion and planning layer."""
from itertools import product,permutations
import unittest
import numpy as np
import pandas as pd

from completion import CompletionModel,GraphPlanner,product_prior,expected_cost_grid
from engine import ASSIGNMENTS,Spec,PROTOCOL


def frames(rows,events):
    return (pd.DataFrame(rows,columns=['cohort','sample','patient','primary','profiled']),
            pd.DataFrame(events,columns=['cohort','sample','gene','mutation_type']))


def exhaustive_oracle(samples,mutations):
    """Direct specimen completion and distinct counting, without block supports."""
    unknown=samples.loc[~samples.profiled,'sample'].tolist()
    known={scope:set(mutations.loc[mutations.mutation_type.isin(PROTOCOL[scope+'_types']),'sample'])
                 &set(samples.loc[samples.profiled,'sample']) for scope in ['broad','truncating']}
    answers=[]
    for completion in product((0,1,2),repeat=len(unknown)):
        eventsets={'broad':known['broad']|{s for s,z in zip(unknown,completion) if z>0},
                   'truncating':known['truncating']|{s for s,z in zip(unknown,completion) if z==2}}
        values=[]
        for bits in ASSIGNMENTS:
            spec=Spec.from_bits('x','TP53',bits)
            rows=samples if spec.sample_context=='all' else samples.loc[samples.primary]
            if spec.assay_denominator=='profiled':rows=rows.loc[rows.profiled]
            denominator=set(rows[spec.analysis_unit])
            positives=set(rows.loc[rows['sample'].isin(eventsets[spec.mutation_scope]),spec.analysis_unit])
            values.append(len(positives)/len(denominator))
        answers.append(values)
    return np.asarray(answers)


class CompletionContracts(unittest.TestCase):
    def setUp(self):
        self.s,self.m=frames([
            ('x','a','p1',True,True),('x','b','p1',False,False),
            ('x','c','p2',True,True),('x','d','p3',True,False)],
            [('x','c','TP53','Nonsense_Mutation'),('x','b','TP53','Nonsense_Mutation')])
        self.model=CompletionModel(self.s,self.m,'x','TP53')

    def test_profiled_count_excludes_unprofiled_record(self):
        self.assertEqual(self.model.observed[0],.5)

    def test_partial_patient_observation_enters_upper_bound(self):
        self.assertAlmostEqual(self.model.lower[4],1/3)
        self.assertEqual(self.model.upper[4],1.)

    def test_primary_context_does_not_use_missing_metastatic_status(self):
        self.assertAlmostEqual(self.model.upper[6],2/3)

    def test_sample_bound_accounts_for_three_possible_positives(self):
        self.assertEqual(self.model.lower[12],.25)
        self.assertEqual(self.model.upper[12],.75)

    def test_shared_status_unit_equality_is_not_semantic_ambiguity(self):
        s,m=frames([('x','a','p1',True,True),('x','b','p2',True,False)],
                   [('x','a','TP53','Missense_Mutation')])
        model=CompletionModel(s,m,'x','TP53')
        absval,_=model.absolute_certificate()
        self.assertEqual(absval[4,12],0)
        self.assertTrue(model.independent_interval_graph()[4,12])
        self.assertFalse(model.independent_interval_graph(merge_equivalent=True)[4,12])

    def test_nested_classes_respected_in_every_local_world(self):
        for block in self.model.blocks:
            self.assertTrue((block['counts'][:,4]>=block['counts'][:,5]).all())

    def test_all_global_worlds_match_sharp_absolute_support(self):
        worlds=exhaustive_oracle(self.s,self.m)
        brute=np.abs(worlds[:,:,None]-worlds[:,None,:]).max(axis=0)*100
        computed,_=self.model.absolute_certificate()
        np.testing.assert_allclose(computed,brute,atol=1e-12)

    def test_all_global_worlds_match_relative_bracket(self):
        worlds=exhaustive_oracle(self.s,self.m)
        brute=(np.abs(worlds[:,:,None]-worlds[:,None,:])/
               ((worlds[:,:,None]+worlds[:,None,:])/2+1e-12)).max(axis=0)
        lo,hi=self.model.relative_certificate()
        self.assertTrue((brute>=lo-2e-12).all())
        self.assertTrue((brute<=hi+2e-12).all())
        self.assertLessEqual((hi-lo).max(),2/2**40+1e-15)

    def test_repeated_unprofiled_specimens_use_patient_or_not_a_sum(self):
        s,m=frames([('x','a','p1',True,True),('x','b','p2',True,False),
                    ('x','c','p2',False,False)],[])
        model=CompletionModel(s,m,'x','TP53')
        self.assertEqual(model.max_unprofiled_per_patient,2)
        self.assertEqual(model.local_state_count,9)
        self.assertEqual(model.upper[4],.5)
        self.assertAlmostEqual(model.upper[12],2/3)
        brute=exhaustive_oracle(s,m)
        np.testing.assert_allclose(model.absolute_certificate()[0],
            np.abs(brute[:,:,None]-brute[:,None,:]).max(axis=0)*100,atol=1e-12)

    def test_exact_strict_threshold_handles_equality(self):
        rows=[('x',f's{i}',f'p{i}',True,True) for i in range(40)]
        events=[('x',f's{i}','TP53','Nonsense_Mutation' if i<19 else 'Missense_Mutation') for i in range(21)]
        s,m=frames(rows,events);model=CompletionModel(s,m,'x','TP53')
        # 21/40 vs 19/40 is exactly five pp and approaches relative 0.1 from below
        # with positive epsilon. Both strict inequalities must be false.
        self.assertFalse(model.certificate_graph()[0,1])

    def test_absolute_branch_when_relative_is_below_tolerance(self):
        rows=[('x',f's{i}',f'p{i}',True,True) for i in range(100)]
        events=[('x',f's{i}','TP53','Nonsense_Mutation' if i<80 else 'Missense_Mutation') for i in range(87)]
        s,m=frames(rows,events);model=CompletionModel(s,m,'x','TP53')
        self.assertFalse(model.relative_threshold()[0][0,1])
        self.assertTrue(model.absolute_threshold()[0,1])

    def test_singleton_semantic_stability_does_not_identify_a_point(self):
        planner=GraphPlanner(self.model.certificate_graph())
        self.assertTrue(planner.stable((4,)))
        self.assertGreater(100*(self.model.upper[4]-self.model.lower[4])/2,2.5)

    def test_undefined_denominator_cannot_receive_a_certificate(self):
        s,m=frames([('x','a','p1',True,False)],[])
        with self.assertRaises(ValueError):CompletionModel(s,m,'x','TP53')


class PlanningContracts(unittest.TestCase):
    def setUp(self):
        self.graph=np.array([[a[3]!=b[3] for b in ASSIGNMENTS] for a in ASSIGNMENTS])

    def test_irrelevant_entropy_fields_are_skipped_by_optimum(self):
        planner=GraphPlanner(self.graph)
        self.assertEqual(planner.solve(planner.root),(1.,3))
        self.assertEqual(planner.policy_costs('entropy')[0],4.)

    def test_nonuniform_prior_and_nonunit_cost_are_explicit(self):
        planner=GraphPlanner(self.graph,product_prior((.2,.8,.5,.9)),(1,1,1,3))
        self.assertEqual(planner.solve(planner.root),(3.,3))

    def test_batch_bellman_matches_scalar_and_enumerated_paths(self):
        priors=np.array([product_prior(p) for p in product((.2,.5,.8),repeat=4)])
        for costs in [(1,1,1,1),(1,3,3,1)]:
            expected,_=expected_cost_grid(self.graph,priors,costs)
            for w,answer in zip(priors,expected):
                p=GraphPlanner(self.graph,w,costs)
                self.assertAlmostEqual(p.solve(p.root)[0],answer)
                self.assertAlmostEqual(p.policy_costs()[0],answer)

    def test_minimax_cost_and_all_terminal_sets(self):
        graph=np.array([[a[0]!=b[0] or a[3]!=b[3] for b in ASSIGNMENTS] for a in ASSIGNMENTS])
        p=GraphPlanner(graph,costs=(2,1,1,3))
        self.assertEqual(p.solve(p.root,'minimax')[0],5.)
        for i in range(16):
            cost,_,state=p.walk(i,'minimax');self.assertEqual(cost,5.);self.assertTrue(p.stable(state))

    def test_exact_expected_value_dominates_every_fixed_order(self):
        graph=np.array([[a[3]!=b[3] or (a[3]==0 and a[0]!=b[0]) for b in ASSIGNMENTS] for a in ASSIGNMENTS])
        p=GraphPlanner(graph,product_prior((.2,.5,.5,.8)))
        optimum=p.solve(p.root)[0]
        self.assertAlmostEqual(optimum,p.policy_costs()[0])
        for order in permutations(range(4)):self.assertLessEqual(optimum,p.policy_costs(order=order)[0]+1e-12)


if __name__=='__main__':
    unittest.main(verbosity=2)
