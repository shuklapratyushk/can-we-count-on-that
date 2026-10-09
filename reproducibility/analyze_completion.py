"""Execute the coupled-completion audit and exact planning scenarios."""
from __future__ import annotations
from datetime import datetime,timezone
from fractions import Fraction
from itertools import product,permutations
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd

from engine import ROOT,PROTOCOL,FIELDS,ASSIGNMENTS,load_snapshot,Result
from analyze import verify_snapshot
from completion import CompletionModel,GraphPlanner,product_prior,expected_cost_grid
from verify_completion_lp import verify_model,cache_stats

OUT=ROOT/'completion_results'
ORDERS=list(permutations(range(4)))
UPPER=np.triu_indices(16,1)


def fixed_question_incidence(planner):
    incidence=np.zeros((24,16,4),dtype=np.int64)
    for o,order in enumerate(ORDERS):
        for i in range(16):
            _,fields,_=planner.walk(i,order=order)
            for field in fields:incidence[o,i,field]+=1
    return incidence


def validate_random_small_cases(count=64):
    from test_completion import frames,exhaustive_oracle
    rng=np.random.default_rng(20261006);world_count=0
    for case in range(count):
        rows=[('x','s0','p0',True,True)];events=[];missing=0
        for i in range(1,7):
            profile=bool(rng.integers(0,2)) if missing<3 else True
            missing+=not profile
            rows.append(('x',f's{i}',f'p{int(rng.integers(0,4))}',bool(rng.integers(0,2)),profile))
            state=int(rng.integers(0,3))
            if state:events.append(('x',f's{i}','TP53','Missense_Mutation' if state==1 else 'Nonsense_Mutation'))
        s,m=frames(rows,events);model=CompletionModel(s,m,'x','TP53')
        worlds=exhaustive_oracle(s,m);world_count+=len(worlds)
        absolute=(np.abs(worlds[:,:,None]-worlds[:,None,:])*100).max(axis=0)
        relative=(np.abs(worlds[:,:,None]-worlds[:,None,:])/
                  ((worlds[:,:,None]+worlds[:,None,:])/2+1e-12)).max(axis=0)
        actual,_=model.absolute_certificate();lo,hi=model.relative_certificate()
        np.testing.assert_allclose(actual,absolute,atol=1e-11)
        assert (lo<=relative+3e-12).all() and (hi>=relative-3e-12).all()
        assert (hi-lo).max()<=2/2**40+1e-15
        observed=model.observed_graph();coupled=model.certificate_graph()
        assert not (observed&~coupled).any()
    return {'synthetic_fixtures':count,'globally_enumerated_worlds':world_count,
            'status':'All absolute supports and relative brackets match direct global enumeration',
            'interpretation':'Software/mathematical verification, not empirical biological observations'}


def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=True)
    verify_snapshot()
    protocol_path=ROOT/'reproducibility/completion_protocol.json'
    protocol=json.loads(protocol_path.read_text())
    record=json.loads((OUT/'protocol_record.json').read_text())
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest()==record['completion_protocol_sha256']
    samples,mutations=load_snapshot()
    old=pd.read_csv(ROOT/'results/interpretations.csv')
    prior_parameters=list(product(protocol['prior_grid']['field_zero_probabilities'],repeat=4))
    weights=np.array([product_prior(p) for p in prior_parameters])
    cost_models=protocol['cost_scenarios']
    families=[];pairs=[];axes=[];spec_rows=[];uniform=[];paths=[];prior_rows=[]
    certs=['observed','coupled','box','box_merged']
    fixed_matrices={k:[] for k in certs}
    global_grid_fixed={k:np.zeros((81,24)) for k in cost_models}
    global_grid_opt={k:np.zeros(81) for k in cost_models}
    global_grid_familyfixed={k:np.zeros(81) for k in cost_models}
    graph_store={};certificate_store={};lp_count=0
    cohort_codes=PROTOCOL['original_cohorts']+PROTOCOL['additional_cohorts']
    for code in cohort_codes:
        for gene in PROTOCOL['genes']:
            baseline=old.loc[old.cohort.eq(code)&old.gene.eq(gene)].sort_values('index')
            baseline_results=[Result(int(r.positive),int(r.denominator),int(r.profiled_units)) for _,r in baseline.iterrows()]
            model=CompletionModel(samples,mutations,code,gene,baseline_results)
            assert np.array_equal(model.observed_counts,np.array(baseline.positive))
            lp_count+=verify_model(model,samples,mutations)
            absval,absnum=model.absolute_certificate()
            rlo,rhi=model.relative_certificate(protocol['relative_root_interval_iterations'])
            obs=model.observed_graph();coupled=model.certificate_graph()
            box=model.independent_interval_graph();merged=model.independent_interval_graph(merge_equivalent=True)
            assert not (obs&~coupled).any()
            assert not (coupled&~box).any()
            assert not (coupled&~merged).any()
            naive_abs,naive_rel=model.independent_interval_certificate()
            label=f'{code}:{gene}'
            graphs={'observed':obs,'coupled':coupled,'box':box,'box_merged':merged}
            graph_store[label]=np.stack([graphs[c] for c in certs])
            certificate_store[label]=np.stack([absval,rlo,rhi,naive_abs,naive_rel])
            any_observed_axes=any(obs[0,1<<(3-f)] for f in range(4))
            any_coupled_axes=any(coupled[0,1<<(3-f)] for f in range(4))
            width_counts=model.upper_counts-model.lower_counts
            point_guard=width_counts*20>model.denominators
            family={'cohort':code.upper(),'gene':gene,'group':'original' if code in PROTOCOL['original_cohorts'] else 'additional',
                    'observed_sensitive':bool(obs.any()),'coupled_sensitive':bool(coupled.any()),
                    'box_sensitive':bool(box.any()),'merged_box_sensitive':bool(merged.any()),
                    'observed_one_axis_stable':not any_observed_axes,'coupled_one_axis_stable':not any_coupled_axes,
                    'observed_absolute_pp':100*(model.observed.max()-model.observed.min()),
                    'coupled_absolute_pp':float(absval.max()),'coupled_relative_lower':float(rlo.max()),
                    'coupled_relative_upper':float(rhi.max()),'independent_absolute_pp':float(naive_abs.max()),
                    'point_unidentified_specs':int(point_guard.sum()),'max_identification_width_pp':float(100*(model.upper-model.lower).max()),
                    'unprofiled_samples':model.unprofiled_samples,'patient_completion_blocks':len(model.blocks),
                    'max_unprofiled_per_patient':model.max_unprofiled_per_patient,'local_completion_states':model.local_state_count,
                    'observed_sensitive_pairs':int(obs[UPPER].sum()),'coupled_sensitive_pairs':int(coupled[UPPER].sum()),
                    'box_sensitive_pairs':int(box[UPPER].sum()),'merged_box_sensitive_pairs':int(merged[UPPER].sum()),
                    'box_only_pairs':int((box&~coupled)[UPPER].sum()),'merged_box_only_pairs':int((merged&~coupled)[UPPER].sum())}
            for i in range(16):
                spec=model.specs[i]
                spec_rows.append({'cohort':code,'gene':gene,'index':i,**spec.__dict__,
                    'positive':int(model.observed_counts[i]),'denominator':int(model.denominators[i]),
                    'lower':model.lower[i],'upper':model.upper[i],
                    'identification_width_pp':100*(model.upper[i]-model.lower[i]),
                    'minimax_point_radius_pp':50*(model.upper[i]-model.lower[i]),
                    'point_error_budget_2_5pp_unavailable':bool(point_guard[i])})
                for j in range(i+1,16):
                    delta=abs(model.observed[i]-model.observed[j])
                    pairs.append({'cohort':code.upper(),'gene':gene,'i':i,'j':j,
                       'observed_absolute_pp':100*delta,'observed_relative':delta/((model.observed[i]+model.observed[j])/2+1e-12),
                       'coupled_absolute_pp':absval[i,j],'coupled_relative_lower':rlo[i,j],'coupled_relative_upper':rhi[i,j],
                       'box_absolute_pp':naive_abs[i,j],'box_relative':naive_rel[i,j],
                       **{k+'_sensitive':bool(g[i,j]) for k,g in graphs.items()},
                       'completion_equivalent':bool(absnum[i,j]==0)})
            for field in range(4):
                index=1<<(3-field)
                axes.append({'cohort':code.upper(),'gene':gene,'field':FIELDS[field],
                             **{k+'_sensitive':bool(g[0,index]) for k,g in graphs.items()},
                             'coupled_absolute_pp':absval[0,index],'box_absolute_pp':naive_abs[0,index]})
            coupled_incidence=None
            for cert in certs:
                p=GraphPlanner(graphs[cert]);incidence=fixed_question_incidence(p)
                matrix=incidence.sum(axis=2).mean(axis=1)
                fixed_matrices[cert].append(matrix)
                value,field=p.solve(p.root)
                expected,maxcost,truth_costs=p.policy_costs()
                assert abs(value-expected)<1e-12
                minimax,_=p.solve(p.root,'minimax')
                minimax_mean,minimax_max,_=p.policy_costs('minimax')
                assert abs(minimax-minimax_max)<1e-12
                greedy,greedy_max,_=p.policy_costs('edge_greedy')
                entropy,entropy_max,_=p.policy_costs('entropy')
                uniform.append({'cohort':code.upper(),'gene':gene,'certificate':cert,
                    'exact_expected':value,'exact_max':maxcost,'minimax_worst':minimax,'minimax_mean':minimax_mean,
                    'greedy_mean':greedy,'entropy_mean':entropy,'best_family_fixed':matrix.min(),
                    'first_field':None if field is None else FIELDS[field]})
                family[cert+'_expected_questions']=value
                for i in range(16):
                    cost,fields,state=p.walk(i)
                    paths.append({'cohort':code.upper(),'gene':gene,'certificate':cert,'truth_index':i,
                                  'questions':cost,'fields':';'.join(FIELDS[f] for f in fields),
                                  'remaining_candidates':len(state),'terminal_verified':p.stable(state)})
                if cert=='coupled':coupled_incidence=incidence
            for cost_name,costs in cost_models.items():
                optimal,first=expected_cost_grid(coupled,weights,costs)
                assignment_fixed=np.einsum('oif,f->oi',coupled_incidence,np.array(costs))
                fixed=weights@assignment_fixed.T
                assert (optimal<=fixed.min(axis=1)+1e-12).all()
                global_grid_fixed[cost_name]+=fixed
                global_grid_opt[cost_name]+=optimal
                global_grid_familyfixed[cost_name]+=fixed.min(axis=1)
                for prior_index,(params,value,field) in enumerate(zip(prior_parameters,optimal,first)):
                    prior_rows.append({'cohort':code.upper(),'gene':gene,'cost_scenario':cost_name,'prior_index':prior_index,
                        **{f'p_{FIELDS[f]}_zero':params[f] for f in range(4)},'optimal_expected_cost':value,
                        'best_family_fixed_cost':fixed[prior_index].min(),
                        'first_field':None if field<0 else FIELDS[field]})
            families.append(family)
        print(f'{code.upper()}: completed certificates, global-LP verification, and prior/cost scenarios',flush=True)

    ff=pd.DataFrame(families);pp=pd.DataFrame(pairs);aa=pd.DataFrame(axes);uu=pd.DataFrame(uniform)
    fixed_summary=[];uniform_summary=[]
    for cert in certs:
        matrix=np.stack(fixed_matrices[cert]);global_means=matrix.mean(axis=0);best=int(global_means.argmin())
        order=';'.join(FIELDS[f] for f in ORDERS[best])
        subset=uu.loc[uu.certificate.eq(cert)]
        uniform_summary.append({'certificate':cert,'exact_expected':subset.exact_expected.mean(),
            'exact_max':subset.exact_max.max(),'minimax_mean':subset.minimax_mean.mean(),
            'minimax_worst':subset.minimax_worst.max(),'greedy_mean':subset.greedy_mean.mean(),
            'entropy_mean':subset.entropy_mean.mean(),'best_family_fixed':subset.best_family_fixed.mean(),
            'best_global_fixed':global_means[best],'best_global_order':order,
            'saving_vs_best_global_fixed':0. if global_means[best]==0 else 1-subset.exact_expected.mean()/global_means[best]})
        for index,means in enumerate(matrix):
            ff.loc[index,cert+'_best_global_fixed_questions']=means[best]
        for o,mean in enumerate(global_means):fixed_summary.append({'certificate':cert,'order':';'.join(FIELDS[f] for f in ORDERS[o]),'mean_questions':mean})
    grid=[]
    for cost_name in cost_models:
        fixed=global_grid_fixed[cost_name]/len(ff);opt=global_grid_opt[cost_name]/len(ff)
        per_family=global_grid_familyfixed[cost_name]/len(ff)
        for prior_index,params in enumerate(prior_parameters):
            best=int(fixed[prior_index].argmin());cost=fixed[prior_index,best]
            grid.append({'cost_scenario':cost_name,'prior_index':prior_index,
                **{f'p_{FIELDS[f]}_zero':params[f] for f in range(4)},'optimal_expected_cost':opt[prior_index],
                'best_global_fixed_cost':cost,'best_family_fixed_cost':per_family[prior_index],
                'saving_vs_best_global_fixed':1-opt[prior_index]/cost,
                'saving_vs_best_family_fixed':1-opt[prior_index]/per_family[prior_index],
                'best_global_order':';'.join(FIELDS[f] for f in ORDERS[best])})
    pd.DataFrame(spec_rows).to_csv(OUT/'interpretations.csv',index=False)
    ff.to_csv(OUT/'families.csv',index=False);pp.to_csv(OUT/'pairs.csv',index=False)
    aa.to_csv(OUT/'single_axis.csv',index=False);uu.to_csv(OUT/'uniform_planning.csv',index=False)
    pd.DataFrame(uniform_summary).to_csv(OUT/'uniform_planning_summary.csv',index=False)
    pd.DataFrame(fixed_summary).to_csv(OUT/'fixed_order_summary.csv',index=False)
    pd.DataFrame(paths).to_csv(OUT/'policy_paths.csv',index=False)
    pd.DataFrame(prior_rows).to_csv(OUT/'prior_cost_families.csv',index=False)
    gg=pd.DataFrame(grid);gg.to_csv(OUT/'prior_cost_grid.csv',index=False)
    np.savez_compressed(OUT/'graphs.npz',certificate_names=np.array(certs),**graph_store)
    np.savez_compressed(OUT/'certificates.npz',**certificate_store)
    validation=validate_random_small_cases()
    summary={'created_at_utc':datetime.now(timezone.utc).isoformat(),'families':len(ff),'interpretations':len(spec_rows),
       'pair_certificates':len(pp),'one_axis_contrasts':len(aa),'tcga_samples':len(samples),'tcga_patients':samples.groupby('cohort').patient.nunique().sum(),
       'unprofiled_specimens':int((~samples.profiled).sum()),'largest_unprofiled_patient_block':int(ff.max_unprofiled_per_patient.max()),
       'global_lp_verified_pairs':lp_count,**cache_stats(),
       'families_sensitive':{c:int(ff[{'observed':'observed_sensitive','coupled':'coupled_sensitive','box':'box_sensitive','box_merged':'merged_box_sensitive'}[c]].sum()) for c in certs},
       'pair_sensitive_counts':{c:int(pp[c+'_sensitive'].sum()) for c in certs},
       'one_axis_sensitive_counts':{c:int(aa[c+'_sensitive'].sum()) for c in certs},
       'box_only_pair_count':int((pp.box_sensitive&~pp.coupled_sensitive).sum()),
       'merged_box_only_pair_count':int((pp.box_merged_sensitive&~pp.coupled_sensitive).sum()),
       'observed_stable_coupled_sensitive_pairs':int((~pp.observed_sensitive&pp.coupled_sensitive).sum()),
       'point_guard_unidentified_specs':int(sum(r['point_error_budget_2_5pp_unavailable'] for r in spec_rows)),
       'relative_bracket_max_width':float((pp.coupled_relative_upper-pp.coupled_relative_lower).max()),
       'uniform_planning':uniform_summary,'scenario_priors':81,'cost_scenarios':len(cost_models),
       'prior_cost_scenarios':len(gg),'scenario_saving_range_vs_global_fixed':[gg.saving_vs_best_global_fixed.min(),gg.saving_vs_best_global_fixed.max()],
       'scenario_saving_range_vs_family_fixed':[gg.saving_vs_best_family_fixed.min(),gg.saving_vs_best_family_fixed.max()],
       'synthetic_validation':validation,'runtime_seconds':time.perf_counter()-start,
       'provenance':'Executed coupled missing-status extension on frozen public TCGA snapshot; scenario priors/costs are stipulated, not empirical user distributions'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item())+'\n')
    print(json.dumps(summary,indent=2,default=lambda x:x.item()),flush=True)


if __name__=='__main__':main()
