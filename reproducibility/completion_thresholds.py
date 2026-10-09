"""Run the prespecified 4-by-4 tolerance grid with exact coupled labels."""
import json
import pandas as pd
from engine import ROOT,PROTOCOL,load_snapshot,Result
from completion import CompletionModel,GraphPlanner
s,m=load_snapshot();old=pd.read_csv(ROOT/'results/interpretations.csv');rows=[]
for code in PROTOCOL['original_cohorts']+PROTOCOL['additional_cohorts']:
 for gene in PROTOCOL['genes']:
  b=old.loc[old.cohort.eq(code)&old.gene.eq(gene)].sort_values('index')
  model=CompletionModel(s,m,code,gene,[Result(int(r.positive),int(r.denominator),int(r.profiled_units)) for _,r in b.iterrows()])
  for rel in PROTOCOL['sensitivity_tolerances']['relative']:
   for abs_pp in PROTOCOL['sensitivity_tolerances']['absolute_percentage_points']:
    g=model.certificate_graph(rel,abs_pp);p=GraphPlanner(g)
    rows.append({'cohort':code.upper(),'gene':gene,'relative':rel,'absolute_pp':abs_pp,
                 'coupled_sensitive':bool(g.any()),'sensitive_pairs':int(g.sum()/2),
                 'optimal_expected_questions':p.solve(p.root)[0]})
f=pd.DataFrame(rows);f.to_csv(ROOT/'completion_results/threshold_families.csv',index=False)
f.groupby(['relative','absolute_pp']).agg(sensitive_families=('coupled_sensitive','sum'),
                                         sensitive_pairs=('sensitive_pairs','sum'),
                                         optimal_expected_questions=('optimal_expected_questions','mean')).reset_index().to_csv(ROOT/'completion_results/threshold_summary.csv',index=False)
print('Exact tolerance grid complete:',len(f),'family-scenarios')
