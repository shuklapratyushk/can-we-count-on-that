"""Completion-aware audit card for supplied typed TCGA specifications.

This is a typed interface, not a natural-language parser. It distinguishes a
stable candidate set from identification of a numerical point answer.
"""
from __future__ import annotations
import argparse,json
import math
from dataclasses import asdict
from fractions import Fraction
import numpy as np
from engine import PROTOCOL,FIELDS,VALUES,ASSIGNMENTS,Spec,SQLiteEngine,load_snapshot
from completion import CompletionModel,GraphPlanner
PROMPTS={
 'analysis_unit':'Count unique patients or individual specimens?',
 'assay_denominator':'Use profiled units, or the full-cohort completion target?',
 'sample_context':'Use all specimens, or source-labeled primary tumors only?',
 'mutation_scope':'Use the broad class set, or the restricted nested class set?'}

def make_audit(cohort,gene,constraints,point_error_budget_pp=2.5):
 if cohort not in PROTOCOL['original_cohorts']+PROTOCOL['additional_cohorts']:raise ValueError('Cohort outside the frozen TCGA snapshot')
 if gene not in PROTOCOL['genes']:raise ValueError('Gene outside the fixed panel')
 if not math.isfinite(point_error_budget_pp) or point_error_budget_pp<0:raise ValueError('Finite nonnegative point budget required')
 if set(constraints)-set(FIELDS):raise ValueError('Unsupported scientific field')
 for field,values in zip(FIELDS,VALUES):
  if constraints.get(field) is not None and constraints[field] not in values:raise ValueError('Unsupported value for '+field)
 s,m=load_snapshot();sql=SQLiteEngine(s,m);specs=[Spec.from_bits(cohort,gene,b) for b in ASSIGNMENTS]
 results=[sql.execute(x) for x in specs];sql.connection.close();model=CompletionModel(s,m,cohort,gene,results)
 state=tuple(i for i,b in enumerate(ASSIGNMENTS) if all(constraints.get(field) is None or VALUES[f][b[f]]==constraints[field] for f,field in enumerate(FIELDS)))
 if not state:raise ValueError('Constraints have no supported interpretation')
 graph=model.certificate_graph();planner=GraphPlanner(graph);value,field=planner.solve(state);stable=planner.stable(state)
 lo=min(Fraction(int(model.lower_counts[i]),int(model.denominators[i])) for i in state)
 hi=max(Fraction(int(model.upper_counts[i]),int(model.denominators[i])) for i in state)
 radius=(hi-lo)/2;budget=Fraction(str(point_error_budget_pp))/100;point_supported=radius<=budget
 absolute,_=model.absolute_certificate();rlo,rhi=model.relative_certificate();chosen=np.ix_(state,state)
 status='clarification_required' if not stable else ('point_answer_with_certificate' if point_supported else 'interval_answer')
 return {'cohort':cohort,'gene':gene,'fixed_fields':constraints,'candidate_count':len(state),'status':status,
  'scope':'Supported typed TCGA interpretations under the shared assay-record completion model; no NLP or clinical correctness claim',
  'semantic_certificate':{'stable':stable,'violating_pairs':int(graph[chosen].sum()/2),'sharp_absolute_diameter_pp':float(absolute[chosen].max()),'sharp_relative_diameter_lower':float(rlo[chosen].max()),'sharp_relative_diameter_upper':float(rhi[chosen].max())},
  'identification_guard':{'candidate_envelope_lower':float(lo),'candidate_envelope_upper':float(hi),'minimax_point_radius_pp':100*float(radius),'point_error_budget_pp':point_error_budget_pp,'point_supported_after_semantic_resolution':bool(stable and point_supported)},
  'answer':None if not stable else {'interval':[float(lo),float(hi)],'minimax_point':float((lo+hi)/2) if point_supported else None,'interpretation':'Sharp envelope of feasible supported values; midpoint is a minimax representative, not an unbiased prevalence estimator'},
  'stopping_tolerance':PROTOCOL['primary_tolerance'],'uniform_prior_expected_questions':value,
  'next_field':None if field is None else FIELDS[field],'next_prompt':None if field is None else PROMPTS[FIELDS[field]],
  'data_note':'Profile labels and recorded classes are fixed. No per-base coverage or technical error model. Clarification does not measure unprofiled specimens.',
  'interpretations':[{'index':i,**asdict(specs[i]),**results[i].to_dict(),'lower':model.lower[i],'upper':model.upper[i]} for i in state]}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--cohort',required=True);parser.add_argument('--gene',required=True)
 for field,values in zip(FIELDS,VALUES):parser.add_argument('--'+field.replace('_','-'),choices=values)
 parser.add_argument('--point-error-budget-pp',type=float,default=2.5);parser.add_argument('--interactive',action='store_true');args=parser.parse_args()
 constraints={field:getattr(args,field) for field in FIELDS}
 while True:
  card=make_audit(args.cohort.lower(),args.gene.upper(),constraints,args.point_error_budget_pp);print(json.dumps(card,indent=2))
  if not args.interactive or card['status']!='clarification_required':break
  field=card['next_field'];allowed=PROTOCOL['field_values'][field]
  while True:
   value=input(card['next_prompt']+' ['+' / '.join(allowed)+']: ').strip()
   if value in allowed:constraints[field]=value;break
   print('Choose one of the listed field values.')
if __name__=='__main__':main()
