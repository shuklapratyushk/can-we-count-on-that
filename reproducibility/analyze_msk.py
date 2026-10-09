"""Separate, full-panel targeted-assay stress test on pinned public MSK data."""
from pathlib import Path
from datetime import datetime,timezone
from itertools import permutations
import hashlib,json,time
import numpy as np,pandas as pd
from engine import ROOT,PROTOCOL,FIELDS,ASSIGNMENTS,Spec,PandasEngine,SQLiteEngine
from completion import CompletionModel,GraphPlanner
from analyze_completion import fixed_question_incidence
OUT=ROOT/'msk_results';DATA=ROOT/'data/msk';ORDERS=list(permutations(range(4)))

def main():
 t0=time.perf_counter();proto=ROOT/'reproducibility/msk_protocol.json'
 assert hashlib.sha256(proto.read_bytes()).hexdigest()==json.loads((OUT/'protocol_record.json').read_text())['sha256']
 manifest=json.loads((DATA/'manifest.json').read_text())
 assert manifest['protocol_sha256']==hashlib.sha256(proto.read_bytes()).hexdigest()
 for entry in manifest['files']:
  p=ROOT/entry['local_path'];assert len(p.read_bytes())==entry['bytes'];assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']
 clinical=pd.read_csv(DATA/'data_clinical_sample.txt',sep='\t',comment='#')
 assert not clinical.SAMPLE_ID.duplicated().any();assert clinical.PATIENT_ID.notna().all()
 matrix=pd.read_csv(DATA/'data_gene_panel_matrix.txt',sep='\t').set_index('SAMPLE_ID')
 assert matrix.index.is_unique;assert set(clinical.SAMPLE_ID)==set(matrix.index)
 panel_genes={}
 for file in DATA.glob('data_gene_panel_*.txt'):
  if file.name=='data_gene_panel_matrix.txt':continue
  lines=dict(line.split(':',1) for line in file.read_text().splitlines() if ':' in line)
  panel_genes[lines['stable_id'].strip()]=set(lines['gene_list'].split())
 assert set(matrix.mutations.dropna().unique())<=set(panel_genes)
 raw=pd.read_csv(DATA/'data_mutations.txt',sep='\t',comment='#',usecols=['Hugo_Symbol','Tumor_Sample_Barcode','Variant_Classification'])
 m=raw[raw.Hugo_Symbol.isin(PROTOCOL['genes'])].rename(columns={'Hugo_Symbol':'gene','Tumor_Sample_Barcode':'sample','Variant_Classification':'mutation_type'}).copy();m['cohort']='msk'
 m=m[['cohort','sample','gene','mutation_type']];m.to_csv(DATA/'normalized_mutations.csv',index=False)
 assert set(m['sample'])<=set(clinical.SAMPLE_ID)
 base=pd.DataFrame({'cohort':'msk','sample':clinical.SAMPLE_ID,'patient':clinical.PATIENT_ID,
                    'primary':clinical.SAMPLE_TYPE.eq('Primary'),'sample_type':clinical.SAMPLE_TYPE.fillna('Unspecified')})
 specs=[];families=[];axes=[];pairs=[];plans=[];paths=[];eligibility=[];fixed=[];graphs={};counts={}
 for gene in PROTOCOL['genes']:
  s=base.copy();s['profiled']=[gene in panel_genes[matrix.loc[x,'mutations']] for x in s['sample']]
  eligibility.extend({'sample':x,'gene':gene,'panel':matrix.loc[x,'mutations'],'profiled':bool(p)} for x,p in zip(s['sample'],s['profiled']))
  pe=PandasEngine(s,m);sq=SQLiteEngine(s,m);rs=[]
  for index,bits in enumerate(ASSIGNMENTS):
   spec=Spec.from_bits('msk',gene,bits);a=pe.execute(spec);b=sq.execute(spec);assert a==b;rs.append(a)
  sq.connection.close();model=CompletionModel(s,m,'msk',gene,rs)
  obs=model.observed_graph();coupled=model.certificate_graph();box=model.independent_interval_graph();merged=model.independent_interval_graph(True)
  assert np.array_equal(obs,coupled) if model.unprofiled_samples==0 else not (obs&~coupled).any()
  graphs[gene]=coupled;counts[gene]=np.stack([model.observed_counts,model.denominators]);av,an=model.absolute_certificate();rlo,rhi=model.relative_certificate();ba,br=model.independent_interval_certificate()
  planner=GraphPlanner(coupled);fm=fixed_question_incidence(planner).sum(axis=2).mean(axis=1);fixed.append(fm)
  value,first=planner.solve(planner.root);expected,maxcost,_=planner.policy_costs();assert abs(value-expected)<1e-12
  obsp=GraphPlanner(obs);obsq=obsp.solve(obsp.root)[0];guard=(model.upper_counts-model.lower_counts)*20>model.denominators
  family={'cohort':'MSK-IMPACT','gene':gene,'observed_sensitive':bool(obs.any()),'coupled_sensitive':bool(coupled.any()),
          'observed_absolute_pp':100*(model.observed.max()-model.observed.min()),'coupled_absolute_pp':av.max(),
          'unprofiled_samples':model.unprofiled_samples,'max_unprofiled_per_patient':model.max_unprofiled_per_patient,
          'observed_expected_questions':obsq,'coupled_expected_questions':value,'coupled_max_questions':maxcost,
          'best_family_fixed':fm.min(),'point_unidentified_specs':guard.sum(),
          'reference_unit_absolute_pp':av[0,8], 'reference_unit_relative':rhi[0,8]}
  families.append(family)
  plans.append({'gene':gene,'exact_expected':value,'exact_max':maxcost,'minimax_mean':planner.policy_costs('minimax')[0],
                'minimax_worst':planner.policy_costs('minimax')[1],'greedy_mean':planner.policy_costs('edge_greedy')[0],
                'entropy_mean':planner.policy_costs('entropy')[0],'best_family_fixed':fm.min()})
  for i,spec in enumerate(model.specs):
   specs.append({'index':i,**spec.__dict__,'positive':int(model.observed_counts[i]),'denominator':int(model.denominators[i]),
                 'profiled_units':rs[i].profiled_units,'lower':model.lower[i],'upper':model.upper[i],
                 'point_error_budget_2_5pp_unavailable':bool(guard[i])})
   cost,fields,state=planner.walk(i);assert planner.stable(state)
   paths.append({'gene':gene,'truth_index':i,'questions':cost,'fields':';'.join(FIELDS[x] for x in fields),'remaining_candidates':len(state),'terminal_verified':True})
   for j in range(i+1,16):pairs.append({'gene':gene,'i':i,'j':j,'coupled_absolute_pp':av[i,j],'coupled_relative_lower':rlo[i,j],'coupled_relative_upper':rhi[i,j],'observed_sensitive':bool(obs[i,j]),'coupled_sensitive':bool(coupled[i,j])})
  for f in range(4):
   j=1<<(3-f);axes.append({'gene':gene,'field':FIELDS[f],'observed_sensitive':bool(obs[0,j]),'coupled_sensitive':bool(coupled[0,j]),'absolute_pp':av[0,j],'relative':rhi[0,j]})
  print(gene,model.unprofiled_samples,value,flush=True)
 ff=pd.DataFrame(families);aa=pd.DataFrame(axes);pp=pd.DataFrame(pairs);pl=pd.DataFrame(plans)
 for name,rows in [('interpretations',specs),('families',families),('single_axis',axes),('pairs',pairs),('uniform_planning',plans),('policy_paths',paths),('gene_eligibility',eligibility)]:pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
 mat=np.stack(fixed);mean=mat.mean(axis=0);best=int(mean.argmin());ff['best_global_fixed_questions']=mat[:,best];ff.to_csv(OUT/'families.csv',index=False)
 pd.DataFrame([{'order':';'.join(FIELDS[x] for x in order),'mean_questions':v} for order,v in zip(ORDERS,mean)]).to_csv(OUT/'fixed_order_summary.csv',index=False)
 np.savez_compressed(OUT/'graphs.npz',**graphs);np.savez_compressed(OUT/'counts.npz',**counts)
 multiplicity=clinical.groupby('PATIENT_ID').size()
 summary={'study':'msk_impact_2017','commit':manifest['commit'],'created_at_utc':datetime.now(timezone.utc).isoformat(),
  'samples':len(clinical),'patients':clinical.PATIENT_ID.nunique(),'patients_with_repeated_specimens':int(multiplicity.gt(1).sum()),
  'max_specimens_per_patient':int(multiplicity.max()),'primary_samples':int(base.primary.sum()),'unspecified_sample_type':int(clinical.SAMPLE_TYPE.isna().sum()),
  'mutation_records_all_genes':len(raw),'panel_mutation_records':len(m),'panel_usage':matrix.mutations.value_counts().to_dict(),
  'gene_panel_missing_specimen_counts':ff.set_index('gene').unprofiled_samples.to_dict(),'families':len(ff),'interpretations':len(specs),'pairs':len(pp),
  'engine_matches':len(specs),'observed_sensitive_families':int(ff.observed_sensitive.sum()),'coupled_sensitive_families':int(ff.coupled_sensitive.sum()),
  'sensitive_axis_counts':aa.groupby('field').observed_sensitive.sum().to_dict(),'point_guard_unidentified_specs':int(ff.point_unidentified_specs.sum()),
  'optimal_mean_questions':float(ff.coupled_expected_questions.mean()),'optimal_max_questions':float(ff.coupled_max_questions.max()),
  'best_family_fixed_mean':float(ff.best_family_fixed.mean()),'best_global_fixed_mean':float(mean[best]),
  'best_global_order':';'.join(FIELDS[x] for x in ORDERS[best]),'greedy_mean':float(pl.greedy_mean.mean()),'entropy_mean':float(pl.entropy_mean.mean()),
  'maximum_reference_unit_absolute_pp':float(ff.reference_unit_absolute_pp.max()),'maximum_reference_unit_relative':float(ff.reference_unit_relative.max()),
  'runtime_seconds':time.perf_counter()-t0,
  'scope':'Separate mixed-cancer clinical targeted-assay stress test, not pooled TCGA, an independent portal, or a clinical validation. All chosen genes happen to be in both panels; this is a negative missing-eligibility finding.'}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item())+'\n');print(json.dumps(summary,indent=2,default=lambda x:x.item()),flush=True)

if __name__=='__main__':main()
