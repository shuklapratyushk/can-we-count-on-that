from pathlib import Path
import pandas as pd
from engine import ROOT,PROTOCOL
c=pd.read_csv(ROOT/'results/cohort_summary.csv').set_index('cohort');f=pd.read_csv(ROOT/'completion_results/families.csv')
rows=[r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Cohort & Specimens & Patients & Unprofiled & Sensitive & Box-only & Beyond equiv.\\',r'\midrule']
for co in [x.upper() for x in PROTOCOL['original_cohorts']+PROTOCOL['additional_cohorts']]:
 s=c.loc[co];v=f[f.cohort.eq(co)];rows.append(f'{co} & {s.samples:,} & {s.patients:,} & {s.samples-s.profiled_samples:,} & {v.coupled_sensitive.sum()}/30 & {v.box_only_pairs.sum()} & {v.merged_box_only_pairs.sum()}'+r'\\')
rows += [r'\bottomrule',r'\end{tabular}']
(ROOT/'manuscript/completion_cohorts.tex').write_text('\n'.join(rows)+'\n')
f=pd.read_csv(ROOT/'msk_results/families.csv')
rows=[r'\begin{longtable}{lrrrr}',r'\toprule',r'Gene & Unit flip (pp) & Unit flip (relative) & Semantic diameter (pp) & Mean questions\\',r'\midrule',r'\endhead']
for _,v in f.iterrows():rows.append(f'{v.gene} & {v.reference_unit_absolute_pp:.3f} & {v.reference_unit_relative:.4f} & {v.coupled_absolute_pp:.2f} & {v.coupled_expected_questions:.3f}'+r'\\')
rows += [r'\bottomrule',r'\end{longtable}']
(ROOT/'manuscript/msk_gene_table.tex').write_text('\n'.join(rows)+'\n')
print('New tables generated directly from result registries.')
