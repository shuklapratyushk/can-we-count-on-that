"""Typeset complete measured runtime configurations from retained CSV rows."""
import pandas as pd
from engine import ROOT


def main():
    registry=pd.read_csv(ROOT/'scale_results/registry_scaling.csv')
    rows=[r'\begin{tabular}{llrrr}',r'\toprule',r'Axis & Cohorts $\times$ genes & Families & Median (s) & Min--max (s)\\',r'\midrule']
    for _,r in registry.sort_values(['axis','families']).iterrows():
        rows.append(f"{r.axis.title()} & {r.cohorts}$\\times${r.genes} & {r.families} & {r.median_ms/1000:.3f} & {r.min_ms/1000:.3f}--{r.max_ms/1000:.3f}\\\\")
    rows.extend([r'\bottomrule',r'\end{tabular}'])
    (ROOT/'manuscript/scale_registry_table.tex').write_text('\n'.join(rows)+'\n')
    rows=[r'\begin{tabular}{rrrrr}',r'\toprule',r'Unknowns & Local states & Table (MiB) & Build (ms) & Support (ms)\\',r'\midrule']
    for _,r in pd.read_csv(ROOT/'scale_results/local_block_scaling.csv').iterrows():
        rows.append(f"{int(r.unknown_specimens)} & {int(r.states):,} & {r.table_bytes/2**20:.4f} & {r.table_construction_ms:.3f} & {r.median_ms:.3f}\\\\")
    rows.extend([r'\bottomrule',r'\end{tabular}'])
    (ROOT/'manuscript/scale_local_table.tex').write_text('\n'.join(rows)+'\n')
    print('Complete scalability tables written from executed results.')


if __name__=='__main__':main()
