"""Keep manuscript numeric macros and cohort table aligned with computed CSVs."""
import json
import pandas as pd
from engine import ROOT


def main():
    r,m=ROOT/"results",ROOT/"manuscript"
    s=json.loads((r/"summary.json").read_text())
    cohorts=pd.read_csv(r/"cohort_summary.csv")
    macros={"SampleCount":f"{s['samples']:,}","PatientCount":f"{s['patients']:,}",
            "ProfiledSampleCount":f"{s['profiled_samples']:,}",
            "PrimarySampleCount":f"{int(cohorts.primary_samples.sum()):,}",
            "InterpretationCount":f"{s['interpretations']:,}",
            "ContrastCount":f"{s['single_axis_contrasts']:,}",
            "SensitiveContrastCount":s["single_axis_sensitive"],
            "SensitiveFamilyCount":s["full_sensitive_families"],
            "OptimalQuestions":f"{s['optimal_mean_questions']:.3f}",
            "FixedQuestions":f"{s['best_global_fixed_order_mean']:.3f}",
            "FixedSaving":f"{100*s['saving_vs_best_global_fixed_order']:.1f}"}
    (m/"results_macros.tex").write_text("\n".join("\\newcommand{\\"+k+"}{"+str(v)+"}" for k,v in macros.items())+"\n")
    lines=[r"\begin{tabular}{@{}lrrrr@{}}",r"\toprule",
           r"Cohort & Samples & Patients & P & Primary\\",r"\midrule"]
    for _,row in cohorts.iterrows():
        code=row["cohort"]+("*" if row["group"]=="additional" else "")
        lines.append(f"{code} & {row['samples']:,} & {row['patients']:,} & {row['profiled_samples']:,} & {row['primary_samples']:,} " + r"\\")
    lines += [r"\midrule",r"Total & \SampleCount{} & \PatientCount{} & \ProfiledSampleCount{} & \PrimarySampleCount{}\\",
              r"\bottomrule",r"\end{tabular}"]
    (m/"cohort_table.tex").write_text("\n".join(lines)+"\n")
    mechanisms=pd.read_csv(r/"mechanism_summary.csv")
    labels={"analysis_unit":"Analysis unit", "assay_denominator":"Assay denominator",
            "sample_context":"Specimen context", "mutation_scope":"Mutation scope"}
    lines=[r"\begin{tabular}{@{}lrrr@{}}",r"\toprule",
           r"Field & Original 8 & Additional 4 & All 12\\",r"\midrule"]
    for field in ("analysis_unit","assay_denominator","sample_context","mutation_scope"):
        rows=mechanisms.loc[mechanisms.field==field].set_index("group")
        entries=[]
        for group in ("original","additional"):
            row=rows.loc[group]
            entries.append(f"{int(row['sum'])}/{int(row['count'])} ({100*row['mean']:.1f}\\%)")
        total=int(rows['sum'].sum())
        entries.append(f"{total}/360 ({100*total/360:.1f}\\%)")
        lines.append(labels[field]+" & "+" & ".join(entries)+r"\\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (m/"mechanism_table.tex").write_text("\n".join(lines)+"\n")



if __name__=="__main__":
    main()
